//! Memory-mapped flat vector index with symmetric i8 quantization.
//! Record layout: [chunk_rowid u64][scale f32][dim × i8]. The mmap keeps resident
//! RAM near zero (page cache serves scans) and top-k runs rayon-parallel with
//! integer i8×i8 dots that the compiler auto-vectorizes with NEON.

use anyhow::{anyhow, Result};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: u32 = 0x4950_5631; // "IPV1"
const HEADER_BYTES: usize = 64;
const MAX_DIM: usize = 1024;
/// Compaction gates: never churn small stores, and only rewrite once the
/// recycled pool holds a real share of the file.
const COMPACT_MIN_FREE_SLOTS: usize = 1024;
const COMPACT_FRAGMENTATION_SHARE: f64 = 0.25;

#[derive(Debug, Clone, PartialEq)]
struct StoreHeader {
    dim: u16,
    model_hash: u64,
    count: u32,
}

pub struct VectorStore {
    file: File,
    mmap: Mmap,
    free_path: PathBuf,
    free_slots: Vec<i64>,
    header: StoreHeader,
    dim: usize,
}

fn record_bytes(dim: usize) -> usize {
    8 + 4 + dim // chunk_rowid + scale + quantized vector
}

/// Stable hash of the embedder identity; a change forces a rebuild.
pub fn model_hash(model_id: &str) -> u64 {
    model_id.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

impl VectorStore {
    /// Opens or creates the store. `compatible` reports whether existing vectors
    /// were produced by the same embedder; if not, the engine wipes and reindexes.
    pub fn open(dir: &Path, dim: usize, model_id: &str) -> Result<(Self, bool)> {
        if dim > MAX_DIM {
            return Err(anyhow!("embedding dim {dim} exceeds {MAX_DIM}"));
        }
        std::fs::create_dir_all(dir)?;
        let path = dir.join("vectors.bin");
        let free_path = dir.join("free-slots.bin");
        let expected = StoreHeader { dim: dim as u16, model_hash: model_hash(model_id), count: 0 };
        let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;
        let length = file.metadata()?.len() as usize;
        let (header, compatible) = if length == 0 {
            let mut bytes = [0u8; HEADER_BYTES];
            write_header(&mut bytes, &expected);
            file.write_all(&bytes)?;
            (expected, true)
        } else {
            let mut bytes = [0u8; HEADER_BYTES];
            file.read_exact(&mut bytes)?;
            let (existing, valid) = read_header(&bytes);
            if !valid {
                return Err(anyhow!("vectors.bin has an unrecognized header"));
            }
            let compatible = existing.dim == expected.dim && existing.model_hash == expected.model_hash;
            (existing, compatible)
        };
        // Prefault the mapping so the first query scans resident pages, not
        // cold disk (harmless on small stores, decisive at multi-GB scale).
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        mmap.advise(memmap2::Advice::WillNeed).ok();
        let mut free_slots = Vec::new();
        if free_path.exists() {
            let mut bytes = Vec::new();
            File::open(&free_path)?.read_to_end(&mut bytes)?;
            free_slots = bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
        }
        let store = Self { file, mmap, free_path, free_slots, header, dim };
        Ok((store, compatible))
    }

    pub fn count(&self) -> u32 {
        self.header.count
    }

    /// Vectors actually reachable by a scan: allocated slots minus the
    /// tombstoned free pool. This is the number that must drop when files
    /// are deleted (raw `count` only shrinks for trailing-slot truncation).
    pub fn live_count(&self) -> u32 {
        self.header.count.saturating_sub(self.free_slots.len() as u32)
    }

    /// Appends quantized vectors, reusing freed slots; returns the slot per input.
    pub fn append_batch(&mut self, chunk_rowids: &[i64], vectors: &[Vec<f32>]) -> Result<Vec<i64>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        let record = record_bytes(self.dim);
        let mut file_len = self.file.metadata()?.len() as usize;
        let mut slots = Vec::with_capacity(vectors.len());
        for (vector, &chunk_rowid) in vectors.iter().zip(chunk_rowids) {
            if vector.len() != self.dim {
                return Err(anyhow!("vector dim {} != store dim {}", vector.len(), self.dim));
            }
            let slot = match self.free_slots.pop() {
                Some(slot) if (slot as usize) < self.header.count as usize => slot,
                _ => {
                    let slot = ((file_len - HEADER_BYTES) / record) as i64;
                    file_len += record;
                    self.file.set_len(file_len as u64)?;
                    self.header.count += 1;
                    slot
                }
            };
            let mut record_bytes_buf = [0u8; 8 + 4 + MAX_DIM];
            encode_record(&mut record_bytes_buf, chunk_rowid, vector);
            self.file.seek(SeekFrom::Start((HEADER_BYTES + slot as usize * record) as u64))?;
            self.file.write_all(&record_bytes_buf[..record])?;
            slots.push(slot);
        }
        self.persist_header()?;
        self.persist_free_slots()?;
        self.remap()?;
        Ok(slots)
    }

    /// Releases slots for reuse; trailing slots truncate straight off the file.
    pub fn free(&mut self, slots: &[i64]) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }
        let record = record_bytes(self.dim);
        let mut sorted: Vec<i64> = slots.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        sorted.dedup();
        let mut count = self.header.count as i64;
        let mut reusable = Vec::new();
        for slot in sorted {
            if slot >= 0 && slot < count {
                if slot == count - 1 {
                    count -= 1; // trailing: shrink
                } else {
                    reusable.push(slot);
                }
            }
        }
        self.header.count = count as u32;
        self.file.set_len((HEADER_BYTES + count as usize * record) as u64)?;
        // Tombstone reused-later middle slots so scans skip stale rowids.
        let mut tombstone = [0u8; 8 + 4 + MAX_DIM];
        tombstone[..8].copy_from_slice(&(-1i64).to_le_bytes());
        for &slot in &reusable {
            self.file.seek(SeekFrom::Start((HEADER_BYTES + slot as usize * record) as u64))?;
            self.file.write_all(&tombstone[..record])?;
        }
        self.free_slots.extend(reusable);
        self.persist_header()?;
        self.persist_free_slots()?;
        self.remap()?;
        Ok(())
    }

    /// True once the recycled pool is big enough that rewriting the store
    /// beats holding the tombstoned space hostage (both a floor, so tiny
    /// stores never churn, and a fragmentation share).
    pub fn needs_compaction(&self) -> bool {
        self.free_slots.len() >= COMPACT_MIN_FREE_SLOTS
            && self.free_slots.len() as f64
                >= self.header.count as f64 * COMPACT_FRAGMENTATION_SHARE
    }

    /// Rewrites live records contiguously and truncates the file, returning
    /// the old→new slot moves in ascending order (every move targets a lower
    /// slot) so the caller can fix up SQLite references. Slots are file
    /// offsets, not rowids: the references live in chunks.vec_slot /
    /// file_vectors.vec_slot and must be remapped with the catalog.
    pub fn compact(&mut self) -> Result<Vec<(i64, i64)>> {
        if self.free_slots.is_empty() {
            return Ok(Vec::new());
        }
        let record = record_bytes(self.dim);
        let freed: std::collections::HashSet<i64> = self.free_slots.iter().copied().collect();
        let mut moves = Vec::new();
        let mut buffer = [0u8; 8 + 4 + MAX_DIM];
        let mut write = 0i64;
        for slot in 0..self.header.count as i64 {
            if freed.contains(&slot) {
                continue;
            }
            if slot != write {
                // Forward in-place move: the write offset never passes the
                // read offset, so the source record is always intact.
                self.file
                    .seek(SeekFrom::Start((HEADER_BYTES + slot as usize * record) as u64))?;
                self.file.read_exact(&mut buffer[..record])?;
                self.file
                    .seek(SeekFrom::Start((HEADER_BYTES + write as usize * record) as u64))?;
                self.file.write_all(&buffer[..record])?;
                moves.push((slot, write));
            }
            write += 1;
        }
        self.header.count = write as u32;
        self.file.set_len((HEADER_BYTES + write as usize * record) as u64)?;
        self.free_slots.clear();
        self.persist_header()?;
        self.persist_free_slots()?;
        self.remap()?;
        Ok(moves)
    }

    /// Parallel approximate-cosine top-k over all records.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        let dim = self.dim;
        if self.header.count == 0 || query.len() != dim || k == 0 {
            return Vec::new();
        }
        // Quantize the query symmetrically; the integer dot is exact up to that scaling.
        let query_scale = 127.0 / query.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        let quantized_query: Vec<i8> = query.iter().map(|v| (v * query_scale).round().clamp(-127.0, 127.0) as i8).collect();
        let query_descale = 1.0 / query_scale;
        let record = record_bytes(dim);
        let per_shard = 4096usize;
        let shards = (self.header.count as usize).div_ceil(per_shard);
        let partial: Vec<Vec<(i64, f32)>> = (0..shards)
            .into_par_iter()
            .map(|shard| {
                let start = shard * per_shard;
                let end = ((shard + 1) * per_shard).min(self.header.count as usize);
                let mut best: Vec<(i64, f32)> = Vec::with_capacity(k.min(64));
                for slot in start..end {
                    let offset = HEADER_BYTES + slot * record;
                    let bytes = &self.mmap[offset..offset + record];
                    let chunk_rowid = i64::from_le_bytes(bytes[..8].try_into().unwrap());
                    if chunk_rowid < 0 {
                        continue; // tombstoned slot
                    }
                    let scale = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
                    // Records hold i8 as raw bytes: reinterpret signed before widening.
                    let mut dot = 0i32;
                    for (quantized, &query_value) in bytes[12..].iter().zip(&quantized_query) {
                        dot += (*quantized as i8) as i32 * query_value as i32;
                    }
                    let score = scale * query_descale * dot as f32;
                    if best.len() < k {
                        best.push((chunk_rowid, score));
                        best.sort_by(|a, b| b.1.total_cmp(&a.1));
                    } else if score > best[k - 1].1 {
                        best[k - 1] = (chunk_rowid, score);
                        best.sort_by(|a, b| b.1.total_cmp(&a.1));
                    }
                }
                best
            })
            .collect();
        let mut merged: Vec<(i64, f32)> = partial.into_iter().flatten().collect();
        merged.sort_by(|a, b| b.1.total_cmp(&a.1));
        merged.dedup_by_key(|(chunk_rowid, _)| *chunk_rowid);
        merged.truncate(k);
        merged
    }

    /// Wipes all vectors (used when the embedder model changes).
    pub fn clear(&mut self) -> Result<()> {
        self.header.count = 0;
        self.free_slots.clear();
        let mut bytes = [0u8; HEADER_BYTES];
        write_header(&mut bytes, &self.header);
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0u64))?;
        self.file.write_all(&bytes)?;
        self.persist_free_slots()?;
        self.remap()?;
        Ok(())
    }

    fn persist_header(&mut self) -> Result<()> {
        let mut bytes = [0u8; HEADER_BYTES];
        write_header(&mut bytes, &self.header);
        self.file.seek(SeekFrom::Start(0u64))?;
        self.file.write_all(&bytes)?;
        Ok(())
    }

    fn persist_free_slots(&self) -> Result<()> {
        let mut bytes = Vec::with_capacity(self.free_slots.len() * 8);
        for slot in &self.free_slots {
            bytes.extend_from_slice(&slot.to_le_bytes());
        }
        std::fs::write(&self.free_path, bytes)?;
        Ok(())
    }

    fn remap(&mut self) -> Result<()> {
        self.file.flush()?;
        self.mmap = unsafe { MmapOptions::new().map(&self.file)? };
        Ok(())
    }
}

fn encode_record(buffer: &mut [u8; 8 + 4 + MAX_DIM], chunk_rowid: i64, vector: &[f32]) {
    // Symmetric quantization: v ≈ q * (max_abs / 127).
    let max_abs = vector.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
    let scale = max_abs / 127.0;
    buffer[..8].copy_from_slice(&chunk_rowid.to_le_bytes());
    buffer[8..12].copy_from_slice(&scale.to_le_bytes());
    for (slot, value) in buffer[12..12 + vector.len()].iter_mut().zip(vector) {
        *slot = (value / scale).round().clamp(-127.0, 127.0) as i8 as u8;
    }
}

fn write_header(bytes: &mut [u8; HEADER_BYTES], header: &StoreHeader) {
    bytes[..4].copy_from_slice(&MAGIC.to_le_bytes());
    bytes[4..6].copy_from_slice(&header.dim.to_le_bytes());
    bytes[8..16].copy_from_slice(&header.model_hash.to_le_bytes());
    bytes[16..20].copy_from_slice(&header.count.to_le_bytes());
}

fn read_header(bytes: &[u8; HEADER_BYTES]) -> (StoreHeader, bool) {
    let magic = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    let header = StoreHeader {
        dim: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
        model_hash: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
    };
    (header, magic == MAGIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_scan_free_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ipic-vec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dimension = 64; // high enough that random vectors never collide with self-match
        let (mut store, compatible) = VectorStore::open(&dir, dimension, "test-model").unwrap();
        assert!(compatible);
        let vectors: Vec<Vec<f32>> = (0..100u32)
            .map(|i| {
                // Well-mixed pseudo-random direction per vector (murmur-style finalizer).
                let mut vector: Vec<f32> = (0..dimension)
                    .map(|j| {
                        let mut hash = i
                            .wrapping_mul(2654435761)
                            .wrapping_add((j as u32).wrapping_mul(40503))
                            .wrapping_add(0x9e37_7b9b);
                        hash ^= hash >> 16;
                        hash = hash.wrapping_mul(0x7feb_352d);
                        hash ^= hash >> 15;
                        hash = hash.wrapping_mul(0x846c_a68b);
                        hash ^= hash >> 16;
                        (hash % 1001) as f32 - 500.0
                    })
                    .collect();
                let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
                vector.iter_mut().for_each(|value| *value /= norm);
                vector
            })
            .collect();
        let rowids: Vec<i64> = (0..100).map(|i| i as i64 + 1).collect();
        let slots = store.append_batch(&rowids, &vectors).unwrap();
        assert_eq!(slots.len(), 100);
        let hits = store.top_k(&vectors[42], 5);
        assert_eq!(hits[0].0, 43, "self-match should rank first");
        // Free a middle slot and re-append: reuse must not corrupt rowid lookup.
        store.free(&[slots[10]]).unwrap();
        // A middle-slot free tombstones rather than truncates: raw count holds,
        // live count drops — deletions must be observable, not silently recycled.
        assert_eq!(store.count(), 100);
        assert_eq!(store.live_count(), 99);
        let new_slots = store.append_batch(&[999], &[vectors[0].clone()]).unwrap();
        assert_eq!(new_slots[0], slots[10]);
        assert_eq!(store.live_count(), 100);
        assert_eq!(store.top_k(&vectors[42], 200).iter().filter(|(r, _)| *r == 11).count(), 0);
        assert!(store.top_k(&vectors[0], 3).iter().any(|(r, _)| *r == 999));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compaction_reclaims_space_and_keeps_lookups() {
        let dir = std::env::temp_dir().join(format!("ipic-vec-compact-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dimension = 64;
        let (mut store, _) = VectorStore::open(&dir, dimension, "test-model").unwrap();
        let vectors: Vec<Vec<f32>> = (0..100u32)
            .map(|i| {
                let mut vector: Vec<f32> = (0..dimension)
                    .map(|j| {
                        let mut hash = i
                            .wrapping_mul(2654435761)
                            .wrapping_add((j as u32).wrapping_mul(40503))
                            .wrapping_add(0x9e37_7b9b);
                        hash ^= hash >> 16;
                        hash = hash.wrapping_mul(0x7feb_352d);
                        hash ^= hash >> 15;
                        hash = hash.wrapping_mul(0x846c_a68b);
                        hash ^= hash >> 16;
                        (hash % 1001) as f32 - 500.0
                    })
                    .collect();
                let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
                vector.iter_mut().for_each(|value| *value /= norm);
                vector
            })
            .collect();
        let rowids: Vec<i64> = (1..=100i64).collect();
        let slots = store.append_batch(&rowids, &vectors).unwrap();
        let size_before = std::fs::metadata(dir.join("vectors.bin")).unwrap().len();

        // Free a middle band: all tombstoned, none eligible for truncation.
        let victims: Vec<i64> = slots[20..50].to_vec();
        store.free(&victims).unwrap();
        assert_eq!(store.count(), 100, "middle frees tombstone, never truncate");

        let moves = store.compact().unwrap();
        assert_eq!(store.count(), 70);
        assert_eq!(store.live_count(), 70);
        let record = (8 + 4 + dimension) as u64;
        let size_after = std::fs::metadata(dir.join("vectors.bin")).unwrap().len();
        assert_eq!(size_after, size_before - 30 * record, "disk must shrink by exactly the freed band");
        assert_eq!(moves.len(), 50, "the 50 live records above the band shift down; the 20 below stay");
        assert!(
            moves.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "moves arrive ascending for safe ordered remap"
        );
        assert!(moves.iter().all(|(old, new)| new < old), "every move targets a lower slot");

        // Survivors stay findable by their rowids; victims are gone.
        assert!(store.top_k(&vectors[0], 3).iter().any(|(r, _)| *r == 1));
        assert!(store.top_k(&vectors[99], 3).iter().any(|(r, _)| *r == 100));
        assert!(!store.top_k(&vectors[35], 200).iter().any(|(r, _)| *r == rowids[35]));

        // Post-compaction appends extend the tail; nothing is recycled.
        let new_slots = store.append_batch(&[9999], &[vectors[0].clone()]).unwrap();
        assert_eq!(new_slots[0], 70);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

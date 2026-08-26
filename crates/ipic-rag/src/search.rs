//! Hybrid retrieval across shards: dense text vectors, CLIP pixel vectors,
//! FTS5 BM25 and filename matches, collected per shard in parallel and fused
//! with weighted reciprocal rank fusion over globally assigned ranks.

use crate::shard::Shard;
use anyhow::Result;
use ipic_core::FileRow;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;

const VECTOR_WEIGHT: f32 = 1.0;
pub const VISION_WEIGHT: f32 = 0.9;
const KEYWORD_WEIGHT: f32 = 0.7;
const NAME_WEIGHT: f32 = 0.5;
pub const RRF_CONSTANT: f32 = 60.0;
const LANE_COUNT: usize = 4;

/// Which retrieval lanes contributed to a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatchSources {
    pub semantic: bool,
    pub keyword: bool,
    pub filename: bool,
    /// CLIP content similarity between the query and the image pixels.
    pub vision: bool,
    /// Audio-to-audio similarity against the acoustic fingerprint store.
    pub acoustic: bool,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub file: FileRow,
    pub path: String,
    pub snippet: String,
    pub score: f32,
    pub sources: MatchSources,
}

#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub hits: Vec<SearchHit>,
    pub elapsed_millis: f32,
    /// Transcript of an audio query, when the query was spoken.
    pub interpreted_query: Option<String>,
    pub vector_count: u32,
}

/// One shard's per-lane candidate lists, pre-hydration.
pub struct ShardCandidates {
    pub shard_index: usize,
    semantic: Vec<Candidate>,
    keyword: Vec<Candidate>,
    filename: Vec<Candidate>,
    vision: Vec<Candidate>,
}

struct Candidate {
    file_id: i64,
    snippet: String,
    score: f64,
}

/// Collects every lane's candidates from one shard. Store locks are held only
/// for the vector scans; hydration happens after fusion.
pub fn collect_shard_candidates(
    shard: &Shard,
    shard_index: usize,
    text_query: &[f32],
    vision_query: Option<&[f32]>,
    raw_query: &str,
    depth: usize,
) -> Result<ShardCandidates> {
    let connection = shard.catalog.reader()?;
    let mut semantic = Vec::new();
    for (chunk_rowid, similarity) in
        shard.text_store.lock().unwrap().top_k(text_query, depth)
    {
        if similarity <= 0.0 {
            break;
        }
        // Stale vector slots (chunk deleted) resolve to None: harmless.
        if let Ok(Some((file_id, snippet))) = fetch_chunk(&connection, chunk_rowid) {
            semantic.push(Candidate { file_id, snippet, score: similarity as f64 });
        }
    }
    let keyword = keyword_search(&connection, raw_query, depth)?
        .into_iter()
        .map(|(file_id, snippet, rank)| Candidate { file_id, snippet, score: rank })
        .collect();
    let filename = shard
        .catalog
        .name_search(&connection, raw_query, (depth / 2) as i64)?
        .into_iter()
        .map(|(file_row, _path)| Candidate { file_id: file_row.id, snippet: String::new(), score: 0.0 })
        .collect();
    let vision = match vision_query {
        Some(vision_query) => shard
            .image_store
            .lock()
            .unwrap()
            .top_k(vision_query, depth)
            .into_iter()
            .filter(|(_, similarity)| *similarity > 0.0)
            .map(|(file_id, similarity)| Candidate {
                file_id,
                snippet: String::new(),
                score: similarity as f64,
            })
            .collect(),
        None => Vec::new(),
    };
    Ok(ShardCandidates { shard_index, semantic, keyword, filename, vision })
}

/// Fuses per-shard candidates into globally ranked hits and hydrates them
/// against their owning shards.
pub fn fuse_and_hydrate(
    shards: &[std::sync::Arc<Shard>],
    shards_candidates: Vec<ShardCandidates>,
    limit: usize,
    vector_count: u32,
    elapsed: std::time::Duration,
) -> Result<SearchOutcome> {
    // Global per-lane rank assignment: sort each lane by its own score across
    // shards (filename keeps per-shard order, interleaved round-robin).
    let mut semantic = Vec::new();
    let mut keyword = Vec::new();
    let mut vision = Vec::new();
    let mut filename_lanes: Vec<Vec<(usize, i64)>> = Vec::new();
    for candidates in shards_candidates {
        let shard_index = candidates.shard_index;
        semantic.extend(candidates.semantic.into_iter().map(|c| (shard_index, c)));
        keyword.extend(candidates.keyword.into_iter().map(|c| (shard_index, c)));
        vision.extend(candidates.vision.into_iter().map(|c| (shard_index, c)));
        filename_lanes.push(candidates.filename.into_iter().map(|c| (shard_index, c.file_id)).collect());
    }
    semantic.sort_by(|left, right| right.1.score.total_cmp(&left.1.score));
    keyword.sort_by(|left, right| left.1.score.total_cmp(&right.1.score));
    vision.sort_by(|left, right| right.1.score.total_cmp(&left.1.score));
    let filename = interleave(filename_lanes);

    let mut aggregations: HashMap<(usize, i64), [usize; LANE_COUNT]> = HashMap::new();
    let mut snippets: HashMap<(usize, i64), String> = HashMap::new();
    let record = |aggregations: &mut HashMap<(usize, i64), [usize; LANE_COUNT]>,
                      snippets: &mut HashMap<(usize, i64), String>,
                      key: (usize, i64),
                      lane: usize,
                      rank: usize,
                      snippet: &str| {
        let ranks = aggregations.entry(key).or_insert([usize::MAX; LANE_COUNT]);
        if rank < ranks[lane] {
            ranks[lane] = rank;
        }
        if !snippet.is_empty() {
            snippets.entry(key).or_insert_with(|| snippet.to_string());
        }
    };
    for (rank, (shard_index, candidate)) in semantic.iter().enumerate() {
        record(&mut aggregations, &mut snippets, (*shard_index, candidate.file_id), 0, rank, &candidate.snippet);
    }
    for (rank, (shard_index, candidate)) in keyword.iter().enumerate() {
        record(&mut aggregations, &mut snippets, (*shard_index, candidate.file_id), 1, rank, &candidate.snippet);
    }
    for (rank, (shard_index, candidate)) in vision.iter().enumerate() {
        record(&mut aggregations, &mut snippets, (*shard_index, candidate.file_id), 3, rank, &candidate.snippet);
    }
    for (rank, (shard_index, file_id)) in filename.iter().enumerate() {
        record(&mut aggregations, &mut snippets, (*shard_index, *file_id), 2, rank, "");
    }

    let lane_weights = [VECTOR_WEIGHT, KEYWORD_WEIGHT, NAME_WEIGHT, VISION_WEIGHT];
    let mut fused: Vec<((usize, i64), f32, MatchSources)> = aggregations
        .into_iter()
        .map(|(key, ranks)| {
            let mut score = 0.0f32;
            let mut sources = MatchSources::default();
            for (lane, (&best_rank, &weight)) in ranks.iter().zip(&lane_weights).enumerate() {
                if best_rank < usize::MAX {
                    score += weight / (RRF_CONSTANT + best_rank as f32);
                    match lane {
                        0 => sources.semantic = true,
                        1 => sources.keyword = true,
                        2 => sources.filename = true,
                        _ => sources.vision = true,
                    }
                }
            }
            (key, score, sources)
        })
        .collect();
    fused.sort_by(|left, right| right.1.total_cmp(&left.1));
    fused.truncate(limit);

    // Hydrate per shard: file ids are shard-local by design.
    let mut wanted: HashMap<usize, Vec<i64>> = HashMap::new();
    for ((shard_index, file_id), _, _) in &fused {
        wanted.entry(*shard_index).or_default().push(*file_id);
    }
    let mut hydrated: HashMap<(usize, i64), (FileRow, String)> = HashMap::new();
    for (shard_index, file_ids) in wanted {
        let Some(shard) = shards.get(shard_index) else { continue };
        let connection = shard.catalog.reader()?;
        for (file_row, path) in shard.catalog.files_by_ids(&connection, &file_ids)? {
            hydrated.insert((shard_index, file_row.id), (file_row, path));
        }
    }

    let hits = fused
        .into_iter()
        .filter_map(|(key, score, sources)| {
            hydrated.get(&key).map(|(file_row, path)| SearchHit {
                file: file_row.clone(),
                path: path.clone(),
                snippet: snippets.get(&key).cloned().unwrap_or_default(),
                score,
                sources,
            })
        })
        .collect();
    Ok(SearchOutcome {
        hits,
        elapsed_millis: elapsed.as_secs_f32() * 1000.0,
        interpreted_query: None,
        vector_count,
    })
}

/// Fair cross-shard merge for scoreless lanes: one from each shard in turn.
fn interleave(lanes: Vec<Vec<(usize, i64)>>) -> Vec<(usize, i64)> {
    let mut merged = Vec::new();
    let mut longest = 0;
    for lane in &lanes {
        longest = longest.max(lane.len());
    }
    for position in 0..longest {
        for lane in &lanes {
            if let Some(entry) = lane.get(position) {
                merged.push(*entry);
            }
        }
    }
    merged
}

fn fetch_chunk(connection: &Connection, chunk_rowid: i64) -> Result<Option<(i64, String)>> {
    let row = connection
        .query_row(
            "SELECT file_id, text FROM chunks WHERE rowid = ?1",
            params![chunk_rowid],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?;
    Ok(row)
}

/// BM25 search returning (file_id, highlighted snippet, rank) — lower rank is
/// better, which keeps the global sort consistent across shards.
fn keyword_search(connection: &Connection, query: &str, limit: usize) -> Result<Vec<(i64, String, f64)>> {
    let match_expression = build_match_expression(query);
    if match_expression.is_empty() {
        return Ok(Vec::new());
    }
    let mut statement = connection.prepare(
        "SELECT file_id, snippet(chunks, 0, '', '', ' … ', 18) AS excerpt, rank
         FROM chunks WHERE chunks MATCH ?1 ORDER BY rank LIMIT ?2",
    )?;
    let rows = statement
        .query_map(params![match_expression, limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, f64>(2)?))
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

/// Safe FTS5 MATCH syntax: every word becomes a quoted phrase (no operators leak in).
pub fn build_match_expression(query: &str) -> String {
    let words: Vec<String> = query
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphanumeric))
        .take(12)
        .map(|word| format!("\"{}\"", word.replace('"', "\"\"")))
        .collect();
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_expression_quotes_words() {
        assert_eq!(build_match_expression("travel plans"), "\"travel\" \"plans\"");
        assert_eq!(build_match_expression(""), "");
    }

    #[test]
    fn fts5_is_available() {
        // Bundled SQLite must ship FTS5; fail loudly here if not.
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("CREATE VIRTUAL TABLE probe USING fts5(text);")
            .expect("FTS5 support missing from bundled SQLite");
    }

    #[test]
    fn interleave_round_robins_across_shards() {
        let lanes = vec![vec![(0, 1), (0, 2)], vec![(1, 10)], vec![]];
        assert_eq!(interleave(lanes), vec![(0, 1), (1, 10), (0, 2)]);
    }
}

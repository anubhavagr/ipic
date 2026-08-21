//! Hybrid retrieval: dense vectors (semantic) + FTS5 BM25 (keyword) + filename
//! match, fused with weighted reciprocal rank fusion. All local, all parallel.

use crate::embed::TextEmbedder;
use crate::vector_store::VectorStore;
use anyhow::Result;
use ipic_core::catalog::Catalog;
use ipic_core::FileRow;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::time::Instant;

const VECTOR_WEIGHT: f32 = 1.0;
const KEYWORD_WEIGHT: f32 = 0.7;
const NAME_WEIGHT: f32 = 0.5;
pub const RRF_CONSTANT: f32 = 60.0;

/// Which retrieval lanes contributed to a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatchSources {
    pub semantic: bool,
    pub keyword: bool,
    pub filename: bool,
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

/// Per-file fusion accumulator.
struct FileAggregation {
    best_rank_by_lane: [usize; 3], // semantic, keyword, filename
    best_snippet: String,
}

pub fn semantic_search(
    catalog: &Catalog,
    connection: &Connection,
    vector_store: &VectorStore,
    embedder: &dyn TextEmbedder,
    query: &str,
    limit: usize,
) -> Result<SearchOutcome> {
    let started = Instant::now();
    let candidate_depth = (limit * 4).clamp(48, 400);
    let mut aggregations: HashMap<i64, FileAggregation> = HashMap::new();

    // Lane 1 — dense semantic over the quantized mmap index.
    let embedded_query = embedder.embed_batch(&[query.to_string()])?;
    for (rank, (chunk_rowid, similarity)) in
        vector_store.top_k(&embedded_query[0], candidate_depth).into_iter().enumerate()
    {
        if similarity <= 0.0 {
            break;
        }
        // Stale vector slots (chunk deleted) resolve to None: harmless.
        if let Ok(Some((file_id, snippet))) = fetch_chunk(connection, chunk_rowid) {
            aggregate(&mut aggregations, file_id, snippet, 0, rank);
        }
    }

    // Lane 2 — FTS5 BM25 over chunk text, with a native snippet.
    if let Ok(keyword_rows) = keyword_search(connection, query, candidate_depth) {
        for (rank, (file_id, snippet)) in keyword_rows.into_iter().enumerate() {
            aggregate(&mut aggregations, file_id, snippet, 1, rank);
        }
    }

    // Lane 3 — filename match.
    for (rank, (file_row, _path)) in catalog.name_search(connection, query, candidate_depth as i64 / 2)?.into_iter().enumerate() {
        aggregate(&mut aggregations, file_row.id, String::new(), 2, rank);
    }

    // Fuse and hydrate.
    let lane_weights = [VECTOR_WEIGHT, KEYWORD_WEIGHT, NAME_WEIGHT];
    let mut fused: Vec<(i64, f32, String, MatchSources)> = aggregations
        .into_iter()
        .map(|(file_id, aggregation)| {
            let mut score = 0.0f32;
            let mut sources = MatchSources::default();
            for (lane_index, (&best_rank, &weight)) in
                aggregation.best_rank_by_lane.iter().zip(&lane_weights).enumerate()
            {
                if best_rank < usize::MAX {
                    score += weight / (RRF_CONSTANT + best_rank as f32);
                    match lane_index {
                        0 => sources.semantic = true,
                        1 => sources.keyword = true,
                        _ => sources.filename = true,
                    }
                }
            }
            (file_id, score, aggregation.best_snippet, sources)
        })
        .collect();
    fused.sort_by(|left, right| right.1.total_cmp(&left.1));
    fused.truncate(limit);

    let wanted_ids: Vec<i64> = fused.iter().map(|(file_id, _, _, _)| *file_id).collect();
    let hydrated: HashMap<i64, (FileRow, String)> = catalog
        .files_by_ids(connection, &wanted_ids)?
        .into_iter()
        .map(|(file_row, path)| (file_row.id, (file_row, path)))
        .collect();

    let hits = fused
        .into_iter()
        .filter_map(|(file_id, score, snippet, sources)| {
            hydrated.get(&file_id).map(|(file_row, path)| SearchHit {
                file: file_row.clone(),
                path: path.clone(),
                snippet,
                score,
                sources,
            })
        })
        .collect();
    Ok(SearchOutcome {
        hits,
        elapsed_millis: started.elapsed().as_secs_f32() * 1000.0,
        interpreted_query: None,
        vector_count: vector_store.count(),
    })
}

fn aggregate(
    aggregations: &mut HashMap<i64, FileAggregation>,
    file_id: i64,
    snippet: String,
    lane_index: usize,
    rank: usize,
) {
    let aggregation = aggregations.entry(file_id).or_insert_with(|| FileAggregation {
        best_rank_by_lane: [usize::MAX; 3],
        best_snippet: String::new(),
    });
    if rank < aggregation.best_rank_by_lane[lane_index] {
        aggregation.best_rank_by_lane[lane_index] = rank;
    }
    if !snippet.is_empty() && aggregation.best_snippet.is_empty() {
        aggregation.best_snippet = snippet;
    }
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

/// BM25 search returning (file_id, highlighted snippet) ranked best-first.
fn keyword_search(connection: &Connection, query: &str, limit: usize) -> Result<Vec<(i64, String)>> {
    let match_expression = build_match_expression(query);
    if match_expression.is_empty() {
        return Ok(Vec::new());
    }
    let mut statement = connection.prepare(
        "SELECT file_id, snippet(chunks, 0, '', '', ' … ', 18) AS excerpt
         FROM chunks WHERE chunks MATCH ?1 ORDER BY rank LIMIT ?2",
    )?;
    let rows = statement
        .query_map(params![match_expression, limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

/// Safe FTS5 MATCH syntax: every word becomes a quoted phrase (no operators leak in).
fn build_match_expression(query: &str) -> String {
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
}

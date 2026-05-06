use std::collections::{HashMap, HashSet};
use std::path::Path;

use rusqlite::{params, Connection, OpenFlags};
use serde_json::{json, Value};

use crate::text::normalize_arabic;

fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mut max_score = scores[0];
    for &s in scores {
        if s > max_score {
            max_score = s;
        }
    }
    let mut exps: Vec<f64> = Vec::with_capacity(scores.len());
    let mut denom = 0.0_f64;
    for &s in scores {
        let e = (s - max_score).exp();
        denom += e;
        exps.push(e);
    }
    if denom <= 0.0 {
        denom = 1.0;
    }
    exps.into_iter().map(|v| v / denom).collect()
}

fn rank_score(row: &Value, rank_mode: &str) -> f64 {
    let key = rank_mode.trim().to_lowercase();
    let best = row.get("best_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let sum_top = row.get("sum_top_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
    match key.as_str() {
        "best_score" => best,
        "hybrid" => best + 0.25 * sum_top,
        _ => sum_top.max(best),
    }
}

fn fts_quote(token: &str) -> String {
    let escaped = token.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

fn expand_search_terms(simplified_text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for tok in simplified_text.split_whitespace() {
        if tok.is_empty() {
            continue;
        }
        out.push(tok.to_string());

        if tok.starts_with("ال") && tok.chars().count() > 2 {
            out.push(tok.chars().skip(2).collect());
        }

        let mut chars = tok.chars();
        let Some(first) = chars.next() else { continue };
        if matches!(first, 'و' | 'ف' | 'ب' | 'ك' | 'ل' | 'س') && tok.chars().count() > 1 {
            let rest: String = chars.collect();
            if !rest.is_empty() {
                out.push(rest.clone());
                if rest.starts_with("ال") && rest.chars().count() > 2 {
                    out.push(rest.chars().skip(2).collect());
                }
            }
        }
    }
    out.join(" ")
}

pub fn build_fts_query_from_transcript(
    transcript_text: &str,
    max_query_tokens: usize,
    query_operator: &str,
) -> Option<(String, Vec<String>)> {
    let transcript_text = transcript_text.trim();
    if transcript_text.is_empty() {
        return None;
    }

    let normalized = normalize_arabic(transcript_text);
    let expanded = expand_search_terms(&normalized);

    let tokens: Vec<String> = expanded
        .split_whitespace()
        .filter_map(|t| {
            let t = t.trim();
            if t.is_empty() {
                return None;
            }
            if t.chars().count() < 2 {
                return None;
            }
            Some(t.to_string())
        })
        .collect();
    if tokens.is_empty() {
        return None;
    }

    let mut token_counts: HashMap<&str, usize> = HashMap::new();
    for t in &tokens {
        *token_counts.entry(t.as_str()).or_insert(0) += 1;
    }

    let mut uniq: Vec<&str> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for t in &tokens {
        let s = t.as_str();
        if seen.insert(s) {
            uniq.push(s);
        }
        if uniq.len() >= 200 {
            break;
        }
    }

    uniq.sort_by(|a, b| {
        let la = a.chars().count();
        let lb = b.chars().count();
        lb.cmp(&la)
            .then_with(|| token_counts.get(b).copied().unwrap_or(0).cmp(&token_counts.get(a).copied().unwrap_or(0)))
            .then_with(|| a.cmp(b))
    });

    let max_query_tokens = max_query_tokens.max(1);
    let chosen: Vec<String> = uniq
        .into_iter()
        .take(max_query_tokens)
        .map(|s| s.to_string())
        .collect();

    let mut op = query_operator.trim().to_lowercase();
    if !matches!(op.as_str(), "or" | "and" | "auto") {
        op = "or".to_string();
    }
    if op == "auto" {
        op = if chosen.len() <= 8 { "and" } else { "or" }.to_string();
    }
    let sep = if op == "and" { " AND " } else { " OR " };
    let fts_query = chosen.iter().map(|t| fts_quote(t)).collect::<Vec<_>>().join(sep);
    Some((fts_query, chosen))
}

fn count_matches(conn: &Connection, fts_query: &str) -> i64 {
    if fts_query.trim().is_empty() {
        return 0;
    }
    conn.query_row(
        "SELECT COUNT(*) FROM ayah_fts WHERE ayah_fts MATCH ?;",
        params![fts_query],
        |row| row.get::<_, i64>(0),
    )
    .unwrap_or(0)
}

fn guess_surah_from_fts_query(
    conn: &Connection,
    fts_query: &str,
    top_k: usize,
    per_surah_top_n: usize,
    rank_mode: &str,
    max_hits: usize,
) -> Vec<Value> {
    if fts_query.trim().is_empty() {
        return Vec::new();
    }
    let per_surah_top_n = per_surah_top_n.max(1);

    let order_by = match rank_mode.trim().to_lowercase().as_str() {
        "best_score" => "best_score DESC, sum_top_score DESC, top_hits DESC, sum_score DESC, ayah_hits DESC",
        "hybrid" => "(best_score + 0.25 * sum_top_score) DESC, best_score DESC, sum_top_score DESC, top_hits DESC, sum_score DESC, ayah_hits DESC",
        _ => "sum_top_score DESC, best_score DESC, top_hits DESC, sum_score DESC, ayah_hits DESC",
    };

    let sql = format!(
        r#"
        WITH hits_raw AS (
          SELECT
            v.surah_id,
            v.surah_name_ar,
            v.surah_transliteration,
            v.ayah_key,
            -bm25(ayah_fts, 1.0, 1.0, 3.0, 4.0) AS score
          FROM ayah_fts
          JOIN v_ayah v ON v.ayah_id = ayah_fts.rowid
          WHERE ayah_fts MATCH ?
          ORDER BY score DESC
          LIMIT ?
        ),
        hits AS (
          SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY surah_id ORDER BY score DESC) AS rn
          FROM hits_raw
        )
        SELECT
          surah_id,
          surah_name_ar,
          surah_transliteration,
          MAX(score) AS best_score,
          SUM(score) AS sum_score,
          COUNT(*) AS ayah_hits,
          SUM(CASE WHEN rn <= ? THEN score ELSE 0 END) AS sum_top_score,
          SUM(CASE WHEN rn <= ? THEN 1 ELSE 0 END) AS top_hits,
          MAX(CASE WHEN rn = 1 THEN ayah_key END) AS best_ayah_key
        FROM hits
        GROUP BY surah_id
        ORDER BY {order_by}
        LIMIT ?;
        "#
    );

    let mut out: Vec<Value> = Vec::new();
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return out,
    };

    let rows = stmt
        .query_map(
            params![
                fts_query,
                max_hits as i64,
                per_surah_top_n as i64,
                per_surah_top_n as i64,
                top_k as i64
            ],
            |row| {
                Ok(json!({
                    "surah_id": row.get::<_, i64>(0)?,
                    "surah_name_ar": row.get::<_, String>(1)?,
                    "surah_transliteration": row.get::<_, String>(2)?,
                    "best_score": row.get::<_, f64>(3)?,
                    "sum_score": row.get::<_, f64>(4)?,
                    "ayah_hits": row.get::<_, i64>(5)?,
                    "sum_top_score": row.get::<_, f64>(6)?,
                    "top_hits": row.get::<_, i64>(7)?,
                    "best_ayah_key": row.get::<_, String>(8)?,
                }))
            },
        )
        .ok();

    if let Some(rows) = rows {
        for r in rows.flatten() {
            out.push(r);
        }
    }
    out
}

pub fn guess_from_transcript_text(
    transcript_text: &str,
    db_path: &Path,
    top_k: usize,
    per_surah_top_n: usize,
    query_operator: &str,
    rank_mode: &str,
    max_query_tokens: usize,
    max_hits: usize,
) -> Value {
    let transcript_text = transcript_text.trim();
    if transcript_text.is_empty() {
        return json!({"guesses": [], "predicted_surah_id": null});
    }
    if !db_path.exists() {
        return json!({"guesses": [], "predicted_surah_id": null, "error": format!("DB not found: {}", db_path.display())});
    }

    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = match Connection::open_with_flags(db_path, flags) {
        Ok(c) => c,
        Err(e) => {
            return json!({"guesses": [], "predicted_surah_id": null, "error": format!("DB open failed: {e}")});
        }
    };

    let Some((fts_query, query_tokens)) = build_fts_query_from_transcript(transcript_text, max_query_tokens, query_operator) else {
        return json!({"guesses": [], "predicted_surah_id": null, "error": "could not build FTS query"});
    };

    let match_count = count_matches(&conn, &fts_query);
    let mut guesses = guess_surah_from_fts_query(&conn, &fts_query, top_k, per_surah_top_n, rank_mode, max_hits);

    let rank_scores: Vec<f64> = guesses.iter().map(|g| rank_score(g, rank_mode)).collect();
    let probs = softmax(&rank_scores);
    for (g, p) in guesses.iter_mut().zip(probs.iter()) {
        if let Some(obj) = g.as_object_mut() {
            obj.insert("probability".to_string(), Value::from(*p));
        }
    }

    let predicted_surah_id = guesses
        .first()
        .and_then(|g| g.get("surah_id"))
        .and_then(|v| v.as_i64());

    json!({
        "query_operator": query_operator,
        "rank_mode": rank_mode,
        "max_query_tokens": max_query_tokens as i64,
        "fts_query": fts_query,
        "query_tokens": query_tokens,
        "match_count": match_count,
        "top_k": top_k as i64,
        "per_surah_top_n": per_surah_top_n as i64,
        "max_hits": max_hits as i64,
        "guesses": guesses,
        "predicted_surah_id": predicted_surah_id,
    })
}


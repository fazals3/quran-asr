use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use rusqlite::Connection;
use serde_json::{json, Value};

use quran_asr_api::alignment_v2::AlignmentAssetsV2;
use quran_asr_api::multispan;

fn parse_ts(ts: &str) -> anyhow::Result<f64> {
    let ts = ts.trim();
    let (hms, ms) = ts
        .split_once('.')
        .ok_or_else(|| anyhow::anyhow!("invalid timestamp: {ts:?}"))?;
    let ms: i64 = ms
        .parse()
        .with_context(|| format!("invalid ms in timestamp: {ts:?}"))?;
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        anyhow::bail!("invalid timestamp: {ts:?}");
    }
    let h: i64 = parts[0].parse()?;
    let m: i64 = parts[1].parse()?;
    let s: i64 = parts[2].parse()?;
    Ok(h as f64 * 3600.0 + m as f64 * 60.0 + s as f64 + (ms as f64) / 1000.0)
}

fn parse_transcript_txt(path: &Path) -> anyhow::Result<(Value, String, Vec<f64>)> {
    let raw = fs::read_to_string(path).with_context(|| format!("read transcript {path:?}"))?;
    let mut segments: Vec<Value> = Vec::new();
    let mut transcript_parts: Vec<String> = Vec::new();
    let mut word_starts: Vec<f64> = Vec::new();

    let mut curr: Option<serde_json::Map<String, Value>> = None;
    for line in raw.lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with('[') && line.contains("]") && line.contains("-->") {
            if let Some(map) = curr.take() {
                segments.push(Value::Object(map));
            }
            let (head, rest) = line
                .split_once(']')
                .ok_or_else(|| anyhow::anyhow!("bad segment header: {line:?}"))?;
            let head = head.trim_start_matches('[').trim();
            let (a, b) = head
                .split_once("-->")
                .ok_or_else(|| anyhow::anyhow!("bad segment header: {line:?}"))?;
            let start_ts = a.trim();
            let end_ts = b.trim();
            let start_s = parse_ts(start_ts)?;
            let end_s = parse_ts(end_ts)?;
            let text = rest.trim().to_string();
            if !text.is_empty() {
                transcript_parts.push(text.clone());
            }
            let mut obj = serde_json::Map::new();
            obj.insert("start_s".to_string(), json!(start_s));
            obj.insert("end_s".to_string(), json!(end_s));
            obj.insert("start_ts".to_string(), json!(start_ts));
            obj.insert("end_ts".to_string(), json!(end_ts));
            obj.insert("text".to_string(), json!(text));
            obj.insert("words".to_string(), json!([]));
            curr = Some(obj);
            continue;
        }

        if !line.contains("-->") {
            continue;
        }
        let Some(curr_obj) = curr.as_mut() else { continue };
        let stripped = line.trim();
        let (left, right) = stripped
            .split_once("-->")
            .ok_or_else(|| anyhow::anyhow!("bad word line: {line:?}"))?;
        let left = left.trim();
        let right = right.trim();
        let mut parts = right.splitn(2, ' ');
        let end_ts = parts.next().unwrap_or("").trim();
        let word = parts.next().unwrap_or("").trim();
        if end_ts.is_empty() || word.is_empty() {
            continue;
        }
        let ws = parse_ts(left)?;
        let we = parse_ts(end_ts)?;
        word_starts.push(ws);
        let w = json!({
            "start_s": ws,
            "end_s": we,
            "start_ts": left,
            "end_ts": end_ts,
            "word": word,
        });
        curr_obj
            .get_mut("words")
            .and_then(|v| v.as_array_mut())
            .unwrap()
            .push(w);
    }
    if let Some(map) = curr.take() {
        segments.push(Value::Object(map));
    }

    word_starts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let transcription = json!({
        "language": "ar",
        "language_probability": 1.0,
        "duration_s": segments.iter().filter_map(|s| s.get("end_s")).filter_map(|v| v.as_f64()).fold(0.0, f64::max),
        "segments": segments,
    });
    let transcript_text = transcript_parts.join(" ").trim().to_string();
    Ok((transcription, transcript_text, word_starts))
}

fn resolve_latest_assets_dir(root: &Path) -> anyhow::Result<PathBuf> {
    let root = root.to_path_buf();
    if root.join("meta.json").is_file() {
        return Ok(root);
    }
    if !root.is_dir() {
        anyhow::bail!("assets dir not found: {root:?}");
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    for e in fs::read_dir(&root).with_context(|| format!("read_dir {root:?}"))? {
        let e = e?;
        let p = e.path();
        if p.is_dir() && p.join("meta.json").is_file() {
            candidates.push(p);
        }
    }
    candidates.sort();
    candidates
        .last()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no assets versions under: {root:?}"))
}

#[derive(Debug, Clone)]
struct SurahReport {
    surah: i64,
    ok: bool,
    updated: bool,
    reason: String,
    confidence: f64,
    ayahs: usize,
    mae_ms: f64,
    median_abs_ms: f64,
    p90_abs_ms: f64,
    max_abs_ms: f64,
}

fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = xs.len() / 2;
    if xs.len() % 2 == 1 {
        xs[mid]
    } else {
        (xs[mid - 1] + xs[mid]) / 2.0
    }
}

fn p90(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((0.9 * ((xs.len() - 1) as f64)).round() as usize).min(xs.len() - 1);
    xs[idx]
}

fn set_segments_first_ms(seg_json: &str, start_ms: i64) -> String {
    let Ok(mut parsed) = serde_json::from_str::<Vec<Vec<i64>>>(seg_json) else {
        return seg_json.to_string();
    };
    let Some(first) = parsed.get_mut(0) else {
        return seg_json.to_string();
    };
    if first.len() == 2 {
        first[1] = start_ms;
        if let Ok(s) = serde_json::to_string(&parsed) {
            return s;
        }
    }
    seg_json.to_string()
}

fn main() -> anyhow::Result<()> {
    let mut db_path = PathBuf::from("quran_audio.db");
    let mut reciter_table = "verses_shaykh_ismael".to_string();
    let mut transcripts_dir = PathBuf::from("transcripts");
    let mut assets_root = PathBuf::from("data/alignment/v2");
    let mut skip_surahs: Vec<i64> = vec![1, 33];
    let mut threshold_ms: i64 = 1500;
    let mut min_confidence: f64 = 0.60;
    let mut max_missing_ayahs: usize = 0;
    let mut report_path = PathBuf::from("outputs/ismael_timestamp_update_report_rust.json");
    let mut force = false;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0usize;
    while i < args.len() {
        let k = args[i].as_str();
        let next = |i: &mut usize, args: &[String]| -> anyhow::Result<String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing value for arg"))
        };
        match k {
            "--db" => db_path = PathBuf::from(next(&mut i, &args)?),
            "--reciter-table" => reciter_table = next(&mut i, &args)?,
            "--transcripts-dir" => transcripts_dir = PathBuf::from(next(&mut i, &args)?),
            "--assets-root" => assets_root = PathBuf::from(next(&mut i, &args)?),
            "--skip-surahs" => {
                let v = next(&mut i, &args)?;
                skip_surahs = v
                    .split(',')
                    .filter_map(|p| p.trim().parse::<i64>().ok())
                    .collect();
            }
            "--threshold-ms" => threshold_ms = next(&mut i, &args)?.parse()?,
            "--min-confidence" => min_confidence = next(&mut i, &args)?.parse()?,
            "--max-missing-ayahs" => max_missing_ayahs = next(&mut i, &args)?.parse()?,
            "--report-path" => report_path = PathBuf::from(next(&mut i, &args)?),
            "--force" => force = true,
            _ => anyhow::bail!("unknown arg: {k}"),
        }
        i += 1;
    }

    let assets_dir = resolve_latest_assets_dir(&assets_root)?;
    let assets = AlignmentAssetsV2::load(&assets_dir, Some(0.75), None)
        .with_context(|| format!("load v2 assets from {assets_dir:?}"))?;

    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut con = Connection::open(&db_path).with_context(|| format!("open db {db_path:?}"))?;
    let skip_set: HashSet<i64> = skip_surahs.iter().copied().collect();

    let mut reports: Vec<SurahReport> = Vec::new();
    let mut updated_surahs: Vec<i64> = Vec::new();

    for surah in (1..=114).rev() {
        eprintln!("surah {surah:03}...");
        if skip_set.contains(&surah) {
            reports.push(SurahReport {
                surah,
                ok: true,
                updated: false,
                reason: "skipped".to_string(),
                confidence: f64::NAN,
                ayahs: 0,
                mae_ms: f64::NAN,
                median_abs_ms: f64::NAN,
                p90_abs_ms: f64::NAN,
                max_abs_ms: f64::NAN,
            });
            continue;
        }

        let tp = transcripts_dir.join(format!("{surah:03}.txt"));
        if !tp.is_file() {
            reports.push(SurahReport {
                surah,
                ok: false,
                updated: false,
                reason: format!("missing transcript: {tp:?}"),
                confidence: f64::NAN,
                ayahs: 0,
                mae_ms: f64::NAN,
                median_abs_ms: f64::NAN,
                p90_abs_ms: f64::NAN,
                max_abs_ms: f64::NAN,
            });
            continue;
        }

        let (transcription, transcript_text, word_starts) = match parse_transcript_txt(&tp) {
            Ok(v) => v,
            Err(e) => {
                reports.push(SurahReport {
                    surah,
                    ok: false,
                    updated: false,
                    reason: format!("transcript parse failed: {e}"),
                    confidence: f64::NAN,
                    ayahs: 0,
                    mae_ms: f64::NAN,
                    median_abs_ms: f64::NAN,
                    p90_abs_ms: f64::NAN,
                    max_abs_ms: f64::NAN,
                });
                continue;
            }
        };

        let ms = multispan::align_multi_span(&transcription, &transcript_text, &assets, "smooth");
        let align = ms.active_alignment;
        if align.get("error").is_some() {
            reports.push(SurahReport {
                surah,
                ok: false,
                updated: false,
                reason: format!("alignment error: {}", align.get("error").unwrap()),
                confidence: f64::NAN,
                ayahs: 0,
                mae_ms: f64::NAN,
                median_abs_ms: f64::NAN,
                p90_abs_ms: f64::NAN,
                max_abs_ms: f64::NAN,
            });
            continue;
        }

        let conf = align.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let start_surah = align
            .get("start")
            .and_then(|v| v.get("surah_id"))
            .and_then(|v| v.as_i64())
            .unwrap_or(-1);
        let end_surah = align
            .get("end")
            .and_then(|v| v.get("surah_id"))
            .and_then(|v| v.as_i64())
            .unwrap_or(-1);
        if start_surah != surah || end_surah != surah {
            reports.push(SurahReport {
                surah,
                ok: false,
                updated: false,
                reason: "aligned range not confined to surah".to_string(),
                confidence: conf,
                ayahs: 0,
                mae_ms: f64::NAN,
                median_abs_ms: f64::NAN,
                p90_abs_ms: f64::NAN,
                max_abs_ms: f64::NAN,
            });
            continue;
        }

        let segs = align
            .get("segments")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if segs.is_empty() {
            reports.push(SurahReport {
                surah,
                ok: false,
                updated: false,
                reason: "missing alignment.segments".to_string(),
                confidence: conf,
                ayahs: 0,
                mae_ms: f64::NAN,
                median_abs_ms: f64::NAN,
                p90_abs_ms: f64::NAN,
                max_abs_ms: f64::NAN,
            });
            continue;
        }

        let mut old: BTreeMap<i64, i64> = BTreeMap::new();
        {
            let mut stmt = con.prepare(&format!(
                "select ayah_number, start_time from {} where surah_number=? order by ayah_number asc",
                reciter_table
            ))?;
            let rows = stmt.query_map([surah], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
            for row in rows {
                let (ayah, start_ms) = row?;
                old.insert(ayah, start_ms);
            }
        }
        if old.is_empty() {
            reports.push(SurahReport {
                surah,
                ok: false,
                updated: false,
                reason: "no db rows for surah".to_string(),
                confidence: conf,
                ayahs: 0,
                mae_ms: f64::NAN,
                median_abs_ms: f64::NAN,
                p90_abs_ms: f64::NAN,
                max_abs_ms: f64::NAN,
            });
            continue;
        }

        // Build new start_time + segments by partitioning word start times by aligned ayah end_s.
        let mut new_rows: Vec<(i64, i64, i64, String)> = Vec::new();
        let mut word_i = 0usize;
        let mut cursor_s = 0.0_f64;
        let mut prev_start_ms: Option<i64> = None;

        for (idx, a) in segs.iter().enumerate() {
            let ayah_num = a.get("ayah_num").and_then(|v| v.as_i64()).unwrap_or(0);
            if ayah_num <= 0 {
                continue;
            }
            let end_s = a.get("end_s").and_then(|v| v.as_f64()).or_else(|| {
                segs.get(idx + 1)
                    .and_then(|n| n.get("start_s"))
                    .and_then(|v| v.as_f64())
            });
            let end_s = end_s.unwrap_or(f64::INFINITY);

            let mut w_ms: Vec<i64> = Vec::new();
            while word_i < word_starts.len() && word_starts[word_i] < end_s {
                let ws = word_starts[word_i];
                if ws >= cursor_s {
                    w_ms.push((ws * 1000.0).round() as i64);
                }
                word_i += 1;
            }
            if end_s.is_finite() {
                cursor_s = end_s;
            }

            let (start_ms, segments_json) = if !w_ms.is_empty() {
                w_ms.sort();
                let mut segs2: Vec<Vec<i64>> = Vec::with_capacity(w_ms.len());
                for (j, ms) in w_ms.iter().enumerate() {
                    segs2.push(vec![j as i64, *ms]);
                }
                (*w_ms.first().unwrap(), serde_json::to_string(&segs2)?)
            } else {
                let s = a.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let start_ms = (s * 1000.0).round() as i64;
                (start_ms, serde_json::to_string(&vec![vec![0i64, start_ms]])?)
            };

            let mut start_ms = start_ms;
            if let Some(prev) = prev_start_ms {
                if start_ms < prev {
                    start_ms = prev;
                }
            }
            prev_start_ms = Some(start_ms);

            let seg_json = set_segments_first_ms(&segments_json, start_ms);

            new_rows.push((surah, ayah_num, start_ms, seg_json));
        }

        if new_rows.len() > old.len() {
            reports.push(SurahReport {
                surah,
                ok: false,
                updated: false,
                reason: format!("ayah count mismatch (db={} align={})", old.len(), new_rows.len()),
                confidence: conf,
                ayahs: old.len(),
                mae_ms: f64::NAN,
                median_abs_ms: f64::NAN,
                p90_abs_ms: f64::NAN,
                max_abs_ms: f64::NAN,
            });
            eprintln!(
                "surah {surah:03} skipped (ayah_count_mismatch) (confidence={conf:.3})"
            );
            continue;
        }
        let missing = old.len().saturating_sub(new_rows.len());
        if missing > max_missing_ayahs {
            reports.push(SurahReport {
                surah,
                ok: false,
                updated: false,
                reason: format!("ayah count mismatch (db={} align={})", old.len(), new_rows.len()),
                confidence: conf,
                ayahs: old.len(),
                mae_ms: f64::NAN,
                median_abs_ms: f64::NAN,
                p90_abs_ms: f64::NAN,
                max_abs_ms: f64::NAN,
            });
            eprintln!(
                "surah {surah:03} skipped (ayah_count_mismatch) (missing={missing} max_missing={max_missing_ayahs}) (confidence={conf:.3})"
            );
            continue;
        }

        // If we're not updating every ayah, ensure the proposed updates won't break monotonicity with
        // the existing (unchanged) rows. Clamp updated ayahs upward as needed; if we'd need to change
        // an unchanged ayah to keep monotonicity, skip the update for this surah.
        if new_rows.len() != old.len() {
            let mut proposed: BTreeMap<i64, (i64, String)> = BTreeMap::new();
            for (_surah_num, ayah_num, start_ms, seg_json) in new_rows.into_iter() {
                proposed.insert(ayah_num, (start_ms, seg_json));
            }

            let mut monotonic_ok = true;
            let mut last_time: Option<i64> = None;
            let mut adjusted: Vec<(i64, i64, i64, String)> = Vec::new();
            for (ayah_num, old_ms) in &old {
                let mut t = *old_ms;
                if let Some((p_ms, _)) = proposed.get(ayah_num) {
                    t = *p_ms;
                }

                if let Some(prev) = last_time {
                    if t < prev {
                        if proposed.contains_key(ayah_num) {
                            t = prev;
                        } else {
                            monotonic_ok = false;
                        }
                    }
                }
                last_time = Some(t);

                if let Some((_, seg_json)) = proposed.get(ayah_num) {
                    let seg_json = set_segments_first_ms(seg_json, t);
                    adjusted.push((surah, *ayah_num, t, seg_json));
                }
            }

            if !monotonic_ok {
                reports.push(SurahReport {
                    surah,
                    ok: false,
                    updated: false,
                    reason: "partial update would break monotonicity".to_string(),
                    confidence: conf,
                    ayahs: old.len(),
                    mae_ms: f64::NAN,
                    median_abs_ms: f64::NAN,
                    p90_abs_ms: f64::NAN,
                    max_abs_ms: f64::NAN,
                });
                eprintln!("surah {surah:03} skipped (partial_monotonicity_conflict) (confidence={conf:.3})");
                continue;
            }

            new_rows = adjusted;
        }

        // Compare to DB.
        let mut diffs: Vec<f64> = Vec::new();
        for (surah_num, ayah_num, start_ms, _) in &new_rows {
            let _ = surah_num;
            if let Some(old_ms) = old.get(ayah_num) {
                diffs.push((*start_ms - *old_ms) as f64);
            }
        }
        let abs_d: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
        let mae = if abs_d.is_empty() {
            f64::NAN
        } else {
            abs_d.iter().sum::<f64>() / (abs_d.len() as f64)
        };
        let med = median(&mut abs_d.clone());
        let p90v = p90(&mut abs_d.clone());
        let max_abs = abs_d.iter().cloned().fold(0.0_f64, f64::max);

        let should_update = force || (conf >= min_confidence && max_abs >= (threshold_ms as f64));
        if should_update {
            let tx = con.transaction()?;
            let q = format!(
                "update {} set start_time=?, segments=? where surah_number=? and ayah_number=?",
                reciter_table
            );
            for (surah_num, ayah_num, start_ms, seg_json) in &new_rows {
                tx.execute(&q, rusqlite::params![start_ms, seg_json, surah_num, ayah_num])?;
            }
            tx.commit()?;
            updated_surahs.push(surah);
            reports.push(SurahReport {
                surah,
                ok: true,
                updated: true,
                reason: "updated".to_string(),
                confidence: conf,
                ayahs: new_rows.len(),
                mae_ms: mae,
                median_abs_ms: med,
                p90_abs_ms: p90v,
                max_abs_ms: max_abs,
            });
            eprintln!(
                "surah {surah:03} updated (confidence={conf:.3} max_abs_ms={max_abs:.0})"
            );
        } else {
            let reason = if conf < min_confidence {
                "low_confidence"
            } else {
                "below_threshold"
            };
            reports.push(SurahReport {
                surah,
                ok: true,
                updated: false,
                reason: reason.to_string(),
                confidence: conf,
                ayahs: new_rows.len(),
                mae_ms: mae,
                median_abs_ms: med,
                p90_abs_ms: p90v,
                max_abs_ms: max_abs,
            });
            eprintln!(
                "surah {surah:03} skipped ({reason}) (confidence={conf:.3} max_abs_ms={max_abs:.0})"
            );
        }
    }

    let payload = json!({
        "db": db_path,
        "reciter_table": reciter_table,
        "transcripts_dir": transcripts_dir,
        "assets_dir": assets_dir,
        "skip_surahs": skip_surahs,
        "threshold_ms": threshold_ms,
        "min_confidence": min_confidence,
        "max_missing_ayahs": max_missing_ayahs,
        "force": force,
        "updated_surahs": updated_surahs,
        "reports": reports.iter().map(|r| json!({
            "surah": r.surah,
            "ok": r.ok,
            "updated": r.updated,
            "reason": r.reason,
            "confidence": r.confidence,
            "ayahs": r.ayahs,
            "mae_ms": r.mae_ms,
            "median_abs_ms": r.median_abs_ms,
            "p90_abs_ms": r.p90_abs_ms,
            "max_abs_ms": r.max_abs_ms,
        })).collect::<Vec<Value>>(),
    });
    fs::write(&report_path, serde_json::to_string_pretty(&payload)?)?;

    println!("updated {} surahs", payload["updated_surahs"].as_array().map(|v| v.len()).unwrap_or(0));
    println!("report: {}", report_path.display());
    Ok(())
}

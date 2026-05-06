use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use half::f16;
use lru::LruCache;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::{json, Value};

use crate::npy::load_npy_memmap;
use crate::text::{char_ngram_jaccard, format_timestamp, hamming64, normalize_arabic, normalize_token, simhash64_from_char_ngrams};

#[derive(Debug, Clone)]
pub struct ConfusionModel {
    pub vocab_size: usize,
    pub global_ref_del_cost: f64,
    pub global_obs_ins_cost: f64,
    pub fallback_sub_cost: f64,
    pub sub_cost_by_ref: Vec<Option<Vec<(u32, f64)>>>,
}

impl ConfusionModel {
    pub fn load(
        path: &Path,
        vocab_size: usize,
        ref_del_cost_cap: Option<f64>,
        obs_ins_cost_cap: Option<f64>,
    ) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("read confusions: {}", path.display()))?;
        let obj: Value = serde_json::from_slice(&bytes)?;

        let defaults = obj.get("defaults").and_then(|v| v.as_object());
        let mut global_ref_del_cost = defaults
            .and_then(|d| d.get("global_ref_del_cost"))
            .and_then(|v| v.as_f64())
            .unwrap_or(2.0);
        let mut global_obs_ins_cost = defaults
            .and_then(|d| d.get("global_obs_ins_cost"))
            .and_then(|v| v.as_f64())
            .unwrap_or(2.0);
        let mut fallback_sub_cost = defaults
            .and_then(|d| d.get("fallback_sub_cost"))
            .and_then(|v| v.as_f64())
            .unwrap_or(2.5);
        fallback_sub_cost = fallback_sub_cost.clamp(0.25, 5.0);

        if let Some(cap) = ref_del_cost_cap.and_then(|c| if c > 0.0 { Some(c.max(0.05)) } else { None }) {
            global_ref_del_cost = global_ref_del_cost.min(cap);
        }
        if let Some(cap) = obs_ins_cost_cap.and_then(|c| if c > 0.0 { Some(c.max(0.05)) } else { None }) {
            global_obs_ins_cost = global_obs_ins_cost.min(cap);
        }

        let mut sub_cost_by_ref: Vec<Option<Vec<(u32, f64)>>> = vec![None; vocab_size + 1];
        if let Some(sub_cost) = obj.get("sub_cost").and_then(|v| v.as_object()) {
            for (rid_s, items) in sub_cost {
                let rid: usize = match rid_s.parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if rid >= sub_cost_by_ref.len() {
                    continue;
                }
                let Some(arr) = items.as_array() else { continue };
                let mut row: Vec<(u32, f64)> = Vec::new();
                for pair in arr {
                    let Some(p) = pair.as_array() else { continue };
                    if p.len() != 2 {
                        continue;
                    }
                    let oid = p[0].as_u64().unwrap_or(0) as u32;
                    let cost = p[1].as_f64().unwrap_or(fallback_sub_cost);
                    row.push((oid, cost));
                }
                if !row.is_empty() {
                    row.sort_by_key(|(oid, _)| *oid);
                    sub_cost_by_ref[rid] = Some(row);
                }
            }
        }

        Ok(Self {
            vocab_size,
            global_ref_del_cost,
            global_obs_ins_cost,
            fallback_sub_cost,
            sub_cost_by_ref,
        })
    }
}

#[derive(Debug)]
pub struct AlignmentAssetsV2 {
    pub assets_dir: PathBuf,
    pub version: String,
    pub id_to_token: Vec<String>,
    pub token_to_id: HashMap<String, u32>,

    pub ref_token_ids: crate::npy::NpyMemmap<u32>,
    pub ref_surah_id: crate::npy::NpyMemmap<u16>,
    pub ref_ayah_num: crate::npy::NpyMemmap<u16>,
    pub ref_ayah_global: crate::npy::NpyMemmap<u16>,
    pub ref_word_pos: crate::npy::NpyMemmap<u16>,
    pub ayah_token_start: crate::npy::NpyMemmap<u32>,
    pub ayah_token_end: crate::npy::NpyMemmap<u32>,

    pub token_idf: crate::npy::NpyMemmap<f32>,
    pub stop_mask: Vec<bool>,
    pub inv_offsets: crate::npy::NpyMemmap<u32>,
    pub inv_positions: crate::npy::NpyMemmap<u32>,

    pub passages_start_ayah_global: crate::npy::NpyMemmap<u16>,
    pub passages_end_ayah_global: crate::npy::NpyMemmap<u16>,
    pub passages_simhash64: crate::npy::NpyMemmap<u64>,
    pub passages_emb_f16: crate::npy::NpyMemmap<f16>,
    pub embedding_model_name: String,
    pub embedding_dim: usize,

    pub confusions: Option<ConfusionModel>,

    // Convenience mapping for span/jump logic.
    pub ayah_key_to_global: HashMap<(u16, u16), u16>,
    pub ayah_global_to_key: Vec<(u16, u16)>, // index by ayah_global
}

impl AlignmentAssetsV2 {
    pub fn total_tokens(&self) -> usize {
        self.ref_token_ids.len()
    }

    pub fn total_ayahs(&self) -> usize {
        self.ayah_token_start.len().saturating_sub(2)
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len().saturating_sub(1)
    }

    pub fn load(
        assets_dir: &Path,
        ref_del_cost_cap: Option<f64>,
        obs_ins_cost_cap: Option<f64>,
    ) -> Result<Self> {
        let assets_dir = assets_dir
            .canonicalize()
            .with_context(|| format!("canonicalize assets dir: {}", assets_dir.display()))?;

        let meta_path = assets_dir.join("meta.json");
        let vocab_path = assets_dir.join("vocab.json");
        if !meta_path.exists() {
            return Err(anyhow!("meta.json not found in assets dir: {}", assets_dir.display()));
        }
        if !vocab_path.exists() {
            return Err(anyhow!("vocab.json not found in assets dir: {}", assets_dir.display()));
        }

        let meta: Value = serde_json::from_slice(&fs::read(&meta_path)?)?;
        let version = meta
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| assets_dir.file_name().and_then(|s| s.to_str()).unwrap_or(""))
            .to_string();

        let vocab_obj: Value = serde_json::from_slice(&fs::read(&vocab_path)?)?;
        let id_to_token_val = vocab_obj
            .get("id_to_token")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("vocab.json missing id_to_token[]"))?;
        let mut id_to_token: Vec<String> = Vec::with_capacity(id_to_token_val.len());
        for v in id_to_token_val {
            id_to_token.push(v.as_str().unwrap_or("").to_string());
        }
        if id_to_token.is_empty() {
            return Err(anyhow!("vocab.json id_to_token empty"));
        }

        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        for (i, t) in id_to_token.iter().enumerate() {
            if !t.is_empty() {
                token_to_id.insert(t.clone(), i as u32);
            }
        }

        let ref_token_ids = load_npy_memmap::<u32>(&assets_dir.join("ref_token_ids.npy"), "<u4", 1)?;
        let ref_ayah_global = load_npy_memmap::<u16>(&assets_dir.join("ref_ayah_global.npy"), "<u2", 1)?;
        let ref_surah_id = load_npy_memmap::<u16>(&assets_dir.join("ref_surah_id.npy"), "<u2", 1)?;
        let ref_ayah_num = load_npy_memmap::<u16>(&assets_dir.join("ref_ayah_num.npy"), "<u2", 1)?;
        let ref_word_pos = load_npy_memmap::<u16>(&assets_dir.join("ref_word_pos.npy"), "<u2", 1)?;
        let ayah_token_start = load_npy_memmap::<u32>(&assets_dir.join("ayah_token_start.npy"), "<u4", 1)?;
        let ayah_token_end = load_npy_memmap::<u32>(&assets_dir.join("ayah_token_end.npy"), "<u4", 1)?;

        let token_idf = load_npy_memmap::<f32>(&assets_dir.join("token_idf.npy"), "<f4", 1)?;
        let stop_token_ids = load_npy_memmap::<u32>(&assets_dir.join("stop_token_ids.npy"), "<u4", 1)?;
        let inv_offsets = load_npy_memmap::<u32>(&assets_dir.join("inv_offsets.npy"), "<u4", 1)?;
        let inv_positions = load_npy_memmap::<u32>(&assets_dir.join("inv_positions.npy"), "<u4", 1)?;

        let passages_start_ayah_global =
            load_npy_memmap::<u16>(&assets_dir.join("passages_start_ayah_global.npy"), "<u2", 1)?;
        let passages_end_ayah_global =
            load_npy_memmap::<u16>(&assets_dir.join("passages_end_ayah_global.npy"), "<u2", 1)?;
        let passages_simhash64 =
            load_npy_memmap::<u64>(&assets_dir.join("passages_simhash64.npy"), "<u8", 1)?;
        let passages_emb_f16 = load_npy_memmap::<f16>(&assets_dir.join("passages_emb_f16.npy"), "<f2", 2)?;

        let embed_meta = meta.get("passages").and_then(|v| v.as_object());
        let embedding_model_name = embed_meta
            .and_then(|m| m.get("embedding_model"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let embedding_dim = embed_meta
            .and_then(|m| m.get("embedding_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let vocab_size = id_to_token.len().saturating_sub(1);
        let mut stop_mask = vec![false; vocab_size + 1];
        for &tid in stop_token_ids.as_slice() {
            let tid = tid as usize;
            if tid < stop_mask.len() {
                stop_mask[tid] = true;
            }
        }

        let conf_path = assets_dir.join("confusions.json");
        let confusions = if conf_path.exists() {
            ConfusionModel::load(&conf_path, vocab_size, ref_del_cost_cap, obs_ins_cost_cap).ok()
        } else {
            None
        };

        let total_ayahs = ayah_token_start.len().saturating_sub(2);
        let mut ayah_key_to_global: HashMap<(u16, u16), u16> = HashMap::with_capacity(total_ayahs.saturating_mul(2));
        let mut ayah_global_to_key: Vec<(u16, u16)> = vec![(0u16, 0u16); total_ayahs + 1];
        {
            let token_starts = ayah_token_start.as_slice();
            let surahs = ref_surah_id.as_slice();
            let ayahs = ref_ayah_num.as_slice();
            for g in 1..=total_ayahs {
                let idx = token_starts.get(g).copied().unwrap_or(0) as usize;
                if idx >= surahs.len() || idx >= ayahs.len() {
                    continue;
                }
                let sid = surahs[idx];
                let anum = ayahs[idx];
                ayah_key_to_global.insert((sid, anum), g as u16);
                ayah_global_to_key[g] = (sid, anum);
            }
        }

        Ok(Self {
            assets_dir,
            version,
            id_to_token,
            token_to_id,
            ref_token_ids,
            ref_surah_id,
            ref_ayah_num,
            ref_ayah_global,
            ref_word_pos,
            ayah_token_start,
            ayah_token_end,
            token_idf,
            stop_mask,
            inv_offsets,
            inv_positions,
            passages_start_ayah_global,
            passages_end_ayah_global,
            passages_simhash64,
            passages_emb_f16,
            embedding_model_name,
            embedding_dim,
            confusions,
            ayah_key_to_global,
            ayah_global_to_key,
        })
    }
}

#[derive(Debug, Clone)]
struct TimedToken {
    raw: String,
    simplified: String,
    start_s: Option<f64>,
    end_s: Option<f64>,
    estimated_time: bool,
}

fn safe_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn safe_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_i64().map(|i| i != 0),
        Value::String(s) => {
            let t = s.trim().to_lowercase();
            if t.is_empty() {
                None
            } else if matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on") {
                Some(true)
            } else if matches!(t.as_str(), "0" | "false" | "no" | "n" | "off") {
                Some(false)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn env_bool(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(
            v.trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ),
        Err(_) => false,
    }
}

fn extract_timed_tokens_from_transcription(transcription: &Value) -> Vec<TimedToken> {
    let mut out: Vec<TimedToken> = Vec::new();
    let Some(segs) = transcription.get("segments").and_then(|v| v.as_array()) else {
        return out;
    };

    for seg in segs {
        let Some(seg_obj) = seg.as_object() else { continue };

        let seg_text = seg_obj
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let mut simplified_parts: Vec<String> = Vec::new();
        if !seg_text.is_empty() {
            for p in seg_text.split_whitespace() {
                let s = normalize_token(p);
                if !s.is_empty() {
                    simplified_parts.push(s);
                }
            }
        }
        let seg_start = seg_obj.get("start_s").and_then(safe_f64);
        let seg_end = seg_obj.get("end_s").and_then(safe_f64);

        if let Some(words) = seg_obj.get("words").and_then(|v| v.as_array()) {
            if !words.is_empty() {
                let mut word_tokens: Vec<TimedToken> = Vec::with_capacity(words.len());
                for w in words {
                    let Some(wobj) = w.as_object() else { continue };
                    let raw = wobj
                        .get("word")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if raw.is_empty() {
                        continue;
                    }
                    let simplified = normalize_token(&raw);
                    if simplified.is_empty() {
                        continue;
                    }
                    word_tokens.push(TimedToken {
                        raw,
                        simplified,
                        start_s: wobj.get("start_s").and_then(safe_f64),
                        end_s: wobj.get("end_s").and_then(safe_f64),
                        estimated_time: wobj
                            .get("estimated")
                            .and_then(safe_bool)
                            .unwrap_or(false),
                    });
                }

                if !word_tokens.is_empty() {
                    if simplified_parts.is_empty() {
                        out.extend(word_tokens);
                        continue;
                    }
                    let word_n = word_tokens.len();
                    let text_n = simplified_parts.len();
                    let looks_broken = (text_n >= 4 && (word_n as f64) / (text_n as f64) < 0.6)
                        || (text_n >= 3 && word_n == 1);
                    if !looks_broken || seg_start.is_none() || seg_end.is_none() {
                        out.extend(word_tokens);
                        continue;
                    }
                }
            }
        }

        let (Some(seg_start), Some(seg_end)) = (seg_start, seg_end) else { continue };
        if simplified_parts.is_empty() {
            continue;
        }
        let dur = (seg_end - seg_start).max(0.0);
        let step = dur / simplified_parts.len() as f64;
        for (idx, tok) in simplified_parts.into_iter().enumerate() {
            let w_start = seg_start + idx as f64 * step;
            let w_end = if step > 0.0 {
                seg_start + (idx as f64 + 1.0) * step
            } else {
                seg_end
            };
            out.push(TimedToken {
                raw: tok.clone(),
                simplified: tok,
                start_s: Some(w_start),
                end_s: Some(w_end),
                estimated_time: true,
            });
        }
    }

    out
}

fn upgrade_word_timestamps(tokens: &[TimedToken], mode: &str) -> Result<Vec<TimedToken>> {
    let key = mode.trim().to_lowercase();
    if key.is_empty() || matches!(key.as_str(), "0" | "false" | "off" | "none") {
        return Ok(tokens.to_vec());
    }
    if key == "smooth" || key == "smoothing" {
        let mut out: Vec<TimedToken> = Vec::with_capacity(tokens.len());
        let mut prev_end: Option<f64> = None;
        for t in tokens {
            let (Some(mut s), Some(mut e)) = (t.start_s, t.end_s) else {
                out.push(t.clone());
                continue;
            };
            if e < s {
                e = s;
            }
            if let Some(pe) = prev_end {
                if s < pe {
                    s = pe;
                    if e < s {
                        e = s;
                    }
                }
            }
            prev_end = Some(e);
            out.push(TimedToken {
                raw: t.raw.clone(),
                simplified: t.simplified.clone(),
                start_s: Some(s),
                end_s: Some(e),
                estimated_time: t.estimated_time,
            });
        }
        return Ok(out);
    }
    if key == "whisperx" || key == "forced" {
        return Err(anyhow!("word_time_upgrade=whisperx is not implemented"));
    }
    Err(anyhow!("unknown word_time_upgrade mode: {mode}"))
}

fn build_query_text(tokens: &[TimedToken], max_words: usize) -> String {
    let toks: Vec<&str> = tokens.iter().map(|t| t.simplified.as_str()).filter(|t| !t.is_empty()).collect();
    if toks.is_empty() {
        return String::new();
    }
    if toks.len() <= max_words {
        return toks.join(" ");
    }
    let head_len = max_words / 2;
    let head = &toks[..head_len];
    let tail = &toks[toks.len().saturating_sub(max_words - head_len)..];
    let mut joined: Vec<&str> = Vec::with_capacity(max_words);
    joined.extend_from_slice(head);
    joined.extend_from_slice(tail);
    joined.join(" ")
}

#[derive(Debug, Clone)]
struct CandidateWindow {
    ref_start: usize,
    ref_end: usize,
    expected_offset: i32,
    source: &'static str,
    hint: String,
    quick_score: f64,
}

fn add_candidate(
    selected: &mut Vec<CandidateWindow>,
    seen_key: &mut HashSet<(usize, usize, i32)>,
    c: &CandidateWindow,
) {
    let key = (c.ref_start, c.ref_end, c.expected_offset);
    if seen_key.insert(key) {
        selected.push(c.clone());
    }
}

#[derive(Debug, Clone)]
struct AlignmentResultV2 {
    cost: f64,
    ref_start: usize,
    ref_end: usize,
    transcript_to_ref: Vec<Option<usize>>,
    transcript_is_match: Vec<bool>,
    matches: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    expected_offset: i32,
    candidate_source: String,
    quick_score: f64,
}

fn weighted_banded_dp(
    obs_ids: &[u32],
    ref_ids: &[u32],
    ref_start_global: usize,
    expected_offset_global: i32,
    band: usize,
    conf: Option<&ConfusionModel>,
    id_to_token: &[String],
) -> AlignmentResultV2 {
    let m = obs_ids.len();
    let n = ref_ids.len();
    if m == 0 || n == 0 {
        return AlignmentResultV2 {
            cost: m as f64,
            ref_start: 0,
            ref_end: 0,
            transcript_to_ref: vec![None; m],
            transcript_is_match: vec![false; m],
            matches: 0,
            substitutions: 0,
            deletions: m,
            insertions: 0,
            expected_offset: expected_offset_global,
            candidate_source: String::new(),
            quick_score: 0.0,
        };
    }

    let global_ref_del = conf.map(|c| c.global_ref_del_cost).unwrap_or(1.0);
    let global_obs_ins = conf.map(|c| c.global_obs_ins_cost).unwrap_or(1.0);
    let fallback_sub = conf.map(|c| c.fallback_sub_cost).unwrap_or(1.0);

    let offset_local = expected_offset_global - ref_start_global as i32;
    let inf = 1e18_f64;

    let mut prev_cost = vec![0.0_f64; n + 1]; // substring start anywhere
    let mut prev_match = vec![0_i32; n + 1];
    let mut curr_cost = vec![inf; n + 1];
    let mut curr_match = vec![0_i32; n + 1];

    let width = 2 * band + 1;
    let mut row_j0 = vec![0_i32; m + 1];
    let mut row_j1 = vec![0_i32; m + 1];
    let mut ptr = vec![0_u8; (m + 1) * width];

    let mut backoff_cache: LruCache<u64, f64> = LruCache::new(std::num::NonZeroUsize::new(200_000).unwrap());

    let sub_cost_by_ref = conf.map(|c| &c.sub_cost_by_ref);

    for i in 1..=m {
        let oi = obs_ids[i - 1] as usize;

        curr_cost[0] = prev_cost[0] + global_obs_ins;
        curr_match[0] = prev_match[0];

        let j_expected = i as i32 + offset_local;
        let mut j0 = (j_expected - band as i32).max(1);
        let mut j1 = (j_expected + band as i32).min(n as i32);
        if j0 > n as i32 {
            j0 = n as i32;
        }
        if j0 > j1 {
            row_j0[i] = 1;
            row_j1[i] = 0;
            curr_cost.fill(inf);
            curr_match.fill(0);
            std::mem::swap(&mut prev_cost, &mut curr_cost);
            std::mem::swap(&mut prev_match, &mut curr_match);
            continue;
        }
        row_j0[i] = j0;
        row_j1[i] = j1;
        let row_base = i * width;

        // Set outside band to INF (reused arrays).
        for j in 1..j0 as usize {
            curr_cost[j] = inf;
            curr_match[j] = 0;
        }
        for j in (j1 as usize + 1)..=n {
            curr_cost[j] = inf;
            curr_match[j] = 0;
        }

        for j in j0 as usize..=j1 as usize {
            let rj = ref_ids[j - 1] as usize;

            let is_match = oi == rj && oi != 0;
            let sub_cost = if is_match {
                0.0
            } else {
                let mut cost = None;
                if let Some(by_ref) = sub_cost_by_ref {
                    if rj < by_ref.len() {
                        if let Some(row) = &by_ref[rj] {
                            if let Ok(idx) = row.binary_search_by_key(&(oi as u32), |(oid, _)| *oid) {
                                cost = Some(row[idx].1);
                            }
                        }
                    }
                }
                cost.unwrap_or_else(|| {
                    if oi == 0 || rj == 0 {
                        return fallback_sub;
                    }
                    if oi == rj {
                        return 0.0;
                    }
                    let key = ((rj as u64) << 32) | (oi as u64);
                    if let Some(v) = backoff_cache.get(&key) {
                        return *v;
                    }
                    let rt = id_to_token.get(rj).map(|s| s.as_str()).unwrap_or("");
                    let ot = id_to_token.get(oi).map(|s| s.as_str()).unwrap_or("");
                    if rt.is_empty() || ot.is_empty() {
                        backoff_cache.put(key, fallback_sub);
                        return fallback_sub;
                    }
                    if rt.chars().next() != ot.chars().next() {
                        backoff_cache.put(key, fallback_sub);
                        return fallback_sub;
                    }
                    let rl = rt.chars().count() as i32;
                    let ol = ot.chars().count() as i32;
                    if (rl - ol).abs() > 3 {
                        backoff_cache.put(key, fallback_sub);
                        return fallback_sub;
                    }
                    let sim = char_ngram_jaccard(rt, ot, 3);
                    let out = (fallback_sub * (1.0 - 0.6 * sim)).max(0.0);
                    backoff_cache.put(key, out);
                    out
                })
            };

            let sub_c = prev_cost[j - 1] + sub_cost;
            let sub_m = prev_match[j - 1] + if is_match { 1 } else { 0 };

            let del_c = prev_cost[j] + global_obs_ins;
            let del_m = prev_match[j];

            let ins_c = curr_cost[j - 1] + global_ref_del;
            let ins_m = curr_match[j - 1];

            let mut best_c = sub_c;
            let mut best_m = sub_m;
            let mut code: u8 = 0;

            let mut take = |c: f64, m_: i32, code_: u8| {
                if c < best_c {
                    best_c = c;
                    best_m = m_;
                    code = code_;
                    return;
                }
                if c == best_c && m_ > best_m {
                    best_c = c;
                    best_m = m_;
                    code = code_;
                }
            };

            take(del_c, del_m, 1);
            take(ins_c, ins_m, 2);

            curr_cost[j] = best_c;
            curr_match[j] = best_m;

            let idx = j as i32 - j0;
            if idx >= 0 && (idx as usize) < width {
                ptr[row_base + idx as usize] = code;
            }
        }

        std::mem::swap(&mut prev_cost, &mut curr_cost);
        std::mem::swap(&mut prev_match, &mut curr_match);
    }

    let mut best_cost = inf;
    for &c in &prev_cost {
        if c < best_cost {
            best_cost = c;
        }
    }
    let mut best_match = 0_i32;
    for (j, &c) in prev_cost.iter().enumerate() {
        if c == best_cost {
            best_match = best_match.max(prev_match[j]);
        }
    }
    let mut j_end: usize = 0;
    for (j, &c) in prev_cost.iter().enumerate() {
        if c == best_cost && prev_match[j] == best_match {
            j_end = j;
            break;
        }
    }

    let mut transcript_to_ref: Vec<Option<usize>> = vec![None; m];
    let mut transcript_is_match: Vec<bool> = vec![false; m];
    let mut matches = 0usize;
    let mut substitutions = 0usize;
    let mut deletions = 0usize;
    let mut insertions = 0usize;

    let mut i = m;
    let mut j = j_end;
    while i > 0 {
        if j == 0 {
            deletions += 1;
            i -= 1;
            continue;
        }
        let j0 = row_j0[i];
        let j1 = row_j1[i];
        if j0 > j1 || (j as i32) < j0 || (j as i32) > j1 {
            deletions += 1;
            i -= 1;
            continue;
        }
        let row_base = i * width;
        let idx = j as i32 - j0;
        if idx < 0 || (idx as usize) >= width {
            deletions += 1;
            i -= 1;
            continue;
        }
        let code = ptr[row_base + idx as usize];
        if code == 0 {
            let i0 = i - 1;
            let j0_local = j - 1;
            let rid = ref_ids[j0_local] as usize;
            let oid = obs_ids[i0] as usize;
            transcript_to_ref[i0] = Some(ref_start_global + j0_local);
            let is_match = rid == oid && oid != 0;
            transcript_is_match[i0] = is_match;
            substitutions += 1;
            if is_match {
                matches += 1;
            }
            i -= 1;
            j -= 1;
        } else if code == 1 {
            deletions += 1;
            i -= 1;
        } else {
            insertions += 1;
            j -= 1;
        }
    }

    let ref_start_local = j;
    let ref_end_local = j_end.max(ref_start_local);

    AlignmentResultV2 {
        cost: best_cost,
        ref_start: ref_start_global + ref_start_local,
        ref_end: ref_start_global + ref_end_local,
        transcript_to_ref,
        transcript_is_match,
        matches,
        substitutions,
        deletions,
        insertions,
        expected_offset: expected_offset_global,
        candidate_source: String::new(),
        quick_score: 0.0,
    }
}

fn generate_candidates(
    assets: &AlignmentAssetsV2,
    obs_ids: &[u32],
    query_text: &str,
    top_simhash: usize,
    top_anchor_offsets: usize,
    anchor_token_budget: usize,
    window_margin_ayahs: usize,
    window_margin_tokens: usize,
) -> Vec<CandidateWindow> {
    let m = obs_ids.len();
    let total_tokens = assets.total_tokens();

    let mut candidates: Vec<CandidateWindow> = Vec::new();

    // 2D: SimHash fallback (cheap + deterministic).
    if !query_text.is_empty() && assets.passages_simhash64.len() > 0 {
        let qh = simhash64_from_char_ngrams(query_text);
        let simhash = assets.passages_simhash64.as_slice();
        let mut dists: Vec<(usize, u32)> = simhash
            .iter()
            .enumerate()
            .map(|(i, &h)| (i, hamming64(h, qh)))
            .collect();
        dists.sort_by_key(|x| x.1);
        for (rank, (pi, dist)) in dists.into_iter().take(top_simhash).enumerate() {
                let mut start_ayah = assets.passages_start_ayah_global.as_slice()[pi] as i32;
                let mut end_ayah = assets.passages_end_ayah_global.as_slice()[pi] as i32;
                start_ayah = (start_ayah - window_margin_ayahs as i32).max(1);
                end_ayah = (end_ayah + window_margin_ayahs as i32).min(assets.total_ayahs() as i32);

                let passage_start = assets.ayah_token_start.as_slice()[start_ayah as usize] as i32;
                let passage_end = assets.ayah_token_end.as_slice()[end_ayah as usize] as i32;
                let passage_center = (passage_start + passage_end) / 2;
                let half = (m / 2) as i32;
                let ref_start = (passage_center - half - window_margin_tokens as i32).max(0) as usize;
                let ref_end = (passage_center + (m as i32 - half) + window_margin_tokens as i32)
                    .min(total_tokens as i32)
                    .max(ref_start as i32) as usize;
                let expected_offset = passage_center - half;
                candidates.push(CandidateWindow {
                    ref_start,
                    ref_end,
                    expected_offset,
                    source: "simhash",
                    hint: format!("dist={dist},rank={rank}"),
                    quick_score: 0.0,
                });
        }
    }

    // 2B: Anchor-offset histogram.
    let mut token_positions: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    for (i, &tid) in obs_ids.iter().enumerate() {
        if tid == 0 {
            continue;
        }
        token_positions.entry(tid).or_default().push(i);
    }

    // Rank transcript tokens by rarity (IDF), excluding stop tokens.
    let idf = assets.token_idf.as_slice();
    let mut ranked: Vec<(f64, u32)> = Vec::new();
    for (&tid, _pos) in &token_positions {
        let tid_usize = tid as usize;
        if tid_usize >= assets.stop_mask.len() || assets.stop_mask[tid_usize] {
            continue;
        }
        let w = idf.get(tid_usize).copied().unwrap_or(0.0) as f64;
        if w <= 0.0 {
            continue;
        }
        ranked.push((w, tid));
    }
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let inv_offsets = assets.inv_offsets.as_slice();
    let inv_positions = assets.inv_positions.as_slice();
    let mut offset_votes: FxHashMap<i32, f64> = FxHashMap::default();

    for (used, (_idf_w, tid)) in ranked.into_iter().enumerate() {
        if used >= anchor_token_budget {
            break;
        }
        let pos_list = token_positions.get(&tid).cloned().unwrap_or_default();
        if pos_list.is_empty() {
            continue;
        }
        let mut picks: Vec<usize> = vec![pos_list[0]];
        if pos_list.len() > 2 {
            picks.push(pos_list[pos_list.len() / 2]);
        }
        if pos_list.len() > 1 {
            picks.push(pos_list[pos_list.len() - 1]);
        }
        picks.sort_unstable();
        picks.dedup();

        let tid_usize = tid as usize;
        if tid_usize + 1 >= inv_offsets.len() {
            continue;
        }
        let off0 = inv_offsets[tid_usize] as usize;
        let off1 = inv_offsets[tid_usize + 1] as usize;
        if off0 >= off1 || off1 > inv_positions.len() {
            continue;
        }
        let occ = &inv_positions[off0..off1];
        let step = if occ.len() > 2000 {
            (occ.len() / 2000).max(1)
        } else {
            1
        };

        let tid_idf = idf.get(tid_usize).copied().unwrap_or(0.0) as f64;
        for tp in picks {
            for &ref_pos in occ.iter().step_by(step) {
                let off = ref_pos as i32 - tp as i32;
                *offset_votes.entry(off).or_insert(0.0) += tid_idf;
            }
        }
    }

    if !offset_votes.is_empty() {
        let mut peaks: Vec<(i32, f64)> = offset_votes.into_iter().collect();
        peaks.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.abs().cmp(&b.0.abs()))
        });

        let mut chosen: Vec<i32> = Vec::new();
        for (off, _score) in peaks {
            if chosen.iter().any(|o2| (off - *o2).abs() < 25) {
                continue;
            }
            chosen.push(off);
            if chosen.len() >= top_anchor_offsets {
                break;
            }
        }

        for off in chosen {
            let ref_start = (off - window_margin_tokens as i32).max(0) as usize;
            let ref_end = (off + m as i32 + window_margin_tokens as i32)
                .min(total_tokens as i32)
                .max(ref_start as i32) as usize;
            candidates.push(CandidateWindow {
                ref_start,
                ref_end,
                expected_offset: off,
                source: "anchor",
                hint: String::new(),
                quick_score: 0.0,
            });
        }
    }

    // Dedup by coarse window signature.
    let mut out: FxHashMap<(i32, i32), CandidateWindow> = FxHashMap::default();
    let prio: FxHashMap<&'static str, i32> = FxHashMap::from_iter([("anchor", 3), ("embed", 2), ("simhash", 1)]);
    for c in candidates {
        if c.ref_end <= c.ref_start {
            continue;
        }
        let key = ((c.ref_start / 256) as i32, (c.expected_offset / 256) as i32);
        match out.get(&key) {
            None => {
                out.insert(key, c);
            }
            Some(prev) => {
                let p_new = prio.get(c.source).copied().unwrap_or(0);
                let p_old = prio.get(prev.source).copied().unwrap_or(0);
                if p_new > p_old {
                    out.insert(key, c);
                }
            }
        }
    }
    out.into_values().collect()
}

pub fn guess_ayah_alignment_v2_from_transcription(
    transcription: &Value,
    transcript_text: &str,
    assets: &AlignmentAssetsV2,
    time_base: &str,
    word_time_upgrade: &str,
    max_transcript_tokens: usize,
) -> Value {
    let t0 = Instant::now();

    let mut timed_tokens = extract_timed_tokens_from_transcription(transcription);
    if timed_tokens.is_empty() {
        return json!({"error":"no tokens found in transcription"});
    }
    match upgrade_word_timestamps(&timed_tokens, word_time_upgrade) {
        Ok(v) => timed_tokens = v,
        Err(e) => return json!({"error": format!("word_time_upgrade failed: {e}")}),
    }

    let m = timed_tokens.len();
    if max_transcript_tokens > 0 && m > max_transcript_tokens {
        return json!({
            "error":"transcript too long for alignment",
            "transcript_tokens": m,
            "max_transcript_tokens": max_transcript_tokens,
        });
    }

    let mut obs_ids: Vec<u32> = Vec::with_capacity(m);
    for t in &timed_tokens {
        let id = assets.token_to_id.get(&t.simplified).copied().unwrap_or(0);
        obs_ids.push(id);
    }

    let norm_text = normalize_arabic(transcript_text);
    let query_text = if !norm_text.is_empty() && norm_text.split_whitespace().count() <= 600 {
        norm_text
    } else {
        build_query_text(&timed_tokens, 600)
    };

    // Precompute top observed token IDs for quick scoring.
    let mut seen: HashSet<u32> = HashSet::new();
    let mut obs_unique: Vec<u32> = Vec::new();
    for &tid in &obs_ids {
        if tid == 0 || seen.contains(&tid) {
            continue;
        }
        seen.insert(tid);
        let tid_usize = tid as usize;
        if tid_usize < assets.stop_mask.len() && assets.stop_mask[tid_usize] {
            continue;
        }
        obs_unique.push(tid);
    }
    let obs_unique_total = obs_unique.len();
    let idf = assets.token_idf.as_slice();
    obs_unique.sort_by(|a, b| {
        let ia = idf.get(*a as usize).copied().unwrap_or(0.0) as f64;
        let ib = idf.get(*b as usize).copied().unwrap_or(0.0) as f64;
        ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
    });
    let obs_top: Vec<u32> = obs_unique.into_iter().take(64).collect();
    let obs_top_set: HashSet<u32> = obs_top.iter().copied().collect();
    let mut obs_top_idf_sum = 0.0_f64;
    for &tid in &obs_top {
        obs_top_idf_sum += idf.get(tid as usize).copied().unwrap_or(0.0) as f64;
    }
    if obs_top_idf_sum <= 0.0 {
        obs_top_idf_sum = 1.0;
    }

    let debug = if env_bool("AYAH_ALIGN_V2_DEBUG") {
        let mut obs_nonzero = 0usize;
        let mut obs_stop = 0usize;
        for &tid in &obs_ids {
            if tid == 0 {
                continue;
            }
            obs_nonzero += 1;
            let tid_usize = tid as usize;
            if tid_usize < assets.stop_mask.len() && assets.stop_mask[tid_usize] {
                obs_stop += 1;
            }
        }
        json!({
            "timed_tokens": m,
            "obs_nonzero": obs_nonzero,
            "obs_unknown": m.saturating_sub(obs_nonzero),
            "obs_stop": obs_stop,
            "obs_effective": obs_nonzero.saturating_sub(obs_stop),
            "obs_unique_effective": obs_unique_total,
            "query_words": query_text.split_whitespace().count(),
            "query_chars": query_text.chars().count(),
        })
    } else {
        Value::Null
    };

    let mut candidates = generate_candidates(
        assets,
        &obs_ids,
        &query_text,
        80,
        12,
        28,
        10,
        450,
    );
    if candidates.is_empty() {
        return json!({"error":"no candidates generated"});
    }

    // Cheap scoring + prune.
    for c in &mut candidates {
        let ref_slice = &assets.ref_token_ids.as_slice()[c.ref_start..c.ref_end];
        let mut seen_ref: HashSet<u32> = HashSet::new();
        let mut num = 0.0_f64;
        for &tid in ref_slice {
            if tid == 0 {
                continue;
            }
            if !obs_top_set.contains(&tid) {
                continue;
            }
            if !seen_ref.insert(tid) {
                continue;
            }
            num += idf.get(tid as usize).copied().unwrap_or(0.0) as f64;
        }
        c.quick_score = num / obs_top_idf_sum;
    }

    candidates.sort_by(|a, b| {
        b.quick_score
            .partial_cmp(&a.quick_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (a.ref_end - a.ref_start).cmp(&(b.ref_end - b.ref_start)))
    });

    // Ensure at least one candidate per source (when present).
    let max_keep = 10usize;
    let mut selected: Vec<CandidateWindow> = Vec::new();
    let mut seen_key: HashSet<(usize, usize, i32)> = HashSet::new();

    for src in ["anchor", "embed", "simhash"] {
        if selected.len() >= max_keep {
            break;
        }
        if let Some(c) = candidates.iter().find(|c| c.source == src) {
            add_candidate(&mut selected, &mut seen_key, c);
        }
    }
    for c in &candidates {
        if selected.len() >= max_keep {
            break;
        }
        add_candidate(&mut selected, &mut seen_key, c);
    }

    let candidates = selected;

    // DP per candidate.
    //
    // For larger transcripts, candidate DP dominates CPU time. Parallelize safely across candidates
    // to reduce latency on multi-core machines.
    let band = ((m as f64 * 0.20).round() as i32 + 60).clamp(120, 700) as usize;
    let use_parallel = candidates.len() >= 4 && m >= 120;
    let mut results: Vec<AlignmentResultV2> = if use_parallel {
        candidates
            .into_par_iter()
            .map(|c| {
                let ref_slice = &assets.ref_token_ids.as_slice()[c.ref_start..c.ref_end];
                let mut align = weighted_banded_dp(
                    &obs_ids,
                    ref_slice,
                    c.ref_start,
                    c.expected_offset,
                    band,
                    assets.confusions.as_ref(),
                    &assets.id_to_token,
                );
                align.expected_offset = c.expected_offset;
                align.candidate_source = c.source.to_string();
                align.quick_score = c.quick_score;
                align
            })
            .collect()
    } else {
        let mut results: Vec<AlignmentResultV2> = Vec::with_capacity(candidates.len());
        for c in candidates {
            let ref_slice = &assets.ref_token_ids.as_slice()[c.ref_start..c.ref_end];
            let mut align = weighted_banded_dp(
                &obs_ids,
                ref_slice,
                c.ref_start,
                c.expected_offset,
                band,
                assets.confusions.as_ref(),
                &assets.id_to_token,
            );
            align.expected_offset = c.expected_offset;
            align.candidate_source = c.source.to_string();
            align.quick_score = c.quick_score;
            results.push(align);
        }
        results
    };
    if results.is_empty() {
        return json!({"error":"no candidates could be aligned"});
    }

    let norm_cost = |r: &AlignmentResultV2| r.cost / (m.max(1) as f64);
    results.sort_by(|a, b| {
        norm_cost(a)
            .partial_cmp(&norm_cost(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.matches.cmp(&a.matches))
    });
    let best = results[0].clone();
    let second = results.get(1).cloned();
    let best_norm = norm_cost(&best);
    let second_norm = second.as_ref().map(norm_cost).unwrap_or(best_norm + 1.0);
    let gap = (second_norm - best_norm).max(0.0);

    let match_ratio = best.matches as f64 / best.substitutions.max(1) as f64;
    let quick = best.quick_score;
    let gap_score = (gap / 0.35).min(1.0);
    let match_score = ((match_ratio - 0.20) / 0.65).clamp(0.0, 1.0);
    let quick_score = quick.clamp(0.0, 1.0);
    let base_confidence = 0.45 * match_score + 0.35 * gap_score + 0.20 * quick_score;
    // Guard against spuriously high confidence on extremely short transcripts (e.g., a single common word).
    // For long transcripts we keep the original calibration (evidence_score=1.0).
    let evidence_score = if best.substitutions < 20 {
        (best.substitutions as f64 / 12.0).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let confidence = base_confidence * evidence_score;

    let mut mapped_ref_idxs: Vec<usize> = best.transcript_to_ref.iter().flatten().copied().collect();
    let mut matched_ref_idxs: Vec<usize> = Vec::new();
    for (r_idx, is_match) in best.transcript_to_ref.iter().zip(best.transcript_is_match.iter()) {
        if let (Some(r), true) = (r_idx, *is_match) {
            matched_ref_idxs.push(*r);
        }
    }
    let (min_ref_idx, max_ref_idx) = if !matched_ref_idxs.is_empty() {
        (*matched_ref_idxs.iter().min().unwrap(), *matched_ref_idxs.iter().max().unwrap())
    } else if !mapped_ref_idxs.is_empty() {
        (*mapped_ref_idxs.iter().min().unwrap(), *mapped_ref_idxs.iter().max().unwrap())
    } else {
        return json!({"error":"alignment produced no mapped tokens"});
    };

    let min_ref_idx = min_ref_idx.min(assets.total_tokens().saturating_sub(1));
    let max_ref_idx = max_ref_idx.min(assets.total_tokens().saturating_sub(1));

    let start_surah = assets.ref_surah_id.as_slice()[min_ref_idx] as i32;
    let start_ayah = assets.ref_ayah_num.as_slice()[min_ref_idx] as i32;
    let end_surah = assets.ref_surah_id.as_slice()[max_ref_idx] as i32;
    let end_ayah = assets.ref_ayah_num.as_slice()[max_ref_idx] as i32;

    // Determine timestamp base.
    let mut base_s = 0.0_f64;
    let time_base_key = time_base.trim().to_lowercase();
    if time_base_key == "first_word" {
        if let Some(min_s) = timed_tokens.iter().filter_map(|t| t.start_s).min_by(|a, b| a.partial_cmp(b).unwrap()) {
            base_s = min_s;
        }
    } else if time_base_key == "first_ayah" {
        let mut base_candidates: Vec<f64> = Vec::new();
        for (i, r_idx) in best.transcript_to_ref.iter().enumerate() {
            let Some(r) = *r_idx else { continue };
            if r < min_ref_idx || r > max_ref_idx {
                continue;
            }
            if let Some(s) = timed_tokens[i].start_s {
                base_candidates.push(s);
            }
        }
        if let Some(min_s) = base_candidates.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            base_s = min_s;
        }
    }

    // Build ayah list across aligned span.
    let mut ayah_order: Vec<(i32, i32)> = Vec::new();
    for ridx in min_ref_idx..=max_ref_idx {
        let sid = assets.ref_surah_id.as_slice()[ridx] as i32;
        let anum = assets.ref_ayah_num.as_slice()[ridx] as i32;
        if ayah_order.last().copied() != Some((sid, anum)) {
            ayah_order.push((sid, anum));
        }
    }

    // Aggregate times per ayah.
    #[derive(Default, Clone)]
    struct AyahTime {
        start_s_real: Option<f64>,
        end_s_real: Option<f64>,
        start_s_est: Option<f64>,
        end_s_est: Option<f64>,
        token_count: usize,
        match_count: usize,
    }
    let mut ayah_times: BTreeMap<(i32, i32), AyahTime> = BTreeMap::new();

    let m = best.transcript_to_ref.len().min(timed_tokens.len());
    let mut token_ayah: Vec<Option<(i32, i32)>> = vec![None; m];
    for (i, r_idx) in best.transcript_to_ref.iter().take(m).enumerate() {
        let Some(r) = *r_idx else { continue };
        if r < min_ref_idx || r > max_ref_idx {
            continue;
        }
        let sid = assets.ref_surah_id.as_slice()[r] as i32;
        let anum = assets.ref_ayah_num.as_slice()[r] as i32;
        token_ayah[i] = Some((sid, anum));
    }

    let mut prev_ayah: Vec<Option<(i32, i32)>> = Vec::with_capacity(m);
    let mut prev_mapped_idx: Vec<Option<usize>> = Vec::with_capacity(m);
    let mut last: Option<(i32, i32)> = None;
    let mut last_idx: Option<usize> = None;
    for i in 0..m {
        if let Some(ay) = token_ayah[i] {
            last = Some(ay);
            last_idx = Some(i);
        }
        prev_ayah.push(last);
        prev_mapped_idx.push(last_idx);
    }
    let mut next_ayah: Vec<Option<(i32, i32)>> = vec![None; m];
    let mut next_mapped_idx: Vec<Option<usize>> = vec![None; m];
    let mut last_next: Option<(i32, i32)> = None;
    let mut last_next_idx: Option<usize> = None;
    for i in (0..m).rev() {
        if let Some(ay) = token_ayah[i] {
            last_next = Some(ay);
            last_next_idx = Some(i);
        }
        next_ayah[i] = last_next;
        next_mapped_idx[i] = last_next_idx;
    }

    let gap_assign_next_s = 0.75_f64;
    let mut token_in_ayah_cache: HashMap<(i32, i32, u32), bool> = HashMap::new();
    let mut ayah_contains_token_id = |sid: i32, anum: i32, tid: u32| -> bool {
        let key = (sid, anum, tid);
        if let Some(v) = token_in_ayah_cache.get(&key) {
            return *v;
        }
        if sid <= 0 || anum <= 0 {
            token_in_ayah_cache.insert(key, false);
            return false;
        }
        let Some(g) = assets.ayah_key_to_global.get(&(sid as u16, anum as u16)) else {
            token_in_ayah_cache.insert(key, false);
            return false;
        };
        let g = *g as usize;
        let starts = assets.ayah_token_start.as_slice();
        let ends = assets.ayah_token_end.as_slice();
        let Some(&st_u32) = starts.get(g) else {
            token_in_ayah_cache.insert(key, false);
            return false;
        };
        let Some(&en_u32) = ends.get(g) else {
            token_in_ayah_cache.insert(key, false);
            return false;
        };
        let st = st_u32 as usize;
        let en = en_u32 as usize;
        let ref_ids = assets.ref_token_ids.as_slice();
        let ok = if st < en && en <= ref_ids.len() {
            ref_ids[st..en].iter().any(|x| *x == tid)
        } else {
            false
        };
        token_in_ayah_cache.insert(key, ok);
        ok
    };

    for i in 0..m {
        let assigned = if let Some(ay) = token_ayah[i] {
            Some(ay)
        } else {
            match (prev_ayah[i], next_ayah[i]) {
                (None, None) => None,
                (None, Some(n)) => {
                    if timed_tokens[i].estimated_time {
                        None
                    } else {
                        let tid = assets
                            .token_to_id
                            .get(timed_tokens[i].simplified.as_str())
                            .copied();
                        if let Some(tid) = tid {
                            if ayah_contains_token_id(n.0, n.1, tid) {
                                Some(n)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                }
                (Some(p), None) => {
                    if timed_tokens[i].estimated_time {
                        None
                    } else {
                        let tid = assets
                            .token_to_id
                            .get(timed_tokens[i].simplified.as_str())
                            .copied();
                        if let Some(tid) = tid {
                            if ayah_contains_token_id(p.0, p.1, tid) {
                                Some(p)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                }
                (Some(p), Some(n)) => {
                    if p == n {
                        Some(p)
                    } else if timed_tokens[i].estimated_time {
                        None
                    } else {
                        let tid = assets
                            .token_to_id
                            .get(timed_tokens[i].simplified.as_str())
                            .copied();
                        if let Some(tid) = tid {
                            let prev_has = ayah_contains_token_id(p.0, p.1, tid);
                            let next_has = ayah_contains_token_id(n.0, n.1, tid);
                            if prev_has && !next_has {
                                Some(p)
                            } else if next_has && !prev_has {
                                Some(n)
                            } else {
                                let next_idx = next_mapped_idx[i];
                                let next_start = next_idx.and_then(|j| timed_tokens[j].start_s.or(timed_tokens[j].end_s));
                                let tok_end = timed_tokens[i].end_s.or(timed_tokens[i].start_s);
                                if let (Some(ns), Some(te)) = (next_start, tok_end) {
                                    if (ns - te) > gap_assign_next_s {
                                        Some(n)
                                    } else {
                                        Some(p)
                                    }
                                } else {
                                    Some(p)
                                }
                            }
                        } else {
                            let next_idx = next_mapped_idx[i];
                            let next_start = next_idx.and_then(|j| timed_tokens[j].start_s.or(timed_tokens[j].end_s));
                            let tok_end = timed_tokens[i].end_s.or(timed_tokens[i].start_s);
                            if let (Some(ns), Some(te)) = (next_start, tok_end) {
                                if (ns - te) > gap_assign_next_s {
                                    Some(n)
                                } else {
                                    Some(p)
                                }
                            } else {
                                Some(p)
                            }
                        }
                    }
                }
            }
        };
        let Some((sid, anum)) = assigned else { continue };
        let entry = ayah_times.entry((sid, anum)).or_default();
        let is_est = timed_tokens[i].estimated_time;
        if let Some(s) = timed_tokens[i].start_s {
            if is_est {
                entry.start_s_est = Some(entry.start_s_est.map(|x| x.min(s)).unwrap_or(s));
            } else {
                entry.start_s_real = Some(entry.start_s_real.map(|x| x.min(s)).unwrap_or(s));
            }
        }
        if let Some(e) = timed_tokens[i].end_s {
            if is_est {
                entry.end_s_est = Some(entry.end_s_est.map(|x| x.max(e)).unwrap_or(e));
            } else {
                entry.end_s_real = Some(entry.end_s_real.map(|x| x.max(e)).unwrap_or(e));
            }
        }
        entry.token_count += 1;
        if token_ayah[i].is_some() && best.transcript_is_match.get(i).copied().unwrap_or(false) {
            entry.match_count += 1;
        }
    }

    let mut segments: Vec<Value> = Vec::new();
    for (sid, anum) in &ayah_order {
        let info = ayah_times.get(&(*sid, *anum)).cloned().unwrap_or_default();
        let start_s = info.start_s_real.or(info.start_s_est);
        let end_s = info.end_s_real.or(info.end_s_est);
        let start_rel = start_s.map(|s| s - base_s);
        let end_rel = end_s.map(|e| e - base_s);
        segments.push(json!({
            "ayah_key": format!("{sid}:{anum}"),
            "surah_id": sid,
            "ayah_num": anum,
            "start_s": start_rel,
            "end_s": end_rel,
            "start_ts": start_rel.map(format_timestamp),
            "end_ts": end_rel.map(format_timestamp),
            "token_count": info.token_count,
            "match_count": info.match_count,
            "estimated_time": info.start_s_real.is_none() || info.end_s_real.is_none(),
        }));
    }

    // Fill missing timestamps and enforce monotonicity.
    let first_non_none = |vals: &[Value]| -> Option<f64> {
        for v in vals {
            if let Some(f) = v.as_f64() {
                return Some(f);
            }
        }
        None
    };

    for idx in 0..segments.len() {
        let (mut start_s, mut end_s) = {
            let seg = &segments[idx];
            (seg.get("start_s").and_then(|v| v.as_f64()), seg.get("end_s").and_then(|v| v.as_f64()))
        };
        if start_s.is_some() && end_s.is_some() {
            continue;
        }

        let prev_end = if idx > 0 {
            segments[idx - 1].get("end_s").and_then(|v| v.as_f64())
        } else {
            None
        };
        let next_starts: Vec<Value> = segments[idx + 1..]
            .iter()
            .filter_map(|s| s.get("start_s").cloned())
            .filter(|v| !v.is_null())
            .collect();
        let next_start = first_non_none(&next_starts);

        if start_s.is_none() {
            start_s = prev_end.or(next_start);
        }
        if end_s.is_none() {
            end_s = next_start.or(start_s);
        }

        let start_ts = start_s.map(format_timestamp);
        let end_ts = end_s.map(format_timestamp);
        if let Some(obj) = segments[idx].as_object_mut() {
            obj.insert("start_s".to_string(), json!(start_s));
            obj.insert("end_s".to_string(), json!(end_s));
            obj.insert("start_ts".to_string(), json!(start_ts));
            obj.insert("end_ts".to_string(), json!(end_ts));
            obj.insert("estimated_time".to_string(), json!(true));
        }
    }

    let mut prev_end: Option<f64> = None;
    for seg in &mut segments {
        let mut s = seg.get("start_s").and_then(|v| v.as_f64());
        let mut e = seg.get("end_s").and_then(|v| v.as_f64());
        if let (Some(ss), Some(pe)) = (s, prev_end) {
            if ss < pe {
                s = Some(pe);
            }
        }
        if let (Some(ss), Some(ee)) = (s, e) {
            if ee < ss {
                e = Some(ss);
            }
        }
        if let Some(obj) = seg.as_object_mut() {
            obj.insert("start_s".to_string(), json!(s));
            obj.insert("end_s".to_string(), json!(e));
            obj.insert("start_ts".to_string(), json!(s.map(format_timestamp)));
            obj.insert("end_ts".to_string(), json!(e.map(format_timestamp)));
        }
        if e.is_some() {
            prev_end = e;
        }
    }

    let t_total = t0.elapsed().as_secs_f64();
    json!({
        "method": "v2_embedding+anchors+noisy_dp",
        "version": assets.version,
        "confidence": confidence,
        "start": {"ayah_key": format!("{start_surah}:{start_ayah}"), "surah_id": start_surah, "ayah_num": start_ayah},
        "end": {"ayah_key": format!("{end_surah}:{end_ayah}"), "surah_id": end_surah, "ayah_num": end_ayah},
        "time_base": time_base_key,
        "time_base_s": base_s,
        "debug": debug,
        "query": {
            "candidates": results.len(),
            "best_norm_cost": best_norm,
            "second_norm_cost": second_norm,
            "gap": gap,
            "quick_score": best.quick_score,
            "candidate_source": best.candidate_source,
        },
        "alignment": {
            "cost": best.cost,
            "norm_cost": best_norm,
            "matches": best.matches,
            "substitutions": best.substitutions,
            "deletions": best.deletions,
            "insertions": best.insertions,
            "match_ratio": match_ratio,
        },
        "timing": {"total_s": t_total},
        "segments": segments,
    })
}

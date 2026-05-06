use serde_json::{json, Value};

use crate::alignment_v2::{guess_ayah_alignment_v2_from_transcription, AlignmentAssetsV2};

#[derive(Debug, Clone)]
struct SegAlign {
    seg_index: usize,
    seg_start_s: f64,
    seg_end_s: f64,
    start_surah_id: u16,
    end_surah_id: u16,
    start_global: u16,
    end_global: u16,
    confidence: f64,
}

impl SegAlign {
    fn center_global(&self) -> u16 {
        if self.start_global <= self.end_global {
            self.start_global + (self.end_global - self.start_global) / 2
        } else {
            self.end_global + (self.start_global - self.end_global) / 2
        }
    }
}

#[derive(Debug, Clone)]
struct SpanCandidate {
    start_seg_index: usize,
    end_seg_index: usize,
}

#[derive(Debug, Clone)]
struct JumpEvent {
    at_s: f64,
    at_segment_index: usize,
    from_span_index: usize,
    to_span_index: usize,
    distance_ayahs: i32,
}

fn segment_text(seg: &Value) -> String {
    if let Some(t) = seg.get("text").and_then(|v| v.as_str()) {
        let s = t.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }
    let mut parts: Vec<String> = Vec::new();
    if let Some(words) = seg.get("words").and_then(|v| v.as_array()) {
        for w in words {
            if let Some(t) = w.get("word").and_then(|v| v.as_str()) {
                let s = t.trim();
                if !s.is_empty() {
                    parts.push(s.to_string());
                }
            }
        }
    }
    parts.join(" ").trim().to_string()
}

fn seg_align_one(
    seg_index: usize,
    seg: &Value,
    assets: &AlignmentAssetsV2,
    word_time_upgrade: &str,
) -> Option<SegAlign> {
    let seg_start_s = seg.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let seg_end_s = seg.get("end_s").and_then(|v| v.as_f64()).unwrap_or(seg_start_s);
    let text = segment_text(seg);
    if text.is_empty() {
        return None;
    }
    // Very short segments (e.g., a single trailing word) are often ambiguous and can cause
    // spurious "jumps" to far-away surahs. Skip them for span detection.
    let word_count = text.split_whitespace().count();
    if word_count < 3 {
        return None;
    }

    let t = json!({"segments":[seg.clone()]});
    let a = guess_ayah_alignment_v2_from_transcription(&t, &text, assets, "absolute", word_time_upgrade, 2000);
    if a.get("error").is_some() {
        return None;
    }

    let conf = a.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let start = a.get("start")?.as_object()?;
    let end = a.get("end")?.as_object()?;
    let sid0 = start.get("surah_id")?.as_u64()? as u16;
    let anum0 = start.get("ayah_num")?.as_u64()? as u16;
    let sid1 = end.get("surah_id")?.as_u64()? as u16;
    let anum1 = end.get("ayah_num")?.as_u64()? as u16;

    let g0 = *assets.ayah_key_to_global.get(&(sid0, anum0))?;
    let g1 = *assets.ayah_key_to_global.get(&(sid1, anum1))?;

    Some(SegAlign {
        seg_index,
        seg_start_s,
        seg_end_s,
        start_surah_id: sid0,
        end_surah_id: sid1,
        start_global: g0,
        end_global: g1,
        confidence: conf,
    })
}

#[derive(Debug, Clone)]
struct PendingJump {
    start_seg_index: usize,
    pos_global: u16,
    surah_id: u16,
    confidence: f64,
    count: usize,
    from_span_index: usize,
}

pub struct MultiSpanOutput {
    pub active_alignment: Value,
    pub spans: Value,
    pub jumps: Value,
    pub active_span_index: Option<usize>,
}

pub fn align_multi_span(
    transcription: &Value,
    transcript_text: &str,
    assets: &AlignmentAssetsV2,
    word_time_upgrade: &str,
) -> MultiSpanOutput {
    let segs = transcription
        .get("segments")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    if segs.is_empty() {
        let a = guess_ayah_alignment_v2_from_transcription(
            transcription,
            transcript_text,
            assets,
            "absolute",
            word_time_upgrade,
            10_000,
        );
        return MultiSpanOutput {
            active_alignment: a,
            spans: Value::Null,
            jumps: Value::Null,
            active_span_index: None,
        };
    }

    // Parameters: tuned to be conservative to avoid false span splits.
    let seg_min_conf = 0.30_f64;
    let very_conf = 0.93_f64;
    let new_span_dist_ayahs = 80_i32;
    let pending_confirm = 2_usize;
    let pending_cluster_dist_ayahs = 80_i32;
    let tail_force_segments = 1_usize; // allow a single far tail segment to start a new span

    let mut seg_aligns: Vec<Option<SegAlign>> = Vec::with_capacity(segs.len());
    for (i, seg) in segs.iter().enumerate() {
        seg_aligns.push(seg_align_one(i, seg, assets, word_time_upgrade));
    }

    let mut spans: Vec<SpanCandidate> = vec![SpanCandidate {
        start_seg_index: 0,
        end_seg_index: 0,
    }];
    let mut jumps: Vec<JumpEvent> = Vec::new();

    let mut cur_span = 0usize;
    let mut last_good_pos: Option<u16> = None;
    let mut last_good_surah: Option<u16> = None;
    let mut pending: Option<PendingJump> = None;

    for i in 0..segs.len() {
        spans[cur_span].end_seg_index = i;

        let Some(sa) = seg_aligns[i]
            .as_ref()
            .filter(|a| a.confidence >= seg_min_conf)
        else {
            continue;
        };

        let pos = sa.center_global();
        let Some(last) = last_good_pos else {
            last_good_pos = Some(pos);
            last_good_surah = Some(sa.start_surah_id);
            pending = None;
            continue;
        };

        let dist = (pos as i32 - last as i32).abs();
        let surah_gap = match last_good_surah {
            None => 0,
            Some(s) => (sa.start_surah_id as i32 - s as i32).abs(),
        };
        let is_far = dist >= new_span_dist_ayahs || surah_gap >= 2;

        if !is_far {
            last_good_pos = Some(pos);
            last_good_surah = Some(sa.start_surah_id);
            pending = None;
            continue;
        }

        // Far away: candidate for new span.
        if sa.confidence >= very_conf {
            let new_start = i;
            spans[cur_span].end_seg_index = new_start.saturating_sub(1);
            spans.push(SpanCandidate {
                start_seg_index: new_start,
                end_seg_index: i,
            });
            let to_span = spans.len() - 1;
            jumps.push(JumpEvent {
                at_s: sa.seg_start_s,
                at_segment_index: new_start,
                from_span_index: cur_span,
                to_span_index: to_span,
                distance_ayahs: dist,
            });
            cur_span = to_span;
            last_good_pos = Some(pos);
            last_good_surah = Some(sa.start_surah_id);
            pending = None;
            continue;
        }

        match pending.as_mut() {
            None => {
                pending = Some(PendingJump {
                    start_seg_index: i,
                    pos_global: pos,
                    surah_id: sa.start_surah_id,
                    confidence: sa.confidence,
                    count: 1,
                    from_span_index: cur_span,
                });
            }
            Some(p) => {
                let dist2 = (pos as i32 - p.pos_global as i32).abs();
                if dist2 <= pending_cluster_dist_ayahs {
                    p.count += 1;
                    if p.count >= pending_confirm {
                        let new_start = p.start_seg_index;
                        let from_span = p.from_span_index;
                        spans[from_span].end_seg_index = new_start.saturating_sub(1);
                        spans.push(SpanCandidate {
                            start_seg_index: new_start,
                            end_seg_index: i,
                        });
                        let to_span = spans.len() - 1;
                        jumps.push(JumpEvent {
                            at_s: sa.seg_start_s,
                            at_segment_index: new_start,
                            from_span_index: from_span,
                            to_span_index: to_span,
                            distance_ayahs: dist,
                        });
                        cur_span = to_span;
                        last_good_pos = Some(pos);
                        last_good_surah = Some(sa.start_surah_id);
                        pending = None;
                    }
                } else {
                    *p = PendingJump {
                        start_seg_index: i,
                        pos_global: pos,
                        surah_id: sa.start_surah_id,
                        confidence: sa.confidence,
                        count: 1,
                        from_span_index: cur_span,
                    };
                }
            }
        }
    }

    // If a far jump happens at the tail and we only see one good segment there, still split.
    if let (Some(p), Some(last_pos), Some(last_surah)) = (pending.take(), last_good_pos, last_good_surah) {
        let remaining = segs.len().saturating_sub(p.start_seg_index);
        let dist = (p.pos_global as i32 - last_pos as i32).abs();
        let surah_gap = (p.surah_id as i32 - last_surah as i32).abs();
        let is_far = dist >= new_span_dist_ayahs || surah_gap >= 2;
        if remaining <= tail_force_segments && is_far && p.confidence >= seg_min_conf {
            let new_start = p.start_seg_index;
            let from_span = p.from_span_index;
            let new_end = segs.len().saturating_sub(1);
            if new_start > spans[from_span].start_seg_index && new_start <= new_end {
                spans[from_span].end_seg_index = new_start.saturating_sub(1);
                spans.push(SpanCandidate {
                    start_seg_index: new_start,
                    end_seg_index: new_end,
                });
                let to_span = spans.len() - 1;
                jumps.push(JumpEvent {
                    at_s: segs
                        .get(new_start)
                        .and_then(|v| v.get("start_s"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    at_segment_index: new_start,
                    from_span_index: from_span,
                    to_span_index: to_span,
                    distance_ayahs: dist,
                });
            }
        }
    }

    // Drop any empty spans and keep jump indexes consistent.
    let mut index_map: Vec<Option<usize>> = vec![None; spans.len()];
    let mut spans2: Vec<SpanCandidate> = Vec::new();
    for (old_idx, sp) in spans.into_iter().enumerate() {
        if sp.end_seg_index >= sp.start_seg_index {
            index_map[old_idx] = Some(spans2.len());
            spans2.push(sp);
        }
    }
    let mut jumps2: Vec<JumpEvent> = Vec::new();
    for j in jumps {
        let Some(from) = index_map.get(j.from_span_index).and_then(|v| *v) else {
            continue;
        };
        let Some(to) = index_map.get(j.to_span_index).and_then(|v| *v) else {
            continue;
        };
        jumps2.push(JumpEvent {
            at_s: j.at_s,
            at_segment_index: j.at_segment_index,
            from_span_index: from,
            to_span_index: to,
            distance_ayahs: j.distance_ayahs,
        });
    }
    let mut spans = spans2;
    let mut jumps = jumps2;

    // Refine each span by aligning only its segment range.
    let mut span_alignments: Vec<Value> = Vec::with_capacity(spans.len());
    for sp in &spans {
        let mut subset: Vec<Value> = Vec::new();
        let mut text_parts: Vec<String> = Vec::new();
        for seg in &segs[sp.start_seg_index..=sp.end_seg_index] {
            let t = segment_text(seg);
            if !t.is_empty() {
                text_parts.push(t);
            }
            subset.push(seg.clone());
        }
        let sub_text = if text_parts.is_empty() {
            transcript_text.to_string()
        } else {
            text_parts.join(" ")
        };
        let sub_transcription = json!({"segments": subset});
        let a = guess_ayah_alignment_v2_from_transcription(
            &sub_transcription,
            &sub_text,
            assets,
            "absolute",
            word_time_upgrade,
            10_000,
        );
        span_alignments.push(a);
    }

    // Heuristic: drop a tiny low-confidence leading span. This is common when the first VAD segment
    // contains non-Quran intro/noise and misaligns far away, producing a bogus "jump".
    while spans.len() >= 2 {
        let sp0 = &spans[0];
        let seg_count0 = sp0.end_seg_index.saturating_sub(sp0.start_seg_index) + 1;
        if seg_count0 > 1 {
            break;
        }

        let start_s0 = segs
            .get(sp0.start_seg_index)
            .and_then(|v| v.get("start_s"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let end_s0 = segs
            .get(sp0.end_seg_index)
            .and_then(|v| v.get("end_s"))
            .and_then(|v| v.as_f64())
            .unwrap_or(start_s0);
        let dur0 = (end_s0 - start_s0).max(0.0);
        if dur0 > 15.0 {
            break;
        }

        let conf0 = span_alignments[0]
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        if conf0 >= 0.75 {
            break;
        }

        let sp1 = &spans[1];
        let start_s1 = segs
            .get(sp1.start_seg_index)
            .and_then(|v| v.get("start_s"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let end_s_all = segs
            .get(spans.last().map(|s| s.end_seg_index).unwrap_or(0))
            .and_then(|v| v.get("end_s"))
            .and_then(|v| v.as_f64())
            .unwrap_or(start_s1);
        let dur_tail = (end_s_all - start_s1).max(0.0);

        // Only drop if the remainder is meaningfully longer than the intro span.
        if dur_tail < (dur0 * 3.0).max(10.0) {
            break;
        }

        spans.remove(0);
        span_alignments.remove(0);
        jumps.retain(|j| j.from_span_index != 0 && j.to_span_index != 0);
        for j in &mut jumps {
            j.from_span_index = j.from_span_index.saturating_sub(1);
            j.to_span_index = j.to_span_index.saturating_sub(1);
        }
    }

    // Active span: last span with a valid alignment.
    let mut active_span_index: Option<usize> = None;
    for i in (0..span_alignments.len()).rev() {
        if span_alignments[i].get("error").is_none() {
            active_span_index = Some(i);
            break;
        }
    }

    // Guard against a tiny low-evidence trailing span becoming "active" due to an ambiguous last segment.
    // Prefer the previous span when the tail span has too few aligned tokens.
    if let Some(i) = active_span_index {
        if i > 0 {
            let evidence = |a: &Value| -> (usize, usize, f64) {
                let conf = a.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let align = a.get("alignment").and_then(|v| v.as_object());
                let matches = align
                    .and_then(|o| o.get("matches"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let subs = align
                    .and_then(|o| o.get("substitutions"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                (matches, subs, conf)
            };
            let (m_last, s_last, c_last) = evidence(&span_alignments[i]);
            let (m_prev, s_prev, _c_prev) = evidence(&span_alignments[i - 1]);
            let last_is_low_evidence = s_last < 8 || m_last < 5;
            let prev_is_better = s_prev >= 8 && m_prev >= 5;
            if last_is_low_evidence && prev_is_better && c_last < 0.85 {
                active_span_index = Some(i - 1);
            }
        }
    }
    let active_alignment = match active_span_index {
        Some(i) => span_alignments[i].clone(),
        None => span_alignments.last().cloned().unwrap_or_else(|| Value::Null),
    };

    let mut spans_out: Vec<Value> = Vec::with_capacity(spans.len());
    for (idx, sp) in spans.iter().enumerate() {
        let start_s = segs
            .get(sp.start_seg_index)
            .and_then(|v| v.get("start_s"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let end_s = segs
            .get(sp.end_seg_index)
            .and_then(|v| v.get("end_s"))
            .and_then(|v| v.as_f64())
            .unwrap_or(start_s);

        let a = &span_alignments[idx];
        spans_out.push(json!({
            "start": a.get("start").cloned().unwrap_or(Value::Null),
            "end": a.get("end").cloned().unwrap_or(Value::Null),
            "start_s": start_s,
            "end_s": end_s,
            "confidence": a.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "segment_indexes": (sp.start_seg_index..=sp.end_seg_index).collect::<Vec<usize>>(),
        }));
    }

    let jumps_out: Vec<Value> = jumps
        .into_iter()
        .map(|j| {
            json!({
                "at_s": j.at_s,
                "at_segment_index": j.at_segment_index,
                "from_span_index": j.from_span_index,
                "to_span_index": j.to_span_index,
                "distance_ayahs": j.distance_ayahs,
            })
        })
        .collect();

    MultiSpanOutput {
        active_alignment,
        spans: Value::Array(spans_out),
        jumps: Value::Array(jumps_out),
        active_span_index,
    }
}

use std::cmp::Ordering as CmpOrdering;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Context;
use futures_util::{SinkExt, StreamExt};
use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{broadcast, Mutex, Notify, Semaphore};
use tokio::time::{interval, Duration, MissedTickBehavior};
use tracing::error;

use crate::alignment_v2::AlignmentAssetsV2;
use crate::multispan;
use crate::transcriber_pool::TranscriberPool;

struct RingBuf {
    buf: Vec<u8>,
    cap: usize,
    write_pos: usize,
    len: usize,
}

impl RingBuf {
    fn new(cap: usize) -> Self {
        Self {
            buf: vec![0u8; cap.max(1)],
            cap: cap.max(1),
            write_pos: 0,
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        if data.len() >= self.cap {
            let tail = &data[data.len() - self.cap..];
            self.buf.copy_from_slice(tail);
            self.write_pos = 0;
            self.len = self.cap;
            return;
        }

        let mut remaining = data;
        while !remaining.is_empty() {
            let to_end = self.cap - self.write_pos;
            let n = to_end.min(remaining.len());
            self.buf[self.write_pos..self.write_pos + n].copy_from_slice(&remaining[..n]);
            self.write_pos = (self.write_pos + n) % self.cap;
            remaining = &remaining[n..];
        }

        self.len = (self.len + data.len()).min(self.cap);
    }

    fn copy_from_offset_to_end(&self, offset: usize) -> Vec<u8> {
        let offset = offset.min(self.len);
        let out_len = self.len - offset;
        let mut out = vec![0u8; out_len];

        let start_pos = (self.write_pos + self.cap - self.len) % self.cap;
        let mut pos = (start_pos + offset) % self.cap;
        let mut remaining = out_len;
        let mut out_idx = 0;
        while remaining > 0 {
            let to_end = self.cap - pos;
            let n = to_end.min(remaining);
            out[out_idx..out_idx + n].copy_from_slice(&self.buf[pos..pos + n]);
            pos = (pos + n) % self.cap;
            out_idx += n;
            remaining -= n;
        }

        out
    }

    fn copy_range(&self, offset: usize, len: usize) -> Vec<u8> {
        let offset = offset.min(self.len);
        let len = len.min(self.len.saturating_sub(offset));
        if len == 0 {
            return Vec::new();
        }

        let mut out = vec![0u8; len];

        let start_pos = (self.write_pos + self.cap - self.len) % self.cap;
        let mut pos = (start_pos + offset) % self.cap;
        let mut remaining = len;
        let mut out_idx = 0;
        while remaining > 0 {
            let to_end = self.cap - pos;
            let n = to_end.min(remaining);
            out[out_idx..out_idx + n].copy_from_slice(&self.buf[pos..pos + n]);
            pos = (pos + n) % self.cap;
            out_idx += n;
            remaining -= n;
        }

        out
    }
}

#[derive(Debug, Clone, Default)]
struct StabilityState {
    stable_surah_id: Option<u16>,
    streak_surah_id: Option<u16>,
    streak_len: usize,

    committed_end: Option<(u16, u16)>,
    committed_end_global: Option<u16>,
    committed_at_s: f64,

    pending_end: Option<(u16, u16)>,
    pending_end_global: Option<u16>,
    pending_streak_len: usize,

    low_conf_streak_len: usize,
}

#[derive(Debug, Deserialize)]
struct StreamJwtClaims {
    sub: Option<String>,
    sid: Option<String>,
    role: Option<String>,
    aud: Option<String>,
    exp: u64,
}

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub window_s: f64,
    pub hop_s: f64,
    pub buffer_s: f64,
    pub min_process_s: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            channels: 1,
            window_s: 20.0,
            hop_s: 2.0,
            buffer_s: 90.0,
            min_process_s: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
struct TimedWord {
    start_s: f64,
    end_s: f64,
    word: String,
}

#[derive(Debug, Default)]
struct RollingTranscript {
    words: Vec<TimedWord>,
    last_full_refresh_end_s: f64,
    force_refresh_next: bool,
}

pub struct StreamSession {
    pub id: String,
    cfg: StreamConfig,
    data_dir: PathBuf,
    ring: Mutex<RingBuf>,
    total_pcm_bytes: AtomicU64,
    last_processed_pcm_bytes: AtomicU64,
    stop_requested: AtomicBool,
    notify: Notify,
    pub out_tx: broadcast::Sender<String>,
    stability: Mutex<StabilityState>,

    http: reqwest::Client,
    api_key: String,
    transcriber_pool: TranscriberPool,
    transcribe_sem: Arc<Semaphore>,
    assets: Arc<AlignmentAssetsV2>,
    word_time_upgrade: String,
    stream_jwt_secret: String,
    stream_jwt_audience: String,
    auth_expires_unix_s: AtomicU64,
    auth_grace_s: u64,
    last_audio_unix_ms: AtomicU64,
    silence_since_unix_ms: AtomicU64,
    silence_dbfs_threshold: f64,
    silence_skip_after_s: f64,
    silence_stop_after_s: f64,
    no_audio_stop_after_s: f64,
    tail_context_s: f64,
    full_refresh_every_s: f64,
    transcript: Mutex<RollingTranscript>,
}

impl StreamSession {
    pub fn spawn(
        id: String,
        cfg: StreamConfig,
        data_dir: PathBuf,
        http: reqwest::Client,
        api_key: String,
        transcriber_pool: TranscriberPool,
        transcribe_sem: Arc<Semaphore>,
        assets: Arc<AlignmentAssetsV2>,
        word_time_upgrade: String,
        stream_jwt_secret: String,
        stream_jwt_audience: String,
        auth_grace_s: u64,
        no_audio_stop_after_s: f64,
        silence_dbfs_threshold: f64,
        silence_skip_after_s: f64,
        silence_stop_after_s: f64,
        tail_context_s: f64,
        full_refresh_every_s: f64,
    ) -> Arc<Self> {
        let (out_tx, _out_rx) = broadcast::channel::<String>(256);
        let bytes_per_s = (cfg.sample_rate_hz as usize)
            .saturating_mul(cfg.channels as usize)
            .saturating_mul(2);
        let ring_max_bytes = bytes_per_s.saturating_mul((cfg.buffer_s.max(cfg.window_s).ceil() as usize).max(1));

        let s = Arc::new(Self {
            id,
            cfg,
            data_dir,
            ring: Mutex::new(RingBuf::new(ring_max_bytes)),
            total_pcm_bytes: AtomicU64::new(0),
            last_processed_pcm_bytes: AtomicU64::new(0),
            stop_requested: AtomicBool::new(false),
            notify: Notify::new(),
            out_tx,
            stability: Mutex::new(StabilityState::default()),
            http,
            api_key,
            transcriber_pool,
            transcribe_sem,
            assets,
            word_time_upgrade,
            stream_jwt_secret,
            stream_jwt_audience,
            auth_expires_unix_s: AtomicU64::new(0),
            auth_grace_s,
            last_audio_unix_ms: AtomicU64::new(0),
            silence_since_unix_ms: AtomicU64::new(0),
            silence_dbfs_threshold,
            silence_skip_after_s,
            silence_stop_after_s,
            no_audio_stop_after_s,
            tail_context_s: tail_context_s.max(0.0),
            full_refresh_every_s: full_refresh_every_s.max(0.0),
            transcript: Mutex::new(RollingTranscript::default()),
        });

        let s2 = s.clone();
        tokio::spawn(async move {
            if let Err(e) = s2.processor_loop().await {
                error!(session_id = %s2.id, "stream processor failed: {e}");
            }
        });

        let s3 = s.clone();
        tokio::spawn(async move {
            s3.watchdog_loop().await;
        });

        s
    }

    pub fn ws_path(&self) -> String {
        format!("/v1/sessions/{}/stream", self.id)
    }

    pub fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::Relaxed);
        self.notify.notify_waiters();
    }

    pub fn set_auth_expiry_unix_s(&self, exp: u64) {
        self.auth_expires_unix_s.store(exp, Ordering::Relaxed);
    }

    pub async fn push_pcm16le(&self, bytes: &[u8]) -> anyhow::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if bytes.len() % 2 != 0 {
            anyhow::bail!("pcm frame must be multiple of 2 bytes");
        }
        if self.cfg.channels != 1 {
            anyhow::bail!("only mono supported for now");
        }

        let now_ms = now_unix_ms();
        self.last_audio_unix_ms.store(now_ms, Ordering::Relaxed);

        let dbfs = pcm16le_dbfs(bytes);
        if dbfs <= self.silence_dbfs_threshold {
            let prev = self.silence_since_unix_ms.load(Ordering::Relaxed);
            if prev == 0 {
                self.silence_since_unix_ms.store(now_ms, Ordering::Relaxed);
            }
        } else {
            self.silence_since_unix_ms.store(0, Ordering::Relaxed);
        }

        let mut ring = self.ring.lock().await;
        ring.push(bytes);
        self.total_pcm_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        drop(ring);
        self.notify.notify_waiters();
        Ok(())
    }

    pub async fn handle_ws(&self, socket: axum::extract::ws::WebSocket) {
        let (mut ws_tx, mut ws_rx) = socket.split();

        let mut out_rx = self.out_tx.subscribe();
        let forwarder = tokio::spawn(async move {
            let mut ping = interval(Duration::from_secs(20));
            ping.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = ping.tick() => {
                        if ws_tx.send(axum::extract::ws::Message::Ping(vec![])).await.is_err() {
                            break;
                        }
                    }
                    msg = out_rx.recv() => {
                        let Ok(msg) = msg else { break };
                        if ws_tx.send(axum::extract::ws::Message::Text(msg)).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });

        while let Some(Ok(msg)) = ws_rx.next().await {
            match msg {
                axum::extract::ws::Message::Binary(data) => {
                    let _ = self.push_pcm16le(&data).await;
                }
                axum::extract::ws::Message::Text(t) => {
                    // Optional control messages: {"type":"stop"} / {"type":"auth_refresh","token":"..."} etc.
                    if let Ok(v) = serde_json::from_str::<Value>(&t) {
                        let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
                        if ty == "stop" || ty == "end" {
                            self.request_stop();
                            break;
                        }
                        if ty == "auth_refresh" {
                            let token = v.get("token").and_then(|x| x.as_str()).unwrap_or("");
                            match self.validate_and_apply_stream_token(token) {
                                Ok(exp) => {
                                    let _ = self.out_tx.send(
                                        json!({
                                            "type":"auth_ok",
                                            "session_id": self.id,
                                            "expires_at_unix_s": exp,
                                        })
                                        .to_string(),
                                    );
                                }
                                Err(e) => {
                                    let _ = self.out_tx.send(
                                        json!({
                                            "type":"auth_error",
                                            "session_id": self.id,
                                            "error": e,
                                        })
                                        .to_string(),
                                    );
                                }
                            }
                        }
                    }
                }
                axum::extract::ws::Message::Close(_) => break,
                _ => {}
            }
        }

        forwarder.abort();
    }

    async fn processor_loop(&self) -> anyhow::Result<()> {
        let cfg = self.cfg.clone();
        let bytes_per_s = (cfg.sample_rate_hz as u64) * (cfg.channels as u64) * 2;
        let hop_bytes = (cfg.hop_s.max(0.1) * bytes_per_s as f64) as u64;
        let window_bytes = (cfg.window_s.max(0.5) * bytes_per_s as f64) as u64;
        let min_process_bytes = (cfg.min_process_s.max(0.1) * bytes_per_s as f64) as u64;
        let tail_s = (cfg.hop_s + self.tail_context_s)
            .max(cfg.hop_s)
            .min(cfg.window_s.max(cfg.hop_s));
        let tail_bytes = (tail_s.max(0.1) * bytes_per_s as f64) as u64;

        let mut seq: u64 = 0;
        loop {
            self.notify.notified().await;

            loop {
                let total = self.total_pcm_bytes.load(Ordering::Relaxed);
                let last = self.last_processed_pcm_bytes.load(Ordering::Relaxed);

                let should_stop = self.stop_requested.load(Ordering::Relaxed);
                let enough_to_process = total.saturating_sub(last) >= hop_bytes && total >= min_process_bytes;
                if !enough_to_process {
                    if should_stop {
                        // If stop requested and there's nothing left to process, exit.
                        return Ok(());
                    }
                    break;
                }

                let process_end_byte = last.saturating_add(hop_bytes).min(total);
                let process_end_s = (process_end_byte as f64) / (bytes_per_s as f64);

                // Optional: if we've been silent long enough, skip ASR entirely and just advance time.
                let silent_since = self.silence_since_unix_ms.load(Ordering::Relaxed);
                if silent_since > 0
                    && self.silence_skip_after_s.is_finite()
                    && self.silence_skip_after_s > 0.0
                {
                    let silent_s = (now_unix_ms().saturating_sub(silent_since)) as f64 / 1000.0;
                    if silent_s >= self.silence_skip_after_s {
                        // Prune transcript window as time advances (avoid unbounded growth).
                        let keep_start_s = (process_end_s - cfg.window_s).max(0.0);
                        {
                            let mut rt = self.transcript.lock().await;
                            rt.words.retain(|w| w.end_s >= keep_start_s);
                        }

                        let _ = self.out_tx.send(
                            json!({
                                "type":"silence",
                                "session_id": self.id,
                                "seq": seq,
                                "at_s": process_end_s,
                                "silent_s": silent_s,
                                "threshold_dbfs": self.silence_dbfs_threshold,
                            })
                            .to_string(),
                        );
                        seq = seq.saturating_add(1);
                        self.last_processed_pcm_bytes
                            .store(process_end_byte, Ordering::Relaxed);
                        continue;
                    }
                }

                let (mode, pcm, transcribe_start_s, transcribe_end_s) = self
                    .extract_transcribe_pcm(process_end_byte, window_bytes, tail_bytes, bytes_per_s)
                    .await?;

                let t0 = std::time::Instant::now();
                let (tx, transcribe_s) = match self.transcribe_window(&pcm).await {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.out_tx.send(
                            json!({
                                "type": "error",
                                "stage": "transcribe",
                                "seq": seq,
                                "error": e.to_string(),
                            })
                            .to_string(),
                        );
                        error!(session_id = %self.id, "transcribe failed: {e}");
                        // Skip ahead to avoid being stuck.
                        self.last_processed_pcm_bytes
                            .store(process_end_byte, Ordering::Relaxed);
                        break;
                    }
                };

                let text = tx
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let mut transcription = tx
                    .get("transcription")
                    .cloned()
                    .unwrap_or_else(|| json!({}));

                // Offset timestamps into the session timeline (slice-relative -> session-absolute).
                offset_transcription_times(&mut transcription, transcribe_start_s);
                let dedupe_meta = transcription.get("dedupe").cloned().unwrap_or(Value::Null);
                let repair_meta = transcription.get("repair").cloned().unwrap_or(Value::Null);

                // Update rolling transcript window (word-level) for alignment.
                let new_words = extract_timed_words(&transcription);
                let align_start_s = (process_end_s - cfg.window_s).max(0.0);
                let (align_transcription, align_text, rolling_words) = {
                    let mut rt = self.transcript.lock().await;

                    if mode == "refresh" {
                        rt.words = new_words;
                        rt.last_full_refresh_end_s = process_end_s;
                    } else {
                        ingest_words_into_rolling(
                            &mut rt.words,
                            new_words,
                            transcribe_start_s,
                            transcribe_end_s,
                            align_start_s,
                        );
                    }

                    let words_for_align: Vec<TimedWord> = rt
                        .words
                        .iter()
                        .filter(|w| w.end_s >= align_start_s && w.start_s <= process_end_s)
                        .cloned()
                        .collect();
                    let (t, text) = build_transcription_from_words(&words_for_align);
                    (t, text, words_for_align.len())
                };

                let t1 = std::time::Instant::now();
                let ms = tokio::task::spawn_blocking({
                    let transcription = align_transcription;
                    let text = align_text.clone();
                    let assets = self.assets.clone();
                    let word_time_upgrade = self.word_time_upgrade.clone();
                    move || {
                        let ms =
                            multispan::align_multi_span(&transcription, &text, assets.as_ref(), &word_time_upgrade);
                        ms
                    }
                })
                .await
                .map_err(|e| anyhow::anyhow!("align join error: {e}"))?;
                let align_s = t1.elapsed().as_secs_f64();

                let conf = ms
                    .active_alignment
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let pred_surah_id: Option<u16> = ms
                    .active_alignment
                    .get("start")
                    .and_then(|v| v.get("surah_id"))
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u16::try_from(v).ok());
                let pred_end: Option<(u16, u16)> = ms
                    .active_alignment
                    .get("end")
                    .and_then(|v| v.as_object())
                    .and_then(|o| {
                        let sid = o.get("surah_id")?.as_u64()? as u16;
                        let anum = o.get("ayah_num")?.as_u64()? as u16;
                        Some((sid, anum))
                    });
                let pred_end_global: Option<u16> = pred_end
                    .and_then(|(sid, anum)| self.assets.ayah_key_to_global.get(&(sid, anum)).copied());
                let matches = ms
                    .active_alignment
                    .get("alignment")
                    .and_then(|v| v.get("matches"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let match_ratio = ms
                    .active_alignment
                    .get("alignment")
                    .and_then(|v| v.get("match_ratio"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                // Streaming stability: avoid flickering to wrong surahs early on by requiring
                // a small streak + minimum confidence before "locking" the surah.
                let min_conf = 0.10_f64;
                let switch_conf = 0.25_f64;
                let subs = ms
                    .active_alignment
                    .get("alignment")
                    .and_then(|v| v.get("substitutions"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                let (stable_surah_id, streak_surah_id, streak_len, progress, want_low_conf_refresh) = {
                    let mut st = self.stability.lock().await;
                    if st.streak_surah_id == pred_surah_id {
                        st.streak_len = st.streak_len.saturating_add(1);
                    } else {
                        st.streak_surah_id = pred_surah_id;
                        st.streak_len = 1;
                    }

                    if st.stable_surah_id.is_none() {
                        if pred_surah_id.is_some() && conf >= min_conf && st.streak_len >= 2 {
                            st.stable_surah_id = pred_surah_id;
                        }
                    } else if st.stable_surah_id != pred_surah_id {
                        if pred_surah_id.is_some() && conf >= switch_conf && st.streak_len >= 2 {
                            st.stable_surah_id = pred_surah_id;
                        }
                    }

                    let mut action = "hold".to_string();
                    let mut skipped: Option<u16> = None;

                    // Commit logic: don't "advance" to a new ayah unless we see it stably.
                    // This prevents mislabeling when a user skips (e.g., ayah 1 -> ayah 3) and the
                    // model briefly guesses ayah 2.
                    // Note: `matches` alone can stay small even on decent alignments; `substitutions`
                    // is a better proxy for "we had enough transcript tokens to be confident".
                    // Thresholds tuned for streaming: allow commits on short surahs (few tokens),
                    // but stay conservative on resets/backward jumps.
                    let strong = conf >= 0.65 && subs >= 10 && match_ratio >= 0.55;
                    let ok = conf >= 0.32 && subs >= 6 && match_ratio >= 0.40;
                    let jump_ok = conf >= 0.45 && subs >= 6 && match_ratio >= 0.55;
                    let commit_streak = 2usize;

                    match (pred_end, pred_end_global) {
                        (Some(end_key), Some(end_g)) => {
                            let prev = st.committed_end_global;

                            let mut should_commit = false;
                            let mut is_reset = false;
                            let mut is_big_jump = false;

                            if prev.is_none() {
                                if strong {
                                    should_commit = true;
                                } else {
                                    match (st.pending_end_global, st.pending_end) {
                                        (Some(pg), Some(pk)) if pg == end_g && pk == end_key => {
                                            st.pending_streak_len = st.pending_streak_len.saturating_add(1);
                                        }
                                        _ => {
                                            st.pending_end = Some(end_key);
                                            st.pending_end_global = Some(end_g);
                                            st.pending_streak_len = 1;
                                        }
                                    }
                                    if ok && st.pending_streak_len >= commit_streak {
                                        should_commit = true;
                                    }
                                }
                            } else if let Some(prev_g) = prev {
                                if end_g + 2 < prev_g {
                                    // Jumping backwards (likely restart or a wrong guess). Only accept if extremely strong.
                                    if strong {
                                        should_commit = true;
                                        is_reset = true;
                                    } else {
                                        action = "ignore_backward".to_string();
                                        st.pending_end = None;
                                        st.pending_end_global = None;
                                        st.pending_streak_len = 0;
                                    }
                                } else if end_g <= prev_g {
                                    // Not advancing (same ayah or jitter). Clear pending and hold.
                                    st.pending_end = None;
                                    st.pending_end_global = None;
                                    st.pending_streak_len = 0;
                                } else {
                                    // Forward progress.
                                    match (st.pending_end_global, st.pending_end) {
                                        (Some(pg), Some(pk)) if pg == end_g && pk == end_key => {
                                            st.pending_streak_len = st.pending_streak_len.saturating_add(1);
                                        }
                                        _ => {
                                            st.pending_end = Some(end_key);
                                            st.pending_end_global = Some(end_g);
                                            st.pending_streak_len = 1;
                                        }
                                    }

                                    let delta = end_g.saturating_sub(prev_g) as u32;
                                    is_big_jump = delta >= 8;
                                    if strong {
                                        should_commit = true;
                                    } else if ok && st.pending_streak_len >= commit_streak {
                                        if is_big_jump {
                                            should_commit = jump_ok;
                                        } else {
                                            should_commit = true;
                                        }
                                    }

                                    if should_commit && end_g > prev_g + 1 {
                                        skipped = Some(end_g.saturating_sub(prev_g).saturating_sub(1));
                                    }
                                }
                            }

                            if should_commit {
                                st.committed_end = Some(end_key);
                                st.committed_end_global = Some(end_g);
                                st.committed_at_s = process_end_s;
                                st.pending_end = None;
                                st.pending_end_global = None;
                                st.pending_streak_len = 0;
                                action = if is_reset {
                                    "reset_committed".to_string()
                                } else if skipped.is_some() {
                                    "commit_skip".to_string()
                                } else {
                                    "commit".to_string()
                                };

                                // After a reset/big-jump commit, re-lock the stable surah immediately to the new prediction.
                                if (is_reset || is_big_jump) && pred_surah_id.is_some() {
                                    st.stable_surah_id = pred_surah_id;
                                    st.streak_surah_id = pred_surah_id;
                                    st.streak_len = st.streak_len.max(commit_streak);
                                }
                            }
                        }
                        _ => {
                            st.pending_end = None;
                            st.pending_end_global = None;
                            st.pending_streak_len = 0;
                        }
                    }

                    // If we're repeatedly low-confidence (often drift/duplication), request a one-off full refresh.
                    // We do the actual refresh decision outside the lock (needs transcript state + window age).
                    if conf < 0.22 {
                        st.low_conf_streak_len = st.low_conf_streak_len.saturating_add(1);
                    } else {
                        st.low_conf_streak_len = 0;
                    }
                    let want_low_conf_refresh = st.low_conf_streak_len >= 3;
                    if want_low_conf_refresh {
                        st.low_conf_streak_len = 0;
                    }

                    let committed = st.committed_end.map(|(sid, anum)| {
                        json!({
                            "surah_id": sid,
                            "ayah_num": anum,
                            "global": st.committed_end_global,
                            "at_s": st.committed_at_s,
                        })
                    });
                    let pending = st.pending_end.map(|(sid, anum)| {
                        json!({
                            "surah_id": sid,
                            "ayah_num": anum,
                            "global": st.pending_end_global,
                            "streak_len": st.pending_streak_len,
                        })
                    });

                    let progress = json!({
                        "action": action,
                        "committed_end": committed,
                        "pending_end": pending,
                        "hypothesis_end": pred_end.map(|(sid, anum)| json!({"surah_id": sid, "ayah_num": anum})),
                        "hypothesis_end_global": pred_end_global,
                        "confidence": conf,
                        "matches": matches,
                        "match_ratio": match_ratio,
                        "skipped_ayahs": skipped,
                    });

                    (
                        st.stable_surah_id,
                        st.streak_surah_id,
                        st.streak_len,
                        progress,
                        want_low_conf_refresh,
                    )
                };
                let stable = stable_surah_id.is_some();

                // Post-commit recovery: if we committed a big jump, clear the rolling transcript so the next
                // update isn't polluted by pre-jump text, and force a full refresh next hop.
                let action = progress.get("action").and_then(|v| v.as_str()).unwrap_or("");
                let skipped_n = progress.get("skipped_ayahs").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
                let big_jump_prune = action == "reset_committed" || (action == "commit_skip" && skipped_n >= 20);
                if big_jump_prune {
                    let mut rt = self.transcript.lock().await;
                    rt.words.clear();
                    rt.last_full_refresh_end_s = 0.0;
                    rt.force_refresh_next = true;
                } else if want_low_conf_refresh && mode != "refresh" && process_end_s >= cfg.window_s {
                    let mut rt = self.transcript.lock().await;
                    let since_refresh = process_end_s - rt.last_full_refresh_end_s;
                    if since_refresh >= 8.0 {
                        rt.force_refresh_next = true;
                    }
                }

                // Client UX helpers: small, stable signals that don't require parsing the full alignment blob.
                let ux_state = if stable && conf >= 0.32 {
                    "tracking"
                } else if stable {
                    "uncertain"
                } else {
                    "seeking"
                };
                let mut alerts: Vec<Value> = Vec::new();
                if action == "commit_skip" && skipped_n > 0 {
                    let code = if skipped_n >= 20 { "surah_jump" } else { "ayah_skip" };
                    alerts.push(json!({
                        "code": code,
                        "skipped_ayahs": skipped_n,
                    }));
                }
                if action == "reset_committed" {
                    alerts.push(json!({ "code": "reset_detected" }));
                }
                if action == "ignore_backward" {
                    alerts.push(json!({ "code": "ignored_backward_guess" }));
                }
                if big_jump_prune {
                    alerts.push(json!({ "code": "jump_pruned", "skipped_ayahs": skipped_n }));
                }
                if want_low_conf_refresh && mode != "refresh" && process_end_s >= cfg.window_s {
                    alerts.push(json!({ "code": "refresh_scheduled", "reason": "low_confidence" }));
                }
                if mode == "refresh" {
                    alerts.push(json!({ "code": "full_refresh" }));
                }
                if rolling_words < 8 && conf < 0.25 && process_end_s >= 4.0 {
                    alerts.push(json!({ "code": "need_more_audio" }));
                }

                let tail_text = if text.len() > 400 {
                    text.chars().skip(text.len().saturating_sub(400)).collect::<String>()
                } else {
                    text.clone()
                };

                let evt = json!({
                    "type": "ayah_update",
                    "session_id": self.id,
                    "seq": seq,
                    "at_s": process_end_s,
                    "window": {
                        "start_s": align_start_s,
                        "end_s": process_end_s,
                    },
                    "transcribe_window": {
                        "mode": mode,
                        "start_s": transcribe_start_s,
                        "end_s": transcribe_end_s,
                    },
                    "stability": {
                        "stable": stable,
                        "stable_surah_id": stable_surah_id,
                        "predicted_surah_id": pred_surah_id,
                        "predicted_streak_surah_id": streak_surah_id,
                        "predicted_streak_len": streak_len,
                        "confidence": conf,
                        "min_conf": min_conf,
                        "switch_conf": switch_conf,
                    },
                    "progress": progress,
                    "ux": {
                        "state": ux_state,
                        "alerts": alerts,
                    },
                    "transcription": {
                        "text_tail": tail_text,
                        "rolling_words": rolling_words,
                        "dedupe": dedupe_meta,
                        "repair": repair_meta,
                    },
                    "ayah_alignment": ms.active_alignment,
                    "ayah_alignment_spans": ms.spans,
                    "ayah_alignment_jumps": ms.jumps,
                    "ayah_alignment_active_span_index": ms.active_span_index.map(|v| Value::from(v as i64)).unwrap_or(Value::Null),
                    "timing": {
                        "transcribe_s": transcribe_s,
                        "align_s": align_s,
                        "loop_s": t0.elapsed().as_secs_f64(),
                    }
                });
                let _ = self.out_tx.send(evt.to_string());

                seq = seq.saturating_add(1);
                self.last_processed_pcm_bytes
                    .store(process_end_byte, Ordering::Relaxed);

                if should_stop && (total.saturating_sub(self.last_processed_pcm_bytes.load(Ordering::Relaxed)) < hop_bytes) {
                    return Ok(());
                }
            }
        }
    }

    async fn extract_transcribe_pcm(
        &self,
        end_pcm_bytes: u64,
        window_bytes: u64,
        tail_bytes: u64,
        bytes_per_s: u64,
    ) -> anyhow::Result<(String, Vec<u8>, f64, f64)> {
        let end_pcm_bytes = end_pcm_bytes.min(self.total_pcm_bytes.load(Ordering::Relaxed));
        let end_s = (end_pcm_bytes as f64) / (bytes_per_s as f64);

        let (mode, want_bytes) = {
            let mut rt = self.transcript.lock().await;
            if rt.force_refresh_next {
                rt.force_refresh_next = false;
                ("refresh".to_string(), window_bytes)
            } else if self.full_refresh_every_s.is_finite()
                && self.full_refresh_every_s > 0.0
                && (end_s - rt.last_full_refresh_end_s) >= self.full_refresh_every_s
            {
                ("refresh".to_string(), window_bytes)
            } else {
                ("tail".to_string(), tail_bytes)
            }
        };

        let (pcm, start_s, end_s2) = self.extract_pcm_range(end_pcm_bytes, want_bytes, bytes_per_s).await?;
        Ok((mode, pcm, start_s, end_s2))
    }

    async fn extract_pcm_range(
        &self,
        end_pcm_bytes: u64,
        window_bytes: u64,
        bytes_per_s: u64,
    ) -> anyhow::Result<(Vec<u8>, f64, f64)> {
        let total_pcm_bytes = self.total_pcm_bytes.load(Ordering::Relaxed);
        let end_pcm_bytes = end_pcm_bytes.min(total_pcm_bytes);

        let ring = self.ring.lock().await;
        let ring_len = ring.len() as u64;
        let ring_start_byte = total_pcm_bytes.saturating_sub(ring_len);

        let end_byte = end_pcm_bytes.max(ring_start_byte);
        let want_start_byte = end_byte.saturating_sub(window_bytes);
        let start_byte = want_start_byte.max(ring_start_byte);

        let start_idx = (start_byte.saturating_sub(ring_start_byte)) as usize;
        let end_idx = (end_byte.saturating_sub(ring_start_byte)) as usize;
        let mut out = ring.copy_range(start_idx, end_idx.saturating_sub(start_idx));
        drop(ring);

        // Keep sample alignment.
        if out.len() % 2 != 0 {
            out.pop();
        }

        let start_s = (start_byte as f64) / (bytes_per_s as f64);
        let end_s = (end_byte as f64) / (bytes_per_s as f64);
        Ok((out, start_s, end_s))
    }

    async fn transcribe_window(&self, pcm_bytes: &[u8]) -> anyhow::Result<(Value, f64)> {
        let _permit = self
            .transcribe_sem
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("transcribe semaphore closed"))?;
        let lease = self.transcriber_pool.pick();
        let url = format!("{}/v1/transcribe", lease.url().trim_end_matches('/'));
        let mut req = self.http.post(url);
        if !self.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.api_key));
        }

        // Avoid disk I/O in the hot streaming loop: build a tiny WAV header + PCM bytes in memory.
        let wav_bytes = wav_bytes_pcm16le(pcm_bytes, self.cfg.sample_rate_hz, self.cfg.channels);
        let part = Part::bytes(wav_bytes)
            .file_name("window.wav")
            .mime_str("audio/wav")?;
        let form = Form::new().part("file", part);

        let t0 = std::time::Instant::now();
        let resp = req.multipart(form).send().await?;
        let status = resp.status();
        let obj: Value = resp.json().await.unwrap_or_else(|_| json!({}));
        let t_transcribe = t0.elapsed().as_secs_f64();

        if !status.is_success() {
            anyhow::bail!("transcriber error ({}) {}", status.as_u16(), obj);
        }
        Ok((obj, t_transcribe))
    }

    async fn watchdog_loop(&self) {
        let tick = std::time::Duration::from_millis(250);
        loop {
            tokio::time::sleep(tick).await;
            if self.stop_requested.load(Ordering::Relaxed) {
                return;
            }

            let now_s = now_unix_s();
            let exp = self.auth_expires_unix_s.load(Ordering::Relaxed);
            if exp > 0 && now_s.saturating_add(self.auth_grace_s) > exp {
                let _ = self.out_tx.send(
                    json!({
                        "type":"session_end",
                        "reason":"auth_expired",
                        "session_id": self.id,
                    })
                    .to_string(),
                );
                self.request_stop();
                return;
            }

            let last_audio_ms = self.last_audio_unix_ms.load(Ordering::Relaxed);
            if last_audio_ms > 0 && self.no_audio_stop_after_s.is_finite() && self.no_audio_stop_after_s > 0.0 {
                let since_s = (now_unix_ms().saturating_sub(last_audio_ms)) as f64 / 1000.0;
                if since_s >= self.no_audio_stop_after_s {
                    let _ = self.out_tx.send(
                        json!({
                            "type":"session_end",
                            "reason":"no_audio_timeout",
                            "session_id": self.id,
                            "since_s": since_s,
                        })
                        .to_string(),
                    );
                    self.request_stop();
                    return;
                }
            }

            let silent_since = self.silence_since_unix_ms.load(Ordering::Relaxed);
            if silent_since > 0 && self.silence_stop_after_s.is_finite() && self.silence_stop_after_s > 0.0 {
                let silent_s = (now_unix_ms().saturating_sub(silent_since)) as f64 / 1000.0;
                if silent_s >= self.silence_stop_after_s {
                    let _ = self.out_tx.send(
                        json!({
                            "type":"session_end",
                            "reason":"silence_timeout",
                            "session_id": self.id,
                            "silent_s": silent_s,
                            "threshold_dbfs": self.silence_dbfs_threshold,
                        })
                        .to_string(),
                    );
                    self.request_stop();
                    return;
                }
            }
        }
    }

    fn validate_and_apply_stream_token(&self, token: &str) -> Result<u64, String> {
        if token.trim().is_empty() {
            return Err("missing token".to_string());
        }
        if self.stream_jwt_secret.trim().is_empty() {
            return Err("server not configured for stream jwt".to_string());
        }

        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_audience(&[self.stream_jwt_audience.clone()]);
        validation.validate_exp = true;

        let data = decode::<StreamJwtClaims>(
            token,
            &DecodingKey::from_secret(self.stream_jwt_secret.as_bytes()),
            &validation,
        )
        .map_err(|e| format!("invalid token: {e}"))?;

        if data.claims.sid.as_deref() != Some(self.id.as_str()) {
            return Err("sid mismatch".to_string());
        }
        if data.claims.role.as_deref() != Some("admin") {
            return Err("admin role required".to_string());
        }

        self.set_auth_expiry_unix_s(data.claims.exp);
        Ok(data.claims.exp)
    }
}

async fn write_wav_pcm16le(
    path: &Path,
    pcm_bytes: &[u8],
    sample_rate_hz: u32,
    channels: u16,
) -> anyhow::Result<()> {
    let bytes = wav_bytes_pcm16le(pcm_bytes, sample_rate_hz, channels);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    tokio::fs::write(path, bytes)
        .await
        .with_context(|| format!("write wav file {}", path.display()))?;
    Ok(())
}

fn wav_bytes_pcm16le(pcm_bytes: &[u8], sample_rate_hz: u32, channels: u16) -> Vec<u8> {
    let data_size = pcm_bytes.len() as u32;
    let byte_rate = sample_rate_hz
        .saturating_mul(channels as u32)
        .saturating_mul(2);
    let block_align = channels.saturating_mul(2);

    let mut out: Vec<u8> = Vec::with_capacity(44usize.saturating_add(pcm_bytes.len()));
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32.saturating_add(data_size)).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&(16u32).to_le_bytes()); // PCM fmt chunk
    out.extend_from_slice(&(1u16).to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&(16u16).to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    out.extend_from_slice(pcm_bytes);
    out
}

pub fn validate_stream_cfg(mut cfg: StreamConfig) -> anyhow::Result<StreamConfig> {
    if cfg.sample_rate_hz != 16_000 {
        anyhow::bail!("only 16kHz supported for now");
    }
    if cfg.channels != 1 {
        anyhow::bail!("only mono supported for now");
    }
    if !(cfg.window_s.is_finite() && cfg.window_s > 0.5) {
        cfg.window_s = 20.0;
    }
    if !(cfg.hop_s.is_finite() && cfg.hop_s > 0.1) {
        cfg.hop_s = 2.0;
    }
    if cfg.hop_s >= cfg.window_s {
        cfg.hop_s = (cfg.window_s / 2.0).max(0.5);
    }
    if !(cfg.buffer_s.is_finite() && cfg.buffer_s >= cfg.window_s) {
        cfg.buffer_s = (cfg.window_s * 2.0).max(30.0);
    }
    if !(cfg.min_process_s.is_finite() && cfg.min_process_s > 0.0) {
        cfg.min_process_s = 2.0;
    }
    Ok(cfg)
}

pub fn ws_url_from_http_base(http_base: &str, ws_path: &str) -> String {
    let mut base = http_base.trim_end_matches('/').to_string();
    if base.starts_with("https://") {
        base = base.replacen("https://", "wss://", 1);
    } else if base.starts_with("http://") {
        base = base.replacen("http://", "ws://", 1);
    }
    format!("{}{}", base, ws_path)
}

fn now_unix_s() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn pcm16le_dbfs(bytes: &[u8]) -> f64 {
    if bytes.len() < 2 {
        return -100.0;
    }
    let mut sum_sq: f64 = 0.0;
    let mut n: usize = 0;
    for chunk in bytes.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f64;
        sum_sq += sample * sample;
        n += 1;
    }
    if n == 0 {
        return -100.0;
    }
    let mean_sq = sum_sq / (n as f64);
    let rms = mean_sq.sqrt();
    if rms <= 0.0 {
        return -100.0;
    }
    let norm = rms / 32768.0;
    20.0 * norm.log10()
}

fn offset_transcription_times(transcription: &mut Value, offset_s: f64) {
    if offset_s == 0.0 {
        return;
    }
    let Some(segs) = transcription.get_mut("segments").and_then(|v| v.as_array_mut()) else {
        return;
    };

    for seg in segs.iter_mut() {
        let Some(obj) = seg.as_object_mut() else { continue };
        if let Some(s) = obj.get("start_s").and_then(|v| v.as_f64()) {
            obj.insert("start_s".to_string(), Value::from(s + offset_s));
        }
        if let Some(e) = obj.get("end_s").and_then(|v| v.as_f64()) {
            obj.insert("end_s".to_string(), Value::from(e + offset_s));
        }
        if let Some(words) = obj.get_mut("words").and_then(|v| v.as_array_mut()) {
            for w in words.iter_mut() {
                let Some(wobj) = w.as_object_mut() else { continue };
                if let Some(s) = wobj.get("start_s").and_then(|v| v.as_f64()) {
                    wobj.insert("start_s".to_string(), Value::from(s + offset_s));
                }
                if let Some(e) = wobj.get("end_s").and_then(|v| v.as_f64()) {
                    wobj.insert("end_s".to_string(), Value::from(e + offset_s));
                }
            }
        }
    }
}

fn extract_timed_words(transcription: &Value) -> Vec<TimedWord> {
    let Some(segs) = transcription.get("segments").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    let mut out: Vec<TimedWord> = Vec::new();
    for seg in segs {
        let Some(words) = seg.get("words").and_then(|v| v.as_array()) else {
            continue;
        };
        for w in words {
            let Some(word) = w.get("word").and_then(|v| v.as_str()) else {
                continue;
            };
            let word = word.trim();
            if word.is_empty() {
                continue;
            }
            let Some(start_s) = w.get("start_s").and_then(|v| v.as_f64()) else {
                continue;
            };
            let Some(end_s) = w.get("end_s").and_then(|v| v.as_f64()) else {
                continue;
            };
            out.push(TimedWord {
                start_s,
                end_s,
                word: word.to_string(),
            });
        }
    }
    out.sort_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap_or(CmpOrdering::Equal));
    out
}

fn ingest_words_into_rolling(
    rolling: &mut Vec<TimedWord>,
    new_words: Vec<TimedWord>,
    replace_start_s: f64,
    replace_end_s: f64,
    keep_start_s: f64,
) {
    let overlap_s = 0.25_f64;
    let remove_start = (replace_start_s - overlap_s).max(0.0);
    let remove_end = replace_end_s + overlap_s;

    rolling.retain(|w| w.end_s >= keep_start_s);
    rolling.retain(|w| !(w.end_s >= remove_start && w.start_s <= remove_end));
    rolling.extend(new_words);

    rolling.sort_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap_or(CmpOrdering::Equal));

    // Deduplicate near-identical repeats from overlapping slices.
    let mut deduped: Vec<TimedWord> = Vec::with_capacity(rolling.len());
    for w in rolling.drain(..) {
        if let Some(last) = deduped.last_mut() {
            if (w.start_s - last.start_s).abs() < 0.03 && w.word == last.word {
                if w.end_s > last.end_s {
                    last.end_s = w.end_s;
                }
                continue;
            }
        }
        deduped.push(w);
    }
    *rolling = deduped;
}

fn build_transcription_from_words(words: &[TimedWord]) -> (Value, String) {
    let seg_gap_s = 0.85_f64;
    let seg_max_words = 18usize;

    let mut segments: Vec<Value> = Vec::new();
    let mut full_text_parts: Vec<String> = Vec::new();

    let mut cur_words: Vec<&TimedWord> = Vec::new();
    let mut cur_start_s = 0.0_f64;
    let mut cur_end_s = 0.0_f64;

    let mut flush = |segments: &mut Vec<Value>,
                     full_text_parts: &mut Vec<String>,
                     cur_words: &mut Vec<&TimedWord>,
                     cur_start_s: &mut f64,
                     cur_end_s: &mut f64| {
        if cur_words.is_empty() {
            return;
        }
        let mut text_parts: Vec<String> = Vec::with_capacity(cur_words.len());
        let mut words_json: Vec<Value> = Vec::with_capacity(cur_words.len());
        let mut start_s = *cur_start_s;
        let mut end_s = *cur_end_s;
        for w in cur_words.iter() {
            start_s = start_s.min(w.start_s);
            end_s = end_s.max(w.end_s);
            text_parts.push(w.word.clone());
            words_json.push(json!({
                "word": w.word,
                "start_s": w.start_s,
                "end_s": w.end_s,
            }));
        }
        let text = text_parts.join(" ").trim().to_string();
        if !text.is_empty() {
            full_text_parts.push(text.clone());
        }
        segments.push(json!({
            "start_s": start_s,
            "end_s": end_s,
            "text": text,
            "words": words_json,
        }));
        cur_words.clear();
        *cur_start_s = 0.0;
        *cur_end_s = 0.0;
    };

    for w in words {
        if cur_words.is_empty() {
            cur_start_s = w.start_s;
            cur_end_s = w.end_s;
            cur_words.push(w);
            continue;
        }

        let gap_s = w.start_s - cur_end_s;
        if gap_s > seg_gap_s || cur_words.len() >= seg_max_words {
            flush(
                &mut segments,
                &mut full_text_parts,
                &mut cur_words,
                &mut cur_start_s,
                &mut cur_end_s,
            );
            cur_start_s = w.start_s;
            cur_end_s = w.end_s;
            cur_words.push(w);
            continue;
        }

        cur_end_s = cur_end_s.max(w.end_s);
        cur_words.push(w);
    }

    flush(
        &mut segments,
        &mut full_text_parts,
        &mut cur_words,
        &mut cur_start_s,
        &mut cur_end_s,
    );

    let text = full_text_parts.join(" ").trim().to_string();
    (json!({ "segments": segments }), text)
}

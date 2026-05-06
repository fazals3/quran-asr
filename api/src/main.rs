mod alignment_v2;
mod cleanup;
mod config;
mod demo;
mod guessing;
mod multispan;
mod npy;
mod state;
mod streaming;
mod text;
mod transcriber_pool;

use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::to_bytes;
use axum::extract::{FromRequest, Multipart, Path as AxumPath, Query, State};
use axum::extract::Request;
use axum::extract::DefaultBodyLimit;
use axum::extract::ws::WebSocketUpgrade;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Html;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use dashmap::DashMap;
use futures_util::StreamExt;
use jsonwebtoken::{DecodingKey, Validation};
use reqwest::multipart::{Form, Part};
use rusqlite::{params, Connection, OpenFlags};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{error, info};
use uuid::Uuid;

use crate::config::{env_f64_opt, load_settings, Settings};
use crate::state::{
    job_input_path, job_request_path, job_result_path, now_ts, AppState, JobMeta,
};
use crate::transcriber_pool::TranscriberPool;

#[derive(Debug, Deserialize)]
struct StreamJwtClaims {
    sub: Option<String>,
    sid: Option<String>,
    role: Option<String>,
    aud: Option<String>,
    exp: u64,
}

pub async fn write_json(path: &Path, value: &Value) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    tokio::fs::write(path, bytes).await?;
    Ok(())
}

pub async fn read_json(path: &Path) -> anyhow::Result<Value> {
    let bytes = tokio::fs::read(path).await?;
    let v: Value = serde_json::from_slice(&bytes)?;
    Ok(v)
}

fn require_internal_access(headers: &HeaderMap, remote_ip: Option<std::net::IpAddr>, settings: &Settings) -> Result<(), StatusCode> {
    if !settings.api_key.is_empty() {
        let auth = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth.strip_prefix("Bearer ").or_else(|| auth.strip_prefix("bearer "));
        if token.is_none() || token.unwrap().trim() != settings.api_key {
            return Err(StatusCode::UNAUTHORIZED);
        }

        if !settings.enforce_allow_cidrs_with_api_key {
            return Ok(());
        }
    }

    let ip = remote_ip.ok_or(StatusCode::FORBIDDEN)?;
    if settings.allow_cidrs.is_empty() {
        return Ok(());
    }
    for net in &settings.allow_cidrs {
        if net.contains(&ip) {
            return Ok(());
        }
    }
    Err(StatusCode::FORBIDDEN)
}

fn require_allow_cidrs(remote_ip: Option<std::net::IpAddr>, settings: &Settings) -> Result<(), StatusCode> {
    let ip = remote_ip.ok_or(StatusCode::FORBIDDEN)?;
    if settings.allow_cidrs.is_empty() {
        return Ok(());
    }
    for net in &settings.allow_cidrs {
        if net.contains(&ip) {
            return Ok(());
        }
    }
    Err(StatusCode::FORBIDDEN)
}

fn require_internal_streaming_test(
    key: Option<&str>,
    remote_ip: Option<std::net::IpAddr>,
    settings: &Settings,
) -> Result<(), StatusCode> {
    if !settings.internal_streaming_test_enabled {
        return Err(StatusCode::NOT_FOUND);
    }
    require_allow_cidrs(remote_ip, settings)?;
    let Some(got) = key.map(|s| s.trim()).filter(|s| !s.is_empty()) else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    if settings.internal_streaming_test_key.trim().is_empty() || got != settings.internal_streaming_test_key {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(())
}

pub fn json_resp(code: StatusCode, v: Value) -> Response {
    (code, Json(v)).into_response()
}

fn content_type_is_json(headers: &HeaderMap) -> bool {
    headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_lowercase().starts_with("application/json"))
        .unwrap_or(false)
}

fn host_allowed(host: &str, allow: &[String]) -> bool {
    if allow.is_empty() {
        return true;
    }
    let h = host.trim().to_lowercase();
    if h.is_empty() {
        return false;
    }
    for rule in allow {
        let r = rule.trim().to_lowercase();
        if r.is_empty() {
            continue;
        }
        if r == h {
            return true;
        }
        if let Some(suffix) = r.strip_prefix("*.") {
            if h == suffix || h.ends_with(&format!(".{suffix}")) {
                return true;
            }
        }
    }
    false
}

async fn health(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
) -> Response {
    if let Err(code) = require_internal_access(&headers, Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }

    let qlen = state
        .settings
        .max_queue_length
        .saturating_sub(state.queue_tx.capacity());
    json_resp(
        StatusCode::OK,
        json!({
            "ok": true,
            "time": now_ts(),
            "queue_length_estimate": qlen,
            "jobs_total": state.jobs.len(),
        }),
    )
}

#[derive(Debug, Deserialize)]
struct TranscribeQuery {
    wait: Option<bool>,
    wait_timeout_s: Option<i64>,

    // Guess params (accepted for compatibility; not used yet).
    top_k: Option<i64>,
    per_surah_top_n: Option<i64>,
    query_operator: Option<String>,
    rank_mode: Option<String>,
    max_query_tokens: Option<i64>,
    max_hits: Option<i64>,

    // Transcriber overrides.
    language: Option<String>,
    beam_size: Option<i64>,
    patience: Option<f64>,
    word_timestamps: Option<bool>,
    log_prob_threshold: Option<f64>,
    no_speech_threshold: Option<f64>,
    batch_size: Option<i64>,
}

#[derive(Debug, Deserialize, Default)]
struct CreateStreamSessionBody {
    sample_rate_hz: Option<u32>,
    channels: Option<u16>,
    window_s: Option<f64>,
    hop_s: Option<f64>,
    buffer_s: Option<f64>,
    min_process_s: Option<f64>,
}

pub async fn safe_stream_copy_to_path(
    mut field: axum::extract::multipart::Field<'_>,
    dst_path: &Path,
    max_bytes: usize,
) -> anyhow::Result<(u64, String, String)> {
    let filename = field
        .file_name()
        .map(|s| s.to_string())
        .unwrap_or_else(|| "upload.m4a".to_string());
    let content_type = field
        .content_type()
        .map(|s| s.to_string())
        .unwrap_or_else(|| "application/octet-stream".to_string());

    if let Some(parent) = dst_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut f = tokio::fs::File::create(dst_path).await?;
    let mut total: usize = 0;
    while let Some(chunk) = field.chunk().await? {
        total += chunk.len();
        if total > max_bytes {
            anyhow::bail!("upload too large ({} bytes > {})", total, max_bytes);
        }
        tokio::io::AsyncWriteExt::write_all(&mut f, &chunk).await?;
    }
    tokio::io::AsyncWriteExt::flush(&mut f).await?;
    Ok((total as u64, filename, content_type))
}

async fn create_stream_session(
    State(state): State<AppState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    req: Request,
) -> Response {
    let (parts, body) = req.into_parts();
    let headers = parts.headers.clone();
    if let Err(code) = require_internal_access(&headers, Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }

    let Some(assets) = state.v2_assets.as_ref().cloned() else {
        return json_resp(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"streaming requires AYAH_ALIGN_MODE=v2 assets"}),
        );
    };

    let bytes = match to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": format!("invalid body: {e}")})),
    };
    let body: CreateStreamSessionBody = if bytes.is_empty() {
        CreateStreamSessionBody::default()
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": format!("invalid json: {e}")})),
        }
    };

    let mut cfg = streaming::StreamConfig::default();
    if let Some(v) = body.sample_rate_hz {
        cfg.sample_rate_hz = v;
    }
    if let Some(v) = body.channels {
        cfg.channels = v;
    }
    if let Some(v) = body.window_s {
        cfg.window_s = v;
    }
    if let Some(v) = body.hop_s {
        cfg.hop_s = v;
    }
    if let Some(v) = body.buffer_s {
        cfg.buffer_s = v;
    }
    if let Some(v) = body.min_process_s {
        cfg.min_process_s = v;
    }
    let cfg = match streaming::validate_stream_cfg(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": e.to_string()})),
    };

    let session_id = Uuid::new_v4().simple().to_string();
    let session = streaming::StreamSession::spawn(
        session_id.clone(),
        cfg.clone(),
        state.settings.data_dir.clone(),
        state.http.clone(),
        state.settings.api_key.clone(),
        state.transcriber_pool.clone(),
        state.transcribe_sem.clone(),
        assets,
        state.settings.ayah_word_time_upgrade.clone(),
        state.settings.stream_jwt_secret.clone(),
        state.settings.stream_jwt_audience.clone(),
        state.settings.stream_auth_grace_s,
        env_f64_opt("STREAM_NO_AUDIO_TIMEOUT_S", Some(30.0)).unwrap_or(30.0),
        env_f64_opt("STREAM_SILENCE_DBFS_THRESHOLD", Some(-45.0)).unwrap_or(-45.0),
        env_f64_opt("STREAM_SILENCE_SKIP_AFTER_S", Some(1.2)).unwrap_or(1.2),
        env_f64_opt("STREAM_SILENCE_TIMEOUT_S", Some(60.0)).unwrap_or(60.0),
        env_f64_opt("STREAM_TAIL_CONTEXT_S", Some(4.0)).unwrap_or(4.0),
        env_f64_opt("STREAM_FULL_REFRESH_EVERY_S", Some(60.0)).unwrap_or(60.0),
    );
    state
        .stream_sessions
        .insert(session_id.clone(), session.clone());

    let ws_path = session.ws_path();
    let ws_url = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .map(|host| {
            let proto = headers
                .get("x-forwarded-proto")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("http");
            let http_base = format!("{proto}://{host}");
            streaming::ws_url_from_http_base(&http_base, &ws_path)
        })
        .unwrap_or_else(|| ws_path.clone());

    json_resp(
        StatusCode::OK,
        json!({
            "session_id": session_id,
            "ws_path": ws_path,
            "ws_url": ws_url,
            "config": {
                "sample_rate_hz": cfg.sample_rate_hz,
                "channels": cfg.channels,
                "window_s": cfg.window_s,
                "hop_s": cfg.hop_s,
                "buffer_s": cfg.buffer_s,
                "min_process_s": cfg.min_process_s,
            }
        }),
    )
}

async fn stop_stream_session(
    State(state): State<AppState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    AxumPath(session_id): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(code) = require_internal_access(&headers, Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }
    let Some(s) = state.stream_sessions.get(&session_id).map(|v| v.clone()) else {
        return json_resp(StatusCode::NOT_FOUND, json!({"error":"session not found"}));
    };
    s.request_stop();
    json_resp(StatusCode::OK, json!({"ok": true, "session_id": session_id}))
}

#[derive(Debug, Deserialize)]
struct InternalKeyQuery {
    key: Option<String>,
}

async fn internal_streaming_test_page(
    State(state): State<AppState>,
    Query(q): Query<InternalKeyQuery>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
) -> Response {
    if let Err(code) = require_internal_streaming_test(q.key.as_deref(), Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }

    Html(include_str!("../assets/streaming_test.html")).into_response()
}

async fn internal_create_stream_session(
    State(state): State<AppState>,
    Query(q): Query<InternalKeyQuery>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    req: Request,
) -> Response {
    if let Err(code) = require_internal_streaming_test(q.key.as_deref(), Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }

    let (_parts, body) = req.into_parts();
    let bytes = match to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": format!("invalid body: {e}")})),
    };
    let body: CreateStreamSessionBody = if bytes.is_empty() {
        CreateStreamSessionBody::default()
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": format!("invalid json: {e}")})),
        }
    };

    let Some(assets) = state.v2_assets.as_ref().cloned() else {
        return json_resp(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"streaming requires AYAH_ALIGN_MODE=v2 assets"}),
        );
    };

    let mut cfg = streaming::StreamConfig::default();
    if let Some(v) = body.sample_rate_hz {
        cfg.sample_rate_hz = v;
    }
    if let Some(v) = body.channels {
        cfg.channels = v;
    }
    if let Some(v) = body.window_s {
        cfg.window_s = v;
    }
    if let Some(v) = body.hop_s {
        cfg.hop_s = v;
    }
    if let Some(v) = body.buffer_s {
        cfg.buffer_s = v;
    }
    if let Some(v) = body.min_process_s {
        cfg.min_process_s = v;
    }
    let cfg = match streaming::validate_stream_cfg(cfg) {
        Ok(c) => c,
        Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": e.to_string()})),
    };

    let session_id = Uuid::new_v4().simple().to_string();
    let session = streaming::StreamSession::spawn(
        session_id.clone(),
        cfg.clone(),
        state.settings.data_dir.clone(),
        state.http.clone(),
        state.settings.api_key.clone(),
        state.transcriber_pool.clone(),
        state.transcribe_sem.clone(),
        assets,
        state.settings.ayah_word_time_upgrade.clone(),
        state.settings.stream_jwt_secret.clone(),
        state.settings.stream_jwt_audience.clone(),
        state.settings.stream_auth_grace_s,
        env_f64_opt("STREAM_NO_AUDIO_TIMEOUT_S", Some(30.0)).unwrap_or(30.0),
        env_f64_opt("STREAM_SILENCE_DBFS_THRESHOLD", Some(-45.0)).unwrap_or(-45.0),
        env_f64_opt("STREAM_SILENCE_SKIP_AFTER_S", Some(1.2)).unwrap_or(1.2),
        env_f64_opt("STREAM_SILENCE_TIMEOUT_S", Some(60.0)).unwrap_or(60.0),
        env_f64_opt("STREAM_TAIL_CONTEXT_S", Some(4.0)).unwrap_or(4.0),
        env_f64_opt("STREAM_FULL_REFRESH_EVERY_S", Some(60.0)).unwrap_or(60.0),
    );
    state
        .stream_sessions
        .insert(session_id.clone(), session.clone());

    let ws_path = session.ws_path();
    let ws_url = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .map(|host| {
            let proto = headers
                .get("x-forwarded-proto")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("http");
            let http_base = format!("{proto}://{host}");
            streaming::ws_url_from_http_base(&http_base, &ws_path)
        })
        .unwrap_or_else(|| ws_path.clone());

    json_resp(
        StatusCode::OK,
        json!({
            "session_id": session_id,
            "ws_path": ws_path,
            "ws_url": ws_url,
            "config": {
                "sample_rate_hz": cfg.sample_rate_hz,
                "channels": cfg.channels,
                "window_s": cfg.window_s,
                "hop_s": cfg.hop_s,
                "buffer_s": cfg.buffer_s,
                "min_process_s": cfg.min_process_s,
            }
        }),
    )
}

async fn internal_stop_stream_session(
    State(state): State<AppState>,
    Query(q): Query<InternalKeyQuery>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    AxumPath(session_id): AxumPath<String>,
) -> Response {
    if let Err(code) = require_internal_streaming_test(q.key.as_deref(), Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }
    let Some(s) = state.stream_sessions.get(&session_id).map(|v| v.clone()) else {
        return json_resp(StatusCode::NOT_FOUND, json!({"error":"session not found"}));
    };
    s.request_stop();
    json_resp(StatusCode::OK, json!({"ok": true, "session_id": session_id}))
}

#[derive(Debug, Deserialize)]
struct InternalAyahQuery {
    key: Option<String>,
    surah_id: u16,
    ayah_num: u16,
}

async fn internal_quran_ayah_lookup(
    State(state): State<AppState>,
    Query(q): Query<InternalAyahQuery>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
) -> Response {
    if let Err(code) = require_internal_streaming_test(q.key.as_deref(), Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }

    let db_path = state.settings.quran_db_path.clone();
    let surah_id = q.surah_id as i64;
    let ayah_num = q.ayah_num as i64;

    let res = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let conn = Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

        let (ayah_global, ayah_key, text_ar): (i64, String, String) = conn.query_row(
            "select ayah_global, ayah_key, text_ar from ayah where surah_id=? and ayah_num=? limit 1",
            params![surah_id, ayah_num],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;

        let (surah_name_ar, surah_transliteration, total_verses): (String, String, i64) = conn.query_row(
            "select name_ar, transliteration, total_verses from surah where surah_id=? limit 1",
            params![surah_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;

        let next = conn
            .query_row(
                "select surah_id, ayah_num, ayah_key from ayah where ayah_global=? limit 1",
                params![ayah_global + 1],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, String>(2)?)),
            )
            .ok()
            .map(|(sid, anum, akey)| json!({"surah_id": sid as u16, "ayah_num": anum as u16, "ayah_key": akey}))
            .unwrap_or(Value::Null);

        let prev = conn
            .query_row(
                "select surah_id, ayah_num, ayah_key from ayah where ayah_global=? limit 1",
                params![ayah_global - 1],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, String>(2)?)),
            )
            .ok()
            .map(|(sid, anum, akey)| json!({"surah_id": sid as u16, "ayah_num": anum as u16, "ayah_key": akey}))
            .unwrap_or(Value::Null);

        Ok(json!({
            "surah": {
                "surah_id": surah_id as u16,
                "name_ar": surah_name_ar,
                "transliteration": surah_transliteration,
                "total_verses": total_verses,
            },
            "ayah": {
                "surah_id": surah_id as u16,
                "ayah_num": ayah_num as u16,
                "ayah_global": ayah_global,
                "ayah_key": ayah_key,
                "text_ar": text_ar,
            },
            "next": next,
            "prev": prev,
        }))
    })
    .await;

    match res {
        Ok(Ok(v)) => json_resp(StatusCode::OK, v),
        Ok(Err(e)) => json_resp(StatusCode::NOT_FOUND, json!({"error": e.to_string()})),
        Err(e) => json_resp(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": format!("join error: {e}")})),
    }
}

async fn stream_session_ws(
    State(state): State<AppState>,
    AxumPath(session_id): AxumPath<String>,
    Query(q): Query<InternalKeyQuery>,
    headers: HeaderMap,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    ws: WebSocketUpgrade,
) -> Response {
    let Some(s) = state.stream_sessions.get(&session_id).map(|v| v.clone()) else {
        return json_resp(StatusCode::NOT_FOUND, json!({"error":"session not found"}));
    };

    // Public streaming auth: accept short-lived session-scoped stream JWTs (admin-only).
    // Falls back to internal access for trusted clients.
    let mut authed = false;
    let mut exp: Option<u64> = None;
    if !state.settings.stream_jwt_secret.is_empty() {
        if let Some(token) = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
            .map(|s| s.trim())
            .filter(|t| !t.is_empty())
        {
            let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
            validation.set_audience(&[state.settings.stream_jwt_audience.clone()]);
            validation.validate_exp = true;
            match jsonwebtoken::decode::<StreamJwtClaims>(
                token,
                &DecodingKey::from_secret(state.settings.stream_jwt_secret.as_bytes()),
                &validation,
            ) {
                Ok(data) => {
                    if data.claims.sid.as_deref() == Some(session_id.as_str())
                        && data.claims.role.as_deref() == Some("admin")
                    {
                        authed = true;
                        exp = Some(data.claims.exp);
                    }
                }
                Err(_) => {}
            }
        }
    }

    if !authed {
        if require_internal_streaming_test(q.key.as_deref(), Some(addr.ip()), &state.settings).is_ok() {
            authed = true;
        }
    }

    if !authed {
        if let Err(code) = require_internal_access(&headers, Some(addr.ip()), &state.settings) {
            return json_resp(code, json!({"error":"forbidden"}));
        }
    } else if let Some(e) = exp {
        s.set_auth_expiry_unix_s(e);
    }

    ws.on_upgrade(move |socket| async move {
        s.handle_ws(socket).await;
    })
}

#[derive(Debug, Deserialize)]
struct SourceUrlBody {
    source_url: String,
    filename: Option<String>,
    content_type: Option<String>,
}

async fn download_source_url_to_path(
    state: &AppState,
    source_url: &str,
    dst_path: &Path,
    max_bytes: usize,
) -> anyhow::Result<(u64, String)> {
    let url = reqwest::Url::parse(source_url)?;
    let host = url.host_str().unwrap_or("").to_string();
    if !host_allowed(&host, &state.settings.allow_source_url_hosts) {
        anyhow::bail!("source_url host not allowed: {host}");
    }

    if let Some(parent) = dst_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let resp = state.http.get(url).send().await?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("download failed: status={status}");
    }

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("audio/mp4")
        .to_string();

    let mut f = tokio::fs::File::create(dst_path).await?;
    let mut total: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        total = total.saturating_add(chunk.len() as u64);
        if total as usize > max_bytes {
            let _ = tokio::fs::remove_file(dst_path).await;
            anyhow::bail!("download too large ({} bytes > {})", total, max_bytes);
        }
        tokio::io::AsyncWriteExt::write_all(&mut f, &chunk).await?;
    }
    tokio::io::AsyncWriteExt::flush(&mut f).await?;

    Ok((total, content_type))
}

async fn create_job(
    State(state): State<AppState>,
    Query(q): Query<TranscribeQuery>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    req: Request,
) -> Response {
    let (parts, body) = req.into_parts();
    let headers = parts.headers.clone();
    if let Err(code) = require_internal_access(&headers, Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }

    if state.queue_tx.capacity() == 0 {
        return json_resp(StatusCode::SERVICE_UNAVAILABLE, json!({"error":"queue full"}));
    }

    let job_id = Uuid::new_v4().simple().to_string();

    let mut file_saved: Option<(PathBuf, u64, String, String)> = None;

    if content_type_is_json(&headers) {
        let bytes = match to_bytes(body, state.settings.max_upload_bytes).await {
            Ok(b) => b,
            Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": format!("invalid body: {e}")})),
        };
        let parsed: SourceUrlBody = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": format!("invalid json: {e}")})),
        };
        let source_url = parsed.source_url.trim().to_string();
        if source_url.is_empty() {
            return json_resp(StatusCode::BAD_REQUEST, json!({"error":"missing source_url"}));
        }

        let filename = parsed
            .filename
            .as_deref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "source.m4a".to_string());
        let input_path = job_input_path(&state.settings.data_dir, &job_id, &filename);
        let ct_override = parsed
            .content_type
            .as_deref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        match download_source_url_to_path(&state, &source_url, &input_path, state.settings.max_upload_bytes).await {
            Ok((size_bytes, detected_ct)) => {
                let content_type = ct_override.unwrap_or(detected_ct);
                file_saved = Some((input_path, size_bytes, filename, content_type));
            }
            Err(e) => {
                cleanup::remove_job_dir_for_upload_failure(&state, &job_id).await;
                return json_resp(
                    StatusCode::BAD_GATEWAY,
                    json!({"error": format!("source_url download failed: {e}")}),
                );
            }
        }
    } else {
        let mut multipart = match Multipart::from_request(Request::from_parts(parts, body), &state).await {
            Ok(m) => m,
            Err(_) => return json_resp(StatusCode::BAD_REQUEST, json!({"error":"missing file"})),
        };

        while let Ok(Some(field)) = multipart.next_field().await {
            if field.name().unwrap_or("") != "file" {
                continue;
            }
            let filename = field
                .file_name()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "upload.m4a".to_string());
            let input_path = job_input_path(&state.settings.data_dir, &job_id, &filename);
            match safe_stream_copy_to_path(field, &input_path, state.settings.max_upload_bytes).await {
                Ok((size_bytes, filename2, content_type)) => {
                    file_saved = Some((input_path, size_bytes, filename2, content_type));
                }
                Err(e) => {
                    cleanup::remove_job_dir_for_upload_failure(&state, &job_id).await;
                    return json_resp(
                        StatusCode::BAD_REQUEST,
                        json!({"error": format!("upload failed: {}", e)}),
                    );
                }
            }
            break;
        }
    }

    let Some((input_path, size_bytes, filename, content_type)) = file_saved else {
        cleanup::remove_job_dir_for_upload_failure(&state, &job_id).await;
        return json_resp(StatusCode::BAD_REQUEST, json!({"error":"missing file"}));
    };

    let req_obj = json!({
        "job_id": job_id,
        "created_at": now_ts(),
        "input": {
            "filename": filename,
            "content_type": content_type,
            "path": input_path,
            "size_bytes": size_bytes,
        },
        "transcription": {
            "language": q.language,
            "beam_size": q.beam_size,
            "patience": q.patience,
            "word_timestamps": q.word_timestamps,
            "log_prob_threshold": q.log_prob_threshold,
            "no_speech_threshold": q.no_speech_threshold,
            "batch_size": q.batch_size,
        },
        "guess": {
            "top_k": q.top_k,
            "per_surah_top_n": q.per_surah_top_n,
            "query_operator": q.query_operator,
            "rank_mode": q.rank_mode,
            "max_query_tokens": q.max_query_tokens,
            "max_hits": q.max_hits,
        }
    });

    if let Err(e) = write_json(&job_request_path(&state.settings.data_dir, &job_id), &req_obj).await {
        error!(job_id = %job_id, "failed to write request.json: {e}");
    }

    let meta = JobMeta {
        job_id: job_id.clone(),
        status: "queued".to_string(),
        created_at: now_ts(),
        updated_at: now_ts(),
        started_at: None,
        finished_at: None,
        filename: filename.clone(),
        content_type: content_type.clone(),
        size_bytes,
        error: None,
    };
    state.jobs.insert(job_id.clone(), meta);

    if let Err(_e) = state.queue_tx.try_send(job_id.clone()) {
        state.jobs.remove(&job_id);
        cleanup::remove_job_dir_for_upload_failure(&state, &job_id).await;
        return json_resp(StatusCode::SERVICE_UNAVAILABLE, json!({"error":"queue full"}));
    }

    if q.wait.unwrap_or(false) {
        let timeout_s = q.wait_timeout_s.unwrap_or(0);
        if timeout_s <= 0 {
            return json_resp(StatusCode::OK, json!({"job_id": job_id, "status": "queued"}));
        }
        let deadline = now_ts() + timeout_s as f64;
        loop {
            let finished = {
                if let Some(m) = state.jobs.get(&job_id) {
                    m.status == "done" || m.status == "error"
                } else {
                    false
                }
            };
            if finished {
                return get_job_status(
                    State(state),
                    headers,
                    axum::extract::ConnectInfo(addr),
                    AxumPath(job_id),
                )
                .await;
            }
            if now_ts() >= deadline {
                break;
            }
            sleep(Duration::from_millis(250)).await;
        }
    }

    json_resp(StatusCode::OK, json!({"job_id": job_id, "status": "queued"}))
}

async fn get_job_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    if let Err(code) = require_internal_access(&headers, Some(addr.ip()), &state.settings) {
        return json_resp(code, json!({"error":"forbidden"}));
    }

    let Some(meta) = state.jobs.get(&job_id).map(|m| m.clone()) else {
        return json_resp(StatusCode::NOT_FOUND, json!({"error":"job not found"}));
    };

    let mut payload = json!({
        "job_id": meta.job_id,
        "status": meta.status,
        "created_at": meta.created_at,
        "updated_at": meta.updated_at,
        "started_at": meta.started_at,
        "finished_at": meta.finished_at,
        "filename": meta.filename,
        "content_type": meta.content_type,
        "size_bytes": meta.size_bytes,
        "error": meta.error,
    });

    if meta.status == "done" {
        let result_path = job_result_path(&state.settings.data_dir, &job_id);
        if result_path.exists() {
            if let Ok(result) = read_json(&result_path).await {
                payload["result"] = result;
            }
        }
    }

    json_resp(StatusCode::OK, payload)
}

pub async fn process_job(job_id: &str, state: &AppState) -> anyhow::Result<()> {
    let Some(mut meta) = state.jobs.get(job_id).map(|m| m.clone()) else {
        anyhow::bail!("job not found");
    };

    meta.status = "running".to_string();
    meta.updated_at = now_ts();
    meta.started_at = Some(now_ts());
    state.jobs.insert(job_id.to_string(), meta.clone());

    let req_path = job_request_path(&state.settings.data_dir, job_id);
    let req_obj = read_json(&req_path).await.unwrap_or_else(|_| json!({}));
    let input = req_obj.get("input").cloned().unwrap_or_else(|| json!({}));

    let filename = input
        .get("filename")
        .and_then(|v| v.as_str())
        .unwrap_or(&meta.filename)
        .to_string();
    let _content_type = input
        .get("content_type")
        .and_then(|v| v.as_str())
        .unwrap_or(&meta.content_type)
        .to_string();
    let input_path = job_input_path(&state.settings.data_dir, job_id, &filename);

    let transcribe_cfg = req_obj.get("transcription").cloned().unwrap_or_else(|| json!({}));

    let (transcriber_json, transcribe_s) = {
        let _permit = state
            .transcribe_sem
            .acquire()
            .await
            .map_err(|_| anyhow::anyhow!("transcribe semaphore closed"))?;
        let lease = state.transcriber_pool.pick();
        let url = format!("{}/v1/transcribe", lease.url().trim_end_matches('/'));
        let mut req = state.http.post(url);
        if !state.settings.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", state.settings.api_key));
        }

        let file = tokio::fs::File::open(&input_path).await?;
        let stream = tokio_util::io::ReaderStream::new(file);
        let body = reqwest::Body::wrap_stream(stream);
        let part = Part::stream(body).file_name(filename.clone());

        let form = Form::new().part("file", part);

        // Pass transcriber overrides via query string (FastAPI parses them as query params).
        let mut qp: Vec<(String, String)> = Vec::new();
        for (k, v) in [
            ("language", transcribe_cfg.get("language")),
            ("beam_size", transcribe_cfg.get("beam_size")),
            ("patience", transcribe_cfg.get("patience")),
            ("word_timestamps", transcribe_cfg.get("word_timestamps")),
            ("log_prob_threshold", transcribe_cfg.get("log_prob_threshold")),
            ("no_speech_threshold", transcribe_cfg.get("no_speech_threshold")),
            ("batch_size", transcribe_cfg.get("batch_size")),
        ] {
            if let Some(v) = v {
                if !v.is_null() {
                    qp.push((k.to_string(), v.to_string().trim_matches('"').to_string()));
                }
            }
        }

        let req = req.query(&qp).multipart(form);

        let t0 = now_ts();
        let resp = req.send().await?;
        let status = resp.status();
        let transcriber_json: Value = resp.json().await.unwrap_or_else(|_| json!({}));
        if !status.is_success() {
            anyhow::bail!(
                "transcriber error ({}): {}",
                status.as_u16(),
                transcriber_json
            );
        }
        let transcribe_s = now_ts() - t0;
        drop(lease);
        (transcriber_json, transcribe_s)
    };

    let transcription = std::sync::Arc::new(
        transcriber_json
            .get("transcription")
            .cloned()
            .unwrap_or_else(|| json!({})),
    );
    let text = std::sync::Arc::new(
        transcriber_json
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    );
    let t_params = transcriber_json.get("params").cloned().unwrap_or_else(|| json!({}));

    let guess_cfg = req_obj.get("guess").cloned().unwrap_or_else(|| json!({}));
    let top_k = guess_cfg
        .get("top_k")
        .and_then(|v| v.as_i64())
        .map(|v| v.max(1) as usize)
        .unwrap_or(state.settings.guess_top_k);
    let per_surah_top_n = guess_cfg
        .get("per_surah_top_n")
        .and_then(|v| v.as_i64())
        .map(|v| v.max(1) as usize)
        .unwrap_or(state.settings.guess_per_surah_top_n);
    let query_operator = guess_cfg
        .get("query_operator")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| state.settings.guess_query_operator.clone());
    let rank_mode = guess_cfg
        .get("rank_mode")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| state.settings.guess_rank_mode.clone());
    let max_query_tokens = guess_cfg
        .get("max_query_tokens")
        .and_then(|v| v.as_i64())
        .map(|v| v.max(1) as usize)
        .unwrap_or(state.settings.guess_max_query_tokens);
    let max_hits = guess_cfg
        .get("max_hits")
        .and_then(|v| v.as_i64())
        .map(|v| v.max(1) as usize)
        .unwrap_or(state.settings.guess_max_hits);

    // Guess + alignment are CPU-bound and independent once transcription is available.
    // Run them in parallel on blocking threads to reduce end-to-end latency and avoid blocking Tokio.
    let guess_task = {
        let text = text.clone();
        let quran_db_path = state.settings.quran_db_path.clone();
        let query_operator = query_operator.clone();
        let rank_mode = rank_mode.clone();
        tokio::task::spawn_blocking(move || {
            let t0 = std::time::Instant::now();
            let guess_full = guessing::guess_from_transcript_text(
                text.as_str(),
                &quran_db_path,
                top_k,
                per_surah_top_n,
                &query_operator,
                &rank_mode,
                max_query_tokens,
                max_hits,
            );
            let guess_s = t0.elapsed().as_secs_f64();
            (guess_full, guess_s)
        })
    };

    let align_task = state.v2_assets.as_ref().cloned().map(|assets| {
        let transcription = transcription.clone();
        let text = text.clone();
        let word_time_upgrade = state.settings.ayah_word_time_upgrade.clone();
        tokio::task::spawn_blocking(move || {
            let t0 = std::time::Instant::now();
            let ms = multispan::align_multi_span(
                transcription.as_ref(),
                text.as_str(),
                assets.as_ref(),
                &word_time_upgrade,
            );
            let align_s = t0.elapsed().as_secs_f64();
            (
                ms.active_alignment,
                ms.spans,
                ms.jumps,
                ms.active_span_index,
                align_s,
            )
        })
    });

    let (guess_full, guess_s) = guess_task
        .await
        .map_err(|e| anyhow::anyhow!("guess join error: {e}"))?;

    let (ayah_alignment, ayah_alignment_spans, ayah_alignment_jumps, ayah_alignment_active_span_index, align_s) =
        match align_task {
            None => (Value::Null, Value::Null, Value::Null, None, 0.0),
            Some(h) => h
                .await
                .map_err(|e| anyhow::anyhow!("alignment join error: {e}"))?,
        };

    let mut guess = guess_full.clone();
    let ayah_alignment_active_span_index = ayah_alignment_active_span_index
        .map(|v| Value::from(v as i64))
        .unwrap_or(Value::Null);

    // If we have a confident alignment, prefer its surah_id for the primary guess.
    if let (Some(g), Some(a)) = (guess.as_object_mut(), ayah_alignment.as_object()) {
        if a.get("error").is_none() {
            let conf = a.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if let Some(start) = a.get("start").and_then(|v| v.as_object()) {
                if let Some(surah_id) = start.get("surah_id").and_then(|v| v.as_i64()) {
                    let end_surah_id = a
                        .get("end")
                        .and_then(|v| v.as_object())
                        .and_then(|o| o.get("surah_id"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(surah_id);

                    let (matches, subs) = a
                        .get("alignment")
                        .and_then(|v| v.as_object())
                        .map(|o| {
                            let m = o.get("matches").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let s = o.get("substitutions").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            (m, s)
                        })
                        .unwrap_or((0, 0));

                    // Calibrated to avoid overriding based on tiny/ambiguous fragments (e.g. 1-2 words),
                    // while still trusting strong single-surah alignments even when confidence is modest.
                    let min_conf = if surah_id == end_surah_id { 0.35 } else { 0.50 };
                    let has_evidence = subs >= 6 || matches >= 4;

                    if conf >= min_conf && has_evidence {
                        let prev = g.get("predicted_surah_id").cloned().unwrap_or(Value::Null);
                        g.insert("predicted_surah_id_guess".to_string(), prev);
                        g.insert("predicted_surah_id".to_string(), Value::from(surah_id));
                        g.insert(
                            "aligned_range".to_string(),
                            json!({
                                "start": a.get("start").cloned().unwrap_or(Value::Null),
                                "end": a.get("end").cloned().unwrap_or(Value::Null),
                                "confidence": conf,
                            }),
                        );
                    }
                }
            }
        }
    }

    let started_at = meta.started_at.unwrap_or_else(now_ts);
    let result = json!({
        "job_id": job_id,
        "input": input,
        "transcription": {
            "language": transcription.get("language"),
            "language_probability": transcription.get("language_probability"),
            "duration_s": transcription.get("duration_s"),
            "segments": transcription.get("segments"),
            "repair": transcription.get("repair"),
            "dedupe": transcription.get("dedupe"),
            "text": text.as_str(),
            "params": t_params,
        },
        "guess": guess,
        "guess_full": guess_full,
        "guess_tail": Value::Null,
        "guess_tail_last_segment": Value::Null,
        "ayah_alignment": ayah_alignment,
        "ayah_alignment_full": ayah_alignment,
        "ayah_alignment_full_v2": ayah_alignment,
        "ayah_alignment_spans": ayah_alignment_spans,
        "ayah_alignment_jumps": ayah_alignment_jumps,
        "ayah_alignment_active_span_index": ayah_alignment_active_span_index,
        "ayah_alignment_tail": Value::Null,
        "ayah_alignment_tail_last_segment": Value::Null,
        "tail_recovery": Value::Null,
        "timing": {
            "transcribe_s": transcribe_s,
            "guess_s": guess_s,
            "align_s": align_s,
            "total_s": now_ts() - started_at,
        }
    });

    write_json(&job_result_path(&state.settings.data_dir, job_id), &result).await?;

    meta.status = "done".to_string();
    meta.updated_at = now_ts();
    meta.finished_at = Some(now_ts());
    state.jobs.insert(job_id.to_string(), meta);
    Ok(())
}

async fn worker_loop(rx: Arc<Mutex<mpsc::Receiver<String>>>, state: AppState, worker_id: usize) {
    loop {
        let job_id = {
            let mut guard = rx.lock().await;
            guard.recv().await
        };
        let Some(job_id) = job_id else { break };

        info!(worker_id = worker_id, job_id = %job_id, "processing");
        let res = process_job(&job_id, &state).await;
        if let Err(e) = res {
            error!(worker_id = worker_id, job_id = %job_id, "job failed: {e}");
            if let Some(mut meta) = state.jobs.get(&job_id).map(|m| m.clone()) {
                meta.status = "error".to_string();
                meta.updated_at = now_ts();
                meta.finished_at = Some(now_ts());
                meta.error = Some(e.to_string());
                state.jobs.insert(job_id, meta);
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let settings = load_settings();

    let v2_assets = if matches!(settings.ayah_align_mode.as_str(), "v2" | "auto") {
        match resolve_v2_assets_dir(&settings.ayah_align_v2_assets_dir) {
            None => None,
            Some(dir) => {
                let assets = alignment_v2::AlignmentAssetsV2::load(
                    &dir,
                    settings.ayah_align_v2_ref_del_cost_cap,
                    settings.ayah_align_v2_obs_ins_cost_cap,
                )
                .map(Arc::new)
                .map_err(|e| anyhow::anyhow!("v2 assets load failed: {e}"))?;
                info!(
                    "v2 assets loaded: version={} embed_model={}",
                    assets.version,
                    if assets.embedding_model_name.is_empty() {
                        "none"
                    } else {
                        assets.embedding_model_name.as_str()
                    }
                );
                Some(assets)
            }
        }
    } else {
        None
    };

    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60 * 60))
        .pool_max_idle_per_host(8)
        .build()?;

    let (tx, rx) = mpsc::channel::<String>(settings.max_queue_length);
    let state = AppState {
        settings: settings.clone(),
        http,
        jobs: Arc::new(DashMap::new()),
        queue_tx: tx,
        v2_assets,
        stream_sessions: Arc::new(DashMap::new()),
        transcriber_pool: TranscriberPool::new(settings.transcriber_urls.clone()),
        transcribe_sem: Arc::new(Semaphore::new(settings.rust_transcribe_max_inflight.max(1))),
    };

    let rx = Arc::new(Mutex::new(rx));
    let workers = settings.worker_concurrency.max(1);
    for wid in 0..workers {
        tokio::spawn(worker_loop(rx.clone(), state.clone(), wid));
    }
    cleanup::spawn_job_cleanup(state.clone());

    let mut app = Router::new()
        .route("/health", get(health))
        .route("/v1/jobs/transcribe", post(create_job))
        .route("/v1/jobs/:job_id", get(get_job_status))
        .route("/v1/sessions", post(create_stream_session))
        .route("/v1/sessions/:session_id/stream", get(stream_session_ws))
        .route("/v1/sessions/:session_id/stop", post(stop_stream_session))
        .route("/internal/streaming_test", get(internal_streaming_test_page))
        .route("/internal/streaming_session", post(internal_create_stream_session))
        .route("/internal/sessions/:session_id/stop", post(internal_stop_stream_session))
        .route("/internal/quran/ayah", get(internal_quran_ayah_lookup))
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(settings.max_upload_bytes))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    if settings.demo_enabled {
        info!("demo mode enabled — mounting /demo/* routes");
        let demo_cors = {
            use tower_http::cors::{CorsLayer, Any};
            use axum::http::{Method, header};
            let origins = &settings.demo_allowed_origins;
            let layer = if origins.iter().any(|o| o == "*") {
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                    .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
            } else {
                let allowed: Vec<_> = origins
                    .iter()
                    .filter_map(|o| o.parse::<axum::http::HeaderValue>().ok())
                    .collect();
                CorsLayer::new()
                    .allow_origin(allowed)
                    .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                    .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
            };
            layer
        };
        let demo = demo::demo_router(state.clone()).layer(demo_cors);
        app = app.nest("/demo", demo);
    }

    let addr: SocketAddr = "0.0.0.0:8001".parse().unwrap();
    info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}

fn resolve_v2_assets_dir(path: &Path) -> Option<PathBuf> {
    let path = PathBuf::from(path);
    let meta = path.join("meta.json");
    if meta.is_file() {
        return Some(path);
    }
    if !path.is_dir() {
        return None;
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    let entries = fs::read_dir(&path).ok()?;
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        if p.join("meta.json").is_file() {
            candidates.push(p);
        }
    }
    candidates.sort();
    candidates.pop()
}

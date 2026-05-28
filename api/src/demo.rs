use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::to_bytes;
use axum::extract::{FromRequest, Multipart, Path as AxumPath, Query, State};
use axum::extract::ws::WebSocketUpgrade;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::state::{AppState, JobMeta, job_input_path, job_request_path, job_result_path, now_ts};
use crate::{json_resp, safe_stream_copy_to_path, write_json, read_json, CreateStreamSessionBody};
use crate::cleanup;

#[derive(Clone)]
pub struct DemoRateLimiter {
    inner: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    max_per_min: usize,
    sweep_counter: Arc<AtomicUsize>,
}

impl DemoRateLimiter {
    pub fn new(max_per_min: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            max_per_min,
            sweep_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Returns `Ok(())` if the request is allowed, or `Err(retry_after_s)` with a hint
    /// (seconds until a slot frees up) when the per-minute limit is exceeded.
    pub async fn check(&self, key: &str) -> Result<(), u64> {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(60);
        let mut map = self.inner.lock().await;

        // Amortized sweep so keys for IPs that stopped sending requests don't pile up;
        // keeps the map proportional to recently-active clients rather than all-time IPs.
        if self.sweep_counter.fetch_add(1, Ordering::Relaxed) % 512 == 0 {
            map.retain(|_k, times| {
                times.retain(|t| *t > cutoff);
                !times.is_empty()
            });
        }

        let entry = map.entry(key.to_string()).or_default();
        entry.retain(|t| *t > cutoff);
        if entry.len() >= self.max_per_min {
            let oldest = entry.first().copied().unwrap_or(now);
            let retry = 60u64
                .saturating_sub(now.duration_since(oldest).as_secs())
                .max(1);
            return Err(retry);
        }
        entry.push(now);
        Ok(())
    }
}

#[derive(Clone)]
pub struct DemoState {
    pub app: AppState,
    pub rate_limiter: DemoRateLimiter,
}

fn client_key(addr: &SocketAddr) -> String {
    addr.ip().to_string()
}

#[derive(Debug, Deserialize)]
struct DemoTranscribeQuery {
    wait: Option<bool>,
    wait_timeout_s: Option<i64>,
}

async fn demo_transcribe(
    State(state): State<DemoState>,
    Query(q): Query<DemoTranscribeQuery>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    req: Request,
) -> Response {
    if let Err(retry_after_s) = state.rate_limiter.check(&client_key(&addr)).await {
        return json_resp(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error": "rate limit exceeded, try again later", "retry_after_s": retry_after_s}),
        );
    }

    let (parts, body) = req.into_parts();
    let app = &state.app;

    if app.queue_tx.capacity() == 0 {
        return json_resp(StatusCode::SERVICE_UNAVAILABLE, json!({"error": "queue full, try again later"}));
    }

    let job_id = Uuid::new_v4().simple().to_string();
    let max_bytes = app.settings.demo_max_upload_bytes;

    let mut multipart: Multipart = match Multipart::from_request(Request::from_parts(parts, body), &state).await {
        Ok(m) => m,
        Err(_) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": "missing audio file"})),
    };

    let mut file_saved: Option<(std::path::PathBuf, u64, String, String)> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name != "file" {
            continue;
        }
        let filename = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "upload.m4a".to_string());
        let input_path = job_input_path(&app.settings.data_dir, &job_id, &filename);
        match safe_stream_copy_to_path(field, &input_path, max_bytes).await {
            Ok((size_bytes, filename2, content_type)) => {
                file_saved = Some((input_path, size_bytes, filename2, content_type));
            }
            Err(e) => {
                cleanup::remove_job_dir_for_upload_failure(app, &job_id).await;
                return json_resp(
                    StatusCode::BAD_REQUEST,
                    json!({"error": format!("upload failed: {}", e)}),
                );
            }
        }
        break;
    }

    let Some((input_path, size_bytes, filename, content_type)) = file_saved else {
        cleanup::remove_job_dir_for_upload_failure(app, &job_id).await;
        return json_resp(StatusCode::BAD_REQUEST, json!({"error": "missing audio file"}));
    };

    let req_obj = json!({
        "job_id": job_id,
        "created_at": now_ts(),
        "demo": true,
        "input": {
            "filename": filename,
            "content_type": content_type,
            "path": input_path,
            "size_bytes": size_bytes,
        },
        "transcription": {},
        "guess": {},
    });

    if let Err(e) = write_json(&job_request_path(&app.settings.data_dir, &job_id), &req_obj).await {
        tracing::error!(job_id = %job_id, "failed to write request.json: {e}");
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
    app.jobs.insert(job_id.clone(), meta);

    if let Err(_e) = app.queue_tx.try_send(job_id.clone()) {
        app.jobs.remove(&job_id);
        cleanup::remove_job_dir_for_upload_failure(app, &job_id).await;
        return json_resp(StatusCode::SERVICE_UNAVAILABLE, json!({"error": "queue full"}));
    }

    let wait = q.wait.unwrap_or(true);
    let timeout_s = q.wait_timeout_s.unwrap_or(120);

    if wait && timeout_s > 0 {
        let deadline = now_ts() + timeout_s as f64;
        loop {
            let finished = {
                if let Some(m) = app.jobs.get(&job_id) {
                    m.status == "done" || m.status == "error"
                } else {
                    false
                }
            };
            if finished {
                let Some(meta) = app.jobs.get(&job_id).map(|m| m.clone()) else {
                    return json_resp(StatusCode::NOT_FOUND, json!({"error": "job not found"}));
                };
                let mut payload = json!({
                    "job_id": meta.job_id,
                    "status": meta.status,
                    "error": meta.error,
                });
                if meta.status == "done" {
                    let result_path = job_result_path(&app.settings.data_dir, &job_id);
                    if result_path.exists() {
                        if let Ok(result) = read_json(&result_path).await {
                            payload["result"] = result;
                        }
                    }
                }
                return json_resp(StatusCode::OK, payload);
            }
            if now_ts() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    json_resp(StatusCode::OK, json!({"job_id": job_id, "status": "queued"}))
}

async fn demo_job_status(
    State(state): State<DemoState>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    let app = &state.app;
    let Some(meta) = app.jobs.get(&job_id).map(|m| m.clone()) else {
        return json_resp(StatusCode::NOT_FOUND, json!({"error": "job not found"}));
    };

    let mut payload = json!({
        "job_id": meta.job_id,
        "status": meta.status,
        "error": meta.error,
    });

    if meta.status == "done" {
        let result_path = job_result_path(&app.settings.data_dir, &job_id);
        if result_path.exists() {
            if let Ok(result) = read_json(&result_path).await {
                payload["result"] = result;
            }
        }
    }

    json_resp(StatusCode::OK, payload)
}

async fn demo_create_stream_session(
    State(state): State<DemoState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    req: Request,
) -> Response {
    if let Err(retry_after_s) = state.rate_limiter.check(&client_key(&addr)).await {
        return json_resp(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error": "rate limit exceeded, try again later", "retry_after_s": retry_after_s}),
        );
    }

    let app = &state.app;

    let Some(assets) = app.v2_assets.as_ref().cloned() else {
        return json_resp(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error": "streaming not available on this instance"}),
        );
    };

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

    // Demo sessions are capped at the configured max audio duration (enforced
    // server-side in the session watchdog).
    let (session_id, cfg, session) = match crate::spawn_and_register_stream_session(
        app,
        assets,
        &body,
        Some(app.settings.demo_max_audio_duration_s),
    ) {
        Ok(v) => v,
        Err(e) => return json_resp(StatusCode::BAD_REQUEST, json!({"error": e})),
    };

    let ws_path = session.ws_path();

    json_resp(
        StatusCode::OK,
        json!({
            "session_id": session_id,
            "ws_path": ws_path,
            "config": {
                "sample_rate_hz": cfg.sample_rate_hz,
                "channels": cfg.channels,
                "window_s": cfg.window_s,
                "hop_s": cfg.hop_s,
                "buffer_s": cfg.buffer_s,
                "min_process_s": cfg.min_process_s,
            },
            "max_duration_s": state.app.settings.demo_max_audio_duration_s,
        }),
    )
}

async fn demo_stream_ws(
    State(state): State<DemoState>,
    AxumPath(session_id): AxumPath<String>,
    ws: WebSocketUpgrade,
) -> Response {
    let app = &state.app;
    let Some(s) = app.stream_sessions.get(&session_id).map(|v| v.clone()) else {
        return json_resp(StatusCode::NOT_FOUND, json!({"error": "session not found"}));
    };

    ws.on_upgrade(move |socket| async move {
        s.handle_ws(socket).await;
    })
}

async fn demo_stop_stream_session(
    State(state): State<DemoState>,
    AxumPath(session_id): AxumPath<String>,
) -> Response {
    let app = &state.app;
    let Some(s) = app.stream_sessions.get(&session_id).map(|v| v.clone()) else {
        return json_resp(StatusCode::NOT_FOUND, json!({"error": "session not found"}));
    };
    s.request_stop();
    json_resp(StatusCode::OK, json!({"ok": true, "session_id": session_id}))
}

async fn demo_health() -> Response {
    json_resp(StatusCode::OK, json!({"ok": true, "demo": true}))
}

pub fn demo_router(app_state: AppState) -> Router {
    let demo_state = DemoState {
        rate_limiter: DemoRateLimiter::new(app_state.settings.demo_rate_limit_per_min),
        app: app_state,
    };

    Router::new()
        .route("/health", get(demo_health))
        .route("/v1/transcribe", post(demo_transcribe))
        .route("/v1/jobs/:job_id", get(demo_job_status))
        .route("/v1/sessions", post(demo_create_stream_session))
        .route("/v1/sessions/:session_id/stream", get(demo_stream_ws))
        .route("/v1/sessions/:session_id/stop", post(demo_stop_stream_session))
        .with_state(demo_state)
}

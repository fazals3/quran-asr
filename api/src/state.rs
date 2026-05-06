use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::sync::Semaphore;

use crate::alignment_v2;
use crate::config::Settings;
use crate::streaming;
use crate::transcriber_pool::TranscriberPool;

pub fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[derive(Clone)]
pub struct AppState {
    pub settings: Settings,
    pub http: reqwest::Client,
    pub jobs: Arc<DashMap<String, JobMeta>>,
    pub queue_tx: mpsc::Sender<String>,
    pub v2_assets: Option<Arc<alignment_v2::AlignmentAssetsV2>>,
    pub stream_sessions: Arc<DashMap<String, Arc<streaming::StreamSession>>>,
    pub transcriber_pool: TranscriberPool,
    pub transcribe_sem: Arc<Semaphore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobMeta {
    pub job_id: String,
    pub status: String,
    pub created_at: f64,
    pub updated_at: f64,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: u64,
    pub error: Option<String>,
}

pub fn job_dir(data_dir: &Path, job_id: &str) -> PathBuf {
    data_dir.join("jobs").join(job_id)
}

pub fn job_input_path(data_dir: &Path, job_id: &str, filename: &str) -> PathBuf {
    job_dir(data_dir, job_id).join("input").join(filename)
}

pub fn job_request_path(data_dir: &Path, job_id: &str) -> PathBuf {
    job_dir(data_dir, job_id).join("request.json")
}

pub fn job_result_path(data_dir: &Path, job_id: &str) -> PathBuf {
    job_dir(data_dir, job_id).join("result.json")
}

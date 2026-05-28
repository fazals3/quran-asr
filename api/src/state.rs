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
    job_dir(data_dir, job_id)
        .join("input")
        .join(sanitize_upload_filename(filename))
}

/// Reduce an untrusted upload filename to a safe basename so it can never escape
/// the job's `input/` directory (path traversal). Strips any directory components
/// and rejects empty / `.` / `..` names, falling back to a default.
pub fn sanitize_upload_filename(filename: &str) -> String {
    Path::new(filename)
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && *s != "." && *s != "..")
        .unwrap_or("upload.bin")
        .to_string()
}

pub fn job_request_path(data_dir: &Path, job_id: &str) -> PathBuf {
    job_dir(data_dir, job_id).join("request.json")
}

pub fn job_result_path(data_dir: &Path, job_id: &str) -> PathBuf {
    job_dir(data_dir, job_id).join("result.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_keeps_plain_names() {
        assert_eq!(sanitize_upload_filename("recitation.m4a"), "recitation.m4a");
        assert_eq!(sanitize_upload_filename("  spaced.wav  "), "spaced.wav");
    }

    #[test]
    fn sanitize_strips_traversal() {
        assert_eq!(sanitize_upload_filename("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_upload_filename("/abs/path/foo.mp3"), "foo.mp3");
        assert_eq!(sanitize_upload_filename("a/b/c.ogg"), "c.ogg");
    }

    #[test]
    fn sanitize_falls_back_on_dangerous_or_empty() {
        for bad in ["", "..", ".", "/", "../", "   "] {
            assert_eq!(sanitize_upload_filename(bad), "upload.bin", "input: {bad:?}");
        }
    }

    #[test]
    fn job_input_path_stays_within_job_dir() {
        let data_dir = Path::new("/data");
        let p = job_input_path(data_dir, "job123", "../../escape.m4a");
        assert_eq!(p, Path::new("/data/jobs/job123/input/escape.m4a"));
        // No `..` components survive into the constructed path.
        assert!(!p.components().any(|c| c.as_os_str() == ".."));
    }
}

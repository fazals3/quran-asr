use std::path::PathBuf;

use ipnet::IpNet;

#[derive(Clone)]
pub struct Settings {
    pub api_key: String,
    pub allow_cidrs: Vec<IpNet>,
    pub enforce_allow_cidrs_with_api_key: bool,
    pub internal_streaming_test_enabled: bool,
    pub internal_streaming_test_key: String,
    pub data_dir: PathBuf,
    pub quran_db_path: PathBuf,
    pub job_cleanup_enabled: bool,
    pub job_retention_s: u64,
    pub job_cleanup_interval_s: u64,
    pub job_cleanup_startup: bool,
    pub job_cleanup_max_delete_per_run: usize,
    pub max_queue_length: usize,
    pub max_upload_bytes: usize,
    pub worker_concurrency: usize,
    pub rust_transcribe_max_inflight: usize,
    pub transcriber_url: String,
    pub transcriber_urls: Vec<String>,
    pub allow_source_url_hosts: Vec<String>,
    pub ayah_align_mode: String,
    pub ayah_align_v2_assets_dir: PathBuf,
    pub ayah_word_time_upgrade: String,
    pub ayah_align_v2_ref_del_cost_cap: Option<f64>,
    pub ayah_align_v2_obs_ins_cost_cap: Option<f64>,
    pub stream_jwt_secret: String,
    pub stream_jwt_audience: String,
    pub stream_auth_grace_s: u64,
    pub guess_top_k: usize,
    pub guess_per_surah_top_n: usize,
    pub guess_query_operator: String,
    pub guess_rank_mode: String,
    pub guess_max_query_tokens: usize,
    pub guess_max_hits: usize,
}

pub fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

pub fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_lowercase();
            match s.as_str() {
                "1" | "true" | "yes" | "on" => true,
                "0" | "false" | "no" | "off" => false,
                _ => default,
            }
        }
        Err(_) => default,
    }
}

pub fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => v.parse().unwrap_or(default),
        _ => default,
    }
}

pub fn env_f64_opt(name: &str, default: Option<f64>) -> Option<f64> {
    match std::env::var(name) {
        Ok(v) if v.trim().is_empty() => default,
        Ok(v) => {
            let s = v.trim().to_lowercase();
            if matches!(s.as_str(), "none" | "null" | "off" | "false" | "disabled") {
                None
            } else {
                v.parse::<f64>().ok().or(default)
            }
        }
        Err(_) => default,
    }
}

fn parse_allow_cidrs(raw: &str) -> Vec<IpNet> {
    raw.split(',')
        .filter_map(|p| p.trim().parse::<IpNet>().ok())
        .collect()
}

fn parse_csv_hosts(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_csv_urls(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn load_settings() -> Settings {
    let allow = env(
        "ALLOW_CIDRS",
        "100.64.0.0/10,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.0/8",
    );
    let transcriber_url = env("TRANSCRIBER_URL", "http://transcriber:9000");
    let mut transcriber_urls = parse_csv_urls(&env("TRANSCRIBER_URLS", ""));
    if transcriber_urls.is_empty() {
        transcriber_urls.push(transcriber_url.clone());
    }
    for u in &mut transcriber_urls {
        while u.ends_with('/') {
            u.pop();
        }
    }
    Settings {
        api_key: env("API_KEY", "").trim().to_string(),
        allow_cidrs: parse_allow_cidrs(&allow),
        enforce_allow_cidrs_with_api_key: env_bool("ENFORCE_ALLOW_CIDRS_WITH_API_KEY", false),
        internal_streaming_test_enabled: env_bool("INTERNAL_STREAMING_TEST_ENABLED", false),
        internal_streaming_test_key: env("INTERNAL_STREAMING_TEST_KEY", "").trim().to_string(),
        data_dir: PathBuf::from(env("DATA_DIR", "/data")),
        quran_db_path: PathBuf::from(env("QURAN_DB_PATH", "/data/quran.db")),
        job_cleanup_enabled: env_bool("JOB_CLEANUP_ENABLED", true),
        job_retention_s: env_usize("JOB_RETENTION_S", 24 * 60 * 60) as u64,
        job_cleanup_interval_s: env_usize("JOB_CLEANUP_INTERVAL_S", 60 * 60) as u64,
        job_cleanup_startup: env_bool("JOB_CLEANUP_STARTUP", true),
        job_cleanup_max_delete_per_run: env_usize("JOB_CLEANUP_MAX_DELETE_PER_RUN", 200),
        max_queue_length: env_usize("MAX_QUEUE_LENGTH", 32),
        max_upload_bytes: env_usize("MAX_UPLOAD_BYTES", 200 * 1024 * 1024),
        worker_concurrency: env_usize("WORKER_CONCURRENCY", 2),
        rust_transcribe_max_inflight: env_usize("RUST_TRANSCRIBE_MAX_INFLIGHT", 2),
        transcriber_url,
        transcriber_urls,
        allow_source_url_hosts: parse_csv_hosts(&env("ALLOW_SOURCE_URL_HOSTS", "")),
        ayah_align_mode: env("AYAH_ALIGN_MODE", "v2").trim().to_lowercase(),
        ayah_align_v2_assets_dir: PathBuf::from(env("AYAH_ALIGN_V2_ASSETS_DIR", "/data/alignment/v2")),
        ayah_word_time_upgrade: env("AYAH_WORD_TIME_UPGRADE", "smooth"),
        ayah_align_v2_ref_del_cost_cap: env_f64_opt("AYAH_ALIGN_V2_REF_DEL_COST_CAP", Some(0.75)),
        ayah_align_v2_obs_ins_cost_cap: env_f64_opt("AYAH_ALIGN_V2_OBS_INS_COST_CAP", None),
        stream_jwt_secret: env("STREAM_JWT_SECRET", "").trim().to_string(),
        stream_jwt_audience: env("STREAM_JWT_AUDIENCE", "inference_stream").trim().to_string(),
        stream_auth_grace_s: env_usize("STREAM_AUTH_GRACE_S", 10) as u64,
        guess_top_k: env_usize("GUESS_TOP_K", 5),
        guess_per_surah_top_n: env_usize("GUESS_PER_SURAH_TOP_N", 3),
        guess_query_operator: env("GUESS_QUERY_OPERATOR", "OR"),
        guess_rank_mode: env("GUESS_RANK_MODE", "sum_top_score"),
        guess_max_query_tokens: env_usize("GUESS_MAX_QUERY_TOKENS", 24),
        guess_max_hits: env_usize("GUESS_MAX_HITS", 800),
    }
}

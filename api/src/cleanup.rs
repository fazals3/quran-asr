use std::time::{Duration, SystemTime};

use tokio::time::sleep;
use tracing::{info, warn};

use crate::state::{job_dir, AppState};

#[derive(Debug, Default)]
struct CleanupStats {
    scanned: usize,
    deleted: usize,
    skipped_active: usize,
    skipped_young: usize,
    errors: usize,
}

pub async fn run_job_cleanup_once(state: &AppState) {
    let jobs_root = state.settings.data_dir.join("jobs");
    let retention = Duration::from_secs(state.settings.job_retention_s.max(1));
    let max_delete = state.settings.job_cleanup_max_delete_per_run.max(1);
    let now = SystemTime::now();
    let mut stats = CleanupStats::default();

    let mut entries = match tokio::fs::read_dir(&jobs_root).await {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            warn!("job cleanup could not read {}: {e}", jobs_root.display());
            return;
        }
    };

    loop {
        let entry = match entries.next_entry().await {
            Ok(Some(entry)) => entry,
            Ok(None) => break,
            Err(e) => {
                stats.errors += 1;
                warn!("job cleanup failed to read a jobs entry: {e}");
                continue;
            }
        };

        let path = entry.path();
        let job_id = entry.file_name().to_string_lossy().to_string();
        let file_type = match entry.file_type().await {
            Ok(file_type) => file_type,
            Err(e) => {
                stats.errors += 1;
                warn!(job_id = %job_id, "job cleanup failed to stat entry type: {e}");
                continue;
            }
        };
        if !file_type.is_dir() {
            continue;
        }

        stats.scanned += 1;
        if stats.deleted >= max_delete {
            break;
        }

        if let Some(meta) = state.jobs.get(&job_id).map(|m| m.clone()) {
            if !matches!(meta.status.as_str(), "done" | "error") {
                stats.skipped_active += 1;
                continue;
            }

            let finished_or_updated = meta.finished_at.unwrap_or(meta.updated_at);
            if finished_or_updated + retention.as_secs_f64() > crate::state::now_ts() {
                stats.skipped_young += 1;
                continue;
            }
        } else {
            let metadata = match entry.metadata().await {
                Ok(metadata) => metadata,
                Err(e) => {
                    stats.errors += 1;
                    warn!(job_id = %job_id, "job cleanup failed to stat entry metadata: {e}");
                    continue;
                }
            };
            let modified = metadata.modified().unwrap_or(now);
            if now.duration_since(modified).unwrap_or_default() < retention {
                stats.skipped_young += 1;
                continue;
            }
        }

        if let Err(e) = tokio::fs::remove_dir_all(&path).await {
            stats.errors += 1;
            warn!(job_id = %job_id, path = %path.display(), "job cleanup failed to delete job dir: {e}");
            continue;
        }

        state.jobs.remove(&job_id);
        stats.deleted += 1;
    }

    if stats.deleted > 0 || stats.errors > 0 {
        info!(
            scanned = stats.scanned,
            deleted = stats.deleted,
            skipped_active = stats.skipped_active,
            skipped_young = stats.skipped_young,
            errors = stats.errors,
            root = %jobs_root.display(),
            retention_s = state.settings.job_retention_s,
            "job cleanup completed"
        );
    }
}

pub fn spawn_job_cleanup(state: AppState) {
    if !state.settings.job_cleanup_enabled {
        info!("job cleanup disabled");
        return;
    }

    tokio::spawn(async move {
        if state.settings.job_cleanup_startup {
            run_job_cleanup_once(&state).await;
        }

        let interval_s = state.settings.job_cleanup_interval_s.max(60);
        loop {
            sleep(Duration::from_secs(interval_s)).await;
            run_job_cleanup_once(&state).await;
        }
    });
}

pub async fn remove_job_dir_for_upload_failure(state: &AppState, job_id: &str) {
    let path = job_dir(&state.settings.data_dir, job_id);
    let _ = tokio::fs::remove_dir_all(path).await;
}

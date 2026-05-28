use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct TranscriberPool {
    urls: Arc<Vec<String>>,
    inflight: Arc<Vec<AtomicUsize>>,
    rr: Arc<AtomicUsize>,
}

pub struct TranscriberLease {
    url: String,
    idx: usize,
    inflight: Arc<Vec<AtomicUsize>>,
}

impl TranscriberLease {
    pub fn url(&self) -> &str {
        self.url.as_str()
    }
}

impl Drop for TranscriberLease {
    fn drop(&mut self) {
        if let Some(a) = self.inflight.get(self.idx) {
            a.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl TranscriberPool {
    pub fn new(urls: Vec<String>) -> Self {
        let urls = urls.into_iter().filter(|u| !u.trim().is_empty()).collect::<Vec<_>>();
        let inflight = urls.iter().map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
        Self {
            urls: Arc::new(urls),
            inflight: Arc::new(inflight),
            rr: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn urls(&self) -> &[String] {
        self.urls.as_slice()
    }

    pub fn pick(&self) -> TranscriberLease {
        let n = self.urls.len();
        if n == 0 {
            return TranscriberLease {
                url: "http://transcriber:9000".to_string(),
                idx: 0,
                inflight: Arc::new(Vec::new()),
            };
        }

        // Least-busy pick with rotating tie-break.
        let start = self.rr.fetch_add(1, Ordering::Relaxed) % n;
        let mut best_idx = start;
        let mut best = self
            .inflight
            .get(best_idx)
            .map(|a| a.load(Ordering::Relaxed))
            .unwrap_or(usize::MAX);

        for offset in 1..n {
            let idx = (start + offset) % n;
            let v = self
                .inflight
                .get(idx)
                .map(|a| a.load(Ordering::Relaxed))
                .unwrap_or(usize::MAX);
            if v < best {
                best = v;
                best_idx = idx;
            }
        }

        if let Some(a) = self.inflight.get(best_idx) {
            a.fetch_add(1, Ordering::Relaxed);
        }

        TranscriberLease {
            url: self.urls[best_idx].clone(),
            idx: best_idx,
            inflight: self.inflight.clone(),
        }
    }
}


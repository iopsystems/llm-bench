# Server-side Metrics Scraping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scrape the benchmarked server's Prometheus `/metrics` on the client snapshot interval and write a time-aligned `server_metrics.parquet` (same schema family as llm-perf's client parquet).

**Architecture:** A sidecar tokio task mirroring `src/snapshot.rs`. A `PrometheusConverter` turns scraped Prometheus text into a `metriken-exposition::Snapshot` (ported from `rezolus/src/recorder/prometheus.rs`), which flows through llm-perf's existing `MsgpackToParquet` pipeline. Coordinated by the existing `RUNNING` flag.

**Tech Stack:** Rust, tokio, reqwest (existing), metriken-exposition (existing), `prometheus-parse` (new), `histogram` (new, histogram task only).

**Spec:** `docs/superpowers/specs/2026-06-29-server-metrics-scrape-design.md`
**Port reference (read it):** `/Users/brian/workspace/brayniac/rezolus/src/recorder/prometheus.rs` — the proven converter this ports. It targets `metriken-exposition 0.16.0`; llm-perf pins the **git** version, so the `Counter`/`Gauge`/`Histogram`/`SnapshotV2` construction API MUST be verified against the pinned source (Task 2 Step 1).

---

## File Structure

- **Modify** `Cargo.toml` — add `prometheus-parse` (Task 1), `histogram` (Task 5).
- **Modify** `src/config.rs` — two `MetricsConfig` fields + a server-output helper.
- **Create** `src/server_metrics.rs` — `PrometheusConverter` + `capture_server_metrics`. One responsibility: scrape→convert→write.
- **Modify** `src/lib.rs` — register `pub mod server_metrics;`.
- **Modify** `src/benchmark.rs` — spawn + await the sidecar task.
- **Modify** `examples/config.example.toml` — document the fields.

---

## Task 1: Config fields + `prometheus-parse` dep

**Files:** `Cargo.toml`, `src/config.rs`

- [ ] **Step 1: Add the dep.** In `Cargo.toml` under `[dependencies]` add:
```toml
prometheus-parse = "0.2.5"
```

- [ ] **Step 2: Add the two `MetricsConfig` fields.** In `src/config.rs`, the struct currently is:
```rust
pub struct MetricsConfig {
    pub output: PathBuf,
    #[serde(default = "default_metrics_interval")]
    pub interval: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
}
```
Add two fields (keep `#[serde(deny_unknown_fields)]` on the struct):
```rust
    /// Prometheus /metrics URL of the server under test. None => feature off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_metrics_url: Option<String>,
    /// Output parquet for server metrics. Defaults to `<output-stem>.server.parquet`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_metrics_output: Option<PathBuf>,
```

- [ ] **Step 3: Add a server-output helper + write the failing test.** Add to `src/config.rs` an `impl MetricsConfig` (or extend an existing one):
```rust
impl MetricsConfig {
    /// Resolved server-metrics parquet path: the explicit `server_metrics_output`,
    /// else the client `output` with its extension replaced by `.server.parquet`.
    pub fn resolved_server_output(&self) -> std::path::PathBuf {
        if let Some(p) = &self.server_metrics_output {
            return p.clone();
        }
        self.output.with_extension("server.parquet")
    }
}
```
Add a test module (or extend the existing config tests):
```rust
#[cfg(test)]
mod server_metrics_config_tests {
    use super::*;

    #[test]
    fn derives_default_server_output_from_client_output() {
        let toml = r#"
            output = "run.parquet"
            interval = "1s"
            server_metrics_url = "http://localhost:4242/metrics"
        "#;
        let cfg: MetricsConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.server_metrics_url.as_deref(), Some("http://localhost:4242/metrics"));
        assert_eq!(cfg.resolved_server_output(), std::path::PathBuf::from("run.server.parquet"));
    }

    #[test]
    fn explicit_server_output_overrides_default() {
        let toml = r#"
            output = "run.parquet"
            server_metrics_url = "http://x/metrics"
            server_metrics_output = "srv.parquet"
        "#;
        let cfg: MetricsConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.resolved_server_output(), std::path::PathBuf::from("srv.parquet"));
    }
}
```

- [ ] **Step 4: Run the tests.**
Run: `cargo test -p llm-perf --lib server_metrics_config_tests` (or `cargo test server_metrics_config_tests` if the crate is unnamed; check `name` in Cargo.toml).
Expected: both tests pass.

- [ ] **Step 5: Commit.**
```bash
git add Cargo.toml Cargo.lock src/config.rs
git commit -m "feat(config): server_metrics_url/output fields + prometheus-parse dep"
```

---

## Task 2: `PrometheusConverter` (counters / gauges / untyped / summary + labels + provenance)

**Files:** Create `src/server_metrics.rs`; modify `src/lib.rs`. (Histograms are added in Task 5 — this task skips them.)

- [ ] **Step 1: Verify the metriken-exposition construction API.** Before writing code, confirm the pinned `metriken-exposition` exposes the types rezolus uses. Find the checkout:
```bash
find ~/.cargo/git/checkouts -path "*metriken*/src/lib.rs" 2>/dev/null | head
grep -rn "pub struct Counter\|pub struct Gauge\|pub struct SnapshotV2\|pub enum Snapshot\b" $(find ~/.cargo/git/checkouts -type d -name "metriken-exposition*" 2>/dev/null | head -1)/src 2>/dev/null
```
Confirm `Counter { name: String, value: u64, metadata: HashMap<String,String> }`, `Gauge { name, value: i64, metadata }`, and `Snapshot::V2(SnapshotV2 { systemtime: SystemTime, duration: Duration, metadata, counters: Vec<Counter>, gauges: Vec<Gauge>, histograms: Vec<Histogram> })`. If field names differ in the git version, adapt the code below to match (the TDD test will fail to compile otherwise).

- [ ] **Step 2: Write the failing test.** Create `src/server_metrics.rs` with the converter + this test (the test drives the API):
```rust
//! Scrapes a server's Prometheus /metrics on the snapshot interval and writes a
//! time-aligned server_metrics.parquet via llm-perf's existing MsgpackToParquet
//! pipeline. Converter ported from rezolus/src/recorder/prometheus.rs.

use metriken_exposition::{Counter, Gauge, Snapshot, SnapshotV2};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[derive(Clone, Hash, Eq, PartialEq)]
struct MetricKey {
    name: String,
    labels: Vec<(String, String)>,
}

/// Converts Prometheus text into metriken-exposition Snapshots, with stable
/// numeric metric IDs across scrapes (consistent parquet column identity) and
/// provenance metadata (metric name, labels, source, endpoint).
pub struct PrometheusConverter {
    metric_ids: HashMap<MetricKey, usize>,
    next_id: usize,
    source: String,
    endpoint: String,
}

impl PrometheusConverter {
    pub fn new(source: String, endpoint: String) -> Self {
        Self { metric_ids: HashMap::new(), next_id: 0, source, endpoint }
    }

    fn get_or_assign_id(&mut self, name: &str, labels: &[(String, String)]) -> String {
        let key = MetricKey { name: name.to_string(), labels: labels.to_vec() };
        if let Some(id) = self.metric_ids.get(&key) {
            return id.to_string();
        }
        let id = self.next_id;
        self.next_id += 1;
        self.metric_ids.insert(key, id);
        id.to_string()
    }

    fn build_metadata(&self, name: &str, labels: &[(String, String)]) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("metric".to_string(), name.to_string());
        for (k, v) in labels {
            m.insert(k.clone(), v.clone());
        }
        m.insert("source".to_string(), self.source.clone());
        m.insert("endpoint".to_string(), self.endpoint.clone());
        m
    }

    pub fn convert(&mut self, text: &str) -> Snapshot {
        let sanitized = sanitize_metric_names(text);
        let lines = sanitized.lines().map(|l| Ok(l.to_string()));
        let scrape = match prometheus_parse::Scrape::parse(lines) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("failed to parse prometheus metrics: {e}");
                return empty_snapshot();
            }
        };

        let mut counters = Vec::new();
        let mut gauges = Vec::new();

        for sample in scrape.samples {
            let mut labels: Vec<(String, String)> =
                sample.labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            labels.sort();

            match sample.value {
                prometheus_parse::Value::Counter(v) if v.is_finite() => {
                    let id = self.get_or_assign_id(&sample.metric, &labels);
                    counters.push(Counter { name: id, value: v as u64,
                        metadata: self.build_metadata(&sample.metric, &labels) });
                }
                prometheus_parse::Value::Gauge(v) if v.is_finite() => {
                    let id = self.get_or_assign_id(&sample.metric, &labels);
                    gauges.push(Gauge { name: id, value: v as i64,
                        metadata: self.build_metadata(&sample.metric, &labels) });
                }
                prometheus_parse::Value::Untyped(v) if v.is_finite() => {
                    let id = self.get_or_assign_id(&sample.metric, &labels);
                    let metadata = self.build_metadata(&sample.metric, &labels);
                    // _total/_sum/_count are monotonic by convention -> counters.
                    if sample.metric.ends_with("_total")
                        || sample.metric.ends_with("_sum")
                        || sample.metric.ends_with("_count")
                    {
                        counters.push(Counter { name: id, value: v as u64, metadata });
                    } else {
                        gauges.push(Gauge { name: id, value: v as i64, metadata });
                    }
                }
                prometheus_parse::Value::Summary(ref quantiles) => {
                    for q in quantiles {
                        if !q.count.is_finite() { continue; }
                        let mut ql = labels.clone();
                        ql.push(("quantile".to_string(), q.quantile.to_string()));
                        ql.sort();
                        let id = self.get_or_assign_id(&sample.metric, &ql);
                        gauges.push(Gauge { name: id, value: q.count as i64,
                            metadata: self.build_metadata(&sample.metric, &ql) });
                    }
                }
                // Histograms handled in a later task; skip for now.
                prometheus_parse::Value::Histogram(_) => {}
                _ => {} // non-finite values
            }
        }

        Snapshot::V2(SnapshotV2 {
            systemtime: SystemTime::now(),
            duration: Duration::ZERO,
            metadata: HashMap::new(),
            counters,
            gauges,
            histograms: Vec::new(),
        })
    }
}

/// Replace colons in metric names with underscores (prometheus-parse uses `\w+`
/// for names, which excludes the colons that namespaced exporters like vLLM use).
/// Label values and HELP text are left untouched.
fn sanitize_metric_names(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            if let Some(rest) = trimmed
                .strip_prefix("# HELP ")
                .or(trimmed.strip_prefix("# TYPE "))
            {
                let prefix = &trimmed[..trimmed.len() - rest.len()];
                let name_end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
                result.push_str(prefix);
                result.push_str(&rest[..name_end].replace(':', "_"));
                result.push_str(&rest[name_end..]);
            } else {
                result.push_str(trimmed);
            }
        } else {
            let name_end = trimmed
                .find(|c: char| c == '{' || c.is_whitespace())
                .unwrap_or(trimmed.len());
            result.push_str(&trimmed[..name_end].replace(':', "_"));
            result.push_str(&trimmed[name_end..]);
        }
        result.push('\n');
    }
    result
}

fn empty_snapshot() -> Snapshot {
    Snapshot::V2(SnapshotV2 {
        systemtime: SystemTime::now(),
        duration: Duration::ZERO,
        metadata: HashMap::new(),
        counters: Vec::new(),
        gauges: Vec::new(),
        histograms: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# HELP ferallm_decode_steps_total Decode steps executed.
# TYPE ferallm_decode_steps_total counter
ferallm_decode_steps_total 384
# TYPE ferallm_active_sequences gauge
ferallm_active_sequences 16
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running{model=\"llama\"} 12
";

    fn snap_parts(s: &Snapshot) -> (&Vec<Counter>, &Vec<Gauge>) {
        match s {
            Snapshot::V2(v2) => (&v2.counters, &v2.gauges),
            _ => panic!("expected V2 snapshot"),
        }
    }

    #[test]
    fn converts_counters_gauges_and_labels_with_provenance() {
        let mut c = PrometheusConverter::new("ferallm".into(), "http://x/metrics".into());
        let s = c.convert(SAMPLE);
        let (counters, gauges) = snap_parts(&s);
        // one counter (decode_steps), two gauges (active_sequences + vllm running)
        assert_eq!(counters.len(), 1);
        assert_eq!(counters[0].value, 384);
        assert_eq!(counters[0].metadata.get("metric").unwrap(), "ferallm_decode_steps_total");
        assert_eq!(counters[0].metadata.get("source").unwrap(), "ferallm");
        assert_eq!(gauges.len(), 2);
        // the labeled series carries its label in metadata
        let labeled = gauges.iter().find(|g| g.metadata.get("metric").unwrap() == "vllm_num_requests_running").unwrap();
        assert_eq!(labeled.value, 12);
        assert_eq!(labeled.metadata.get("model").unwrap(), "llama");
    }

    #[test]
    fn metric_ids_are_stable_across_scrapes() {
        let mut c = PrometheusConverter::new("s".into(), "e".into());
        let id1 = {
            let s = c.convert(SAMPLE);
            snap_parts(&s).0[0].name.clone()
        };
        let id2 = {
            let s = c.convert(SAMPLE);
            snap_parts(&s).0[0].name.clone()
        };
        assert_eq!(id1, id2, "same (name,labels) must keep the same id across scrapes");
    }
}
```

- [ ] **Step 3: Register the module.** In `src/lib.rs` add `pub mod server_metrics;` (alphabetically, between `pub mod saturation;` and `pub mod snapshot;`).

- [ ] **Step 4: Run the tests.**
Run: `cargo test -p llm-perf --lib server_metrics::tests`
Expected: both tests pass. If compilation fails on metriken-exposition field names, adapt to the pinned API confirmed in Step 1.

- [ ] **Step 5: Commit.**
```bash
git add src/server_metrics.rs src/lib.rs
git commit -m "feat(server-metrics): PrometheusConverter (counters/gauges/labels)"
```

---

## Task 3: `capture_server_metrics` sidecar + benchmark wiring

**Files:** `src/server_metrics.rs`, `src/benchmark.rs`

- [ ] **Step 1: Add the sidecar function.** Append to `src/server_metrics.rs` (mirrors `src/snapshot.rs::capture_snapshots`):
```rust
use crate::config::Config;
use crate::metrics::RUNNING;
use anyhow::Result;
use chrono::{Timelike, Utc};
use metriken_exposition::{MsgpackToParquet, ParquetHistogramType, ParquetOptions};
use std::sync::atomic::Ordering;
use tempfile::NamedTempFile;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::time::{Instant, interval_at, timeout};

/// Scrapes `server_metrics_url` each interval and writes a parquet of server-side
/// metrics, time-aligned (same wall-clock grid) with the client snapshot parquet.
pub async fn capture_server_metrics(config: Config) -> Result<()> {
    let mc = match config.metrics.as_ref() {
        Some(cfg) if cfg.server_metrics_url.is_some() => cfg,
        _ => return Ok(()),
    };
    let url = mc.server_metrics_url.clone().unwrap();
    let output = mc.resolved_server_output();
    let interval_duration = humantime::parse_duration(&mc.interval)?;

    let source = reqwest::Url::parse(&url)
        .ok()
        .and_then(|u| u.host_str().map(|h| match u.port() {
            Some(p) => format!("{h}:{p}"),
            None => h.to_string(),
        }))
        .unwrap_or_else(|| "server".to_string());
    let mut converter = PrometheusConverter::new(source, url.clone());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let temp_file = NamedTempFile::new()?;
    let file = File::from_std(temp_file.reopen()?);
    let mut writer = BufWriter::new(file);

    // Same second-aligned start as snapshot.rs, so ticks match the client grid.
    let start = Instant::now() - Duration::from_nanos(Utc::now().nanosecond() as u64)
        + Duration::from_secs(1);
    let mut interval = interval_at(start, interval_duration);

    log::info!("Scraping server metrics from {url} into {output:?} every {interval_duration:?}");

    let mut wrote_any = false;
    while RUNNING.load(Ordering::Relaxed) {
        if timeout(Duration::from_secs(1), interval.tick()).await.is_err() {
            continue;
        }
        let text = match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => match resp.text().await {
                Ok(t) => t,
                Err(e) => { log::warn!("server /metrics read failed: {e}"); continue; }
            },
            Ok(resp) => { log::warn!("server /metrics returned {}", resp.status()); continue; }
            Err(e) => { log::warn!("server /metrics scrape failed: {e}"); continue; }
        };
        let snapshot = converter.convert(&text);
        let buf = Snapshot::to_msgpack(&snapshot).expect("serialize server snapshot");
        if let Err(e) = writer.write_all(&buf).await {
            log::error!("error writing server metrics snapshot: {e}");
            break;
        }
        wrote_any = true;
    }

    writer.flush().await?;
    drop(writer);

    if !wrote_any {
        log::warn!("no server metrics captured (scrapes all failed?) — skipping {output:?}");
        return Ok(());
    }

    let mut opts = ParquetOptions::new().histogram_type(ParquetHistogramType::Standard);
    if let Some(bs) = mc.batch_size {
        opts = opts.max_batch_size(bs);
    }
    MsgpackToParquet::with_options(opts).convert_file_path(temp_file.path(), &output)?;
    log::info!("Wrote server metrics to {output:?} (parquet)");
    Ok(())
}
```
(Add `Snapshot::to_msgpack` is already in scope via the `metriken_exposition::Snapshot` import at the top of the file from Task 2. If not, add `Snapshot` to the import.)

- [ ] **Step 2: Wire it into the run loop.** In `src/benchmark.rs`, right after the existing snapshot-task spawn (`let snapshot_handle = if self.config.metrics.is_some() { ... }`), add a sibling:
```rust
        // Spawn server-metrics scrape task if a server /metrics URL is configured
        let server_metrics_handle = if self
            .config
            .metrics
            .as_ref()
            .map(|m| m.server_metrics_url.is_some())
            .unwrap_or(false)
        {
            let config = self.config.clone();
            Some(tokio::spawn(async move {
                if let Err(e) = crate::server_metrics::capture_server_metrics(config).await {
                    log::error!("Server metrics capture error: {}", e);
                }
            }))
        } else {
            None
        };
```
Then, next to the existing `if let Some(handle) = snapshot_handle { let _ = handle.await; }` (after `RUNNING.store(false, ...)`), add:
```rust
        if let Some(handle) = server_metrics_handle {
            let _ = handle.await;
        }
```

- [ ] **Step 3: Integration test — round trip against a local mock.** Create `tests/server_metrics_scrape.rs`:
```rust
//! Integration: capture_server_metrics scrapes a local mock /metrics and writes
//! a non-empty parquet. (Content correctness is covered by the converter unit
//! tests; this proves the scrape->msgpack->parquet pipeline + lifecycle.)

use std::sync::atomic::Ordering;
use std::time::Duration;

#[tokio::test]
async fn scrapes_mock_and_writes_parquet() {
    // 1) Local mock server returning a fixed /metrics body.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await { Ok(x) => x, Err(_) => break };
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let body = "# TYPE ferallm_decode_steps_total counter\nferallm_decode_steps_total 7\n# TYPE ferallm_active_sequences gauge\nferallm_active_sequences 3\n";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(), body);
                let _ = sock.write_all(resp.as_bytes()).await;
            });
        }
    });

    // 2) Config pointing at the mock; short interval; temp output.
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run.server.parquet");
    let toml = format!(
        "[endpoint]\nbase_url = \"http://unused/v1\"\nmodel = \"m\"\n\
         [load]\nconcurrent_requests = 1\ntotal_requests = 1\n\
         [input]\nfile = \"{}\"\n\
         [metrics]\noutput = \"{}\"\ninterval = \"500ms\"\n\
         server_metrics_url = \"http://{}/metrics\"\nserver_metrics_output = \"{}\"\n",
        // a 1-line prompts file so config validation passes
        {
            let pf = dir.path().join("p.jsonl");
            std::fs::write(&pf, "{\"prompt\":\"hi\",\"max_tokens\":1}\n").unwrap();
            pf.display()
        },
        dir.path().join("run.parquet").display(),
        addr,
        out.display(),
    );
    let cfg: llm_perf::config::Config = toml::from_str(&toml).unwrap();

    // 3) Run the scraper for ~2 ticks, then stop via RUNNING.
    llm_perf::metrics::RUNNING.store(true, Ordering::Relaxed);
    let h = tokio::spawn(llm_perf::server_metrics::capture_server_metrics(cfg));
    tokio::time::sleep(Duration::from_millis(1300)).await;
    llm_perf::metrics::RUNNING.store(false, Ordering::Relaxed);
    h.await.unwrap().unwrap();

    // 4) Parquet exists and is non-empty.
    let meta = std::fs::metadata(&out).expect("server parquet written");
    assert!(meta.len() > 0, "server parquet should be non-empty");
}
```
Notes for the implementer: the crate name in `use llm_perf::...` must match `[package] name` in `Cargo.toml` (it may be `llm-perf` → crate path `llm_perf`). Confirm `pub mod metrics;` exposes `RUNNING` (it does — `crate::metrics::RUNNING`). If `Config`/the input file require other mandatory fields to deserialize+validate, add the minimum to make `toml::from_str` succeed (check `src/config.rs` `validate()` and the `[input]`/`[endpoint]` required fields; adjust the TOML accordingly). If a raw-socket mock is awkward, an equivalent `axum`/`hyper` oneshot server is acceptable as long as it returns the fixed body.

- [ ] **Step 4: Run tests.**
Run: `cargo test -p llm-perf --test server_metrics_scrape`
Expected: `scrapes_mock_and_writes_parquet` passes (non-empty parquet written).
Also run `cargo test -p llm-perf` to confirm nothing else broke.

- [ ] **Step 5: Commit.**
```bash
git add src/server_metrics.rs src/benchmark.rs tests/server_metrics_scrape.rs
git commit -m "feat(server-metrics): scrape sidecar task + benchmark wiring"
```

---

## Task 4: Docs + example config

**Files:** `examples/config.example.toml`

- [ ] **Step 1: Document the fields.** In `examples/config.example.toml`, in the `[metrics]` section area, add (commented, matching the file's style):
```toml
# Server-side Prometheus metrics scraping (optional).
# Scrapes the server-under-test's /metrics on the same interval as the client
# snapshots and writes a time-aligned parquet (same schema family, join on the
# wall-clock timestamp column).
# server_metrics_url = "http://localhost:4242/metrics"
# server_metrics_output = "run.server.parquet"  # default: <output-stem>.server.parquet
```

- [ ] **Step 2: Commit.**
```bash
git add examples/config.example.toml
git commit -m "docs(config): document server_metrics_url/output"
```

---

## Task 5: Histogram support (general parser completion) — vLLM/TGI latency histograms

**Files:** `Cargo.toml`, `src/server_metrics.rs`

Ferallm's `/metrics` has no histograms; this task completes general support for servers that do (vLLM/TGI latency). It is separable — if the `metriken-exposition` `Histogram` API or the `histogram`-crate version proves incompatible with the pinned `metriken-exposition`, it may be deferred without affecting Tasks 1–3.

- [ ] **Step 1: Pin the `histogram` crate to match metriken-exposition.** Determine the `histogram` version `metriken-exposition` expects (its `Histogram.value` type). Inspect the pinned crate:
```bash
grep -rn "histogram" $(find ~/.cargo/git/checkouts -type d -name "metriken-exposition*" | head -1)/Cargo.toml
```
Add the matching version to llm-perf `Cargo.toml` `[dependencies]`, e.g.:
```toml
histogram = "<version metriken-exposition uses>"
```

- [ ] **Step 2: Write the failing test.** Add to the `tests` module in `src/server_metrics.rs`:
```rust
    const HIST_SAMPLE: &str = "\
# TYPE vllm_ttft_seconds histogram
vllm_ttft_seconds_bucket{le=\"0.1\"} 1
vllm_ttft_seconds_bucket{le=\"0.5\"} 3
vllm_ttft_seconds_bucket{le=\"+Inf\"} 4
vllm_ttft_seconds_sum 0.8
vllm_ttft_seconds_count 4
";

    #[test]
    fn converts_histogram_to_snapshot_histogram() {
        let mut c = PrometheusConverter::new("vllm".into(), "e".into());
        let s = c.convert(HIST_SAMPLE);
        match s {
            Snapshot::V2(v2) => {
                assert_eq!(v2.histograms.len(), 1, "one histogram series");
                assert_eq!(v2.histograms[0].metadata.get("metric").unwrap(), "vllm_ttft_seconds");
            }
            _ => panic!("expected V2"),
        }
    }
```

- [ ] **Step 3: Implement the histogram branch.** Replace the `prometheus_parse::Value::Histogram(_) => {}` arm in `convert()` with the real conversion, and add the helper functions. Port them verbatim from `/Users/brian/workspace/brayniac/rezolus/src/recorder/prometheus.rs` (functions `convert_histogram` and `compute_generic_scale`, lines ~206–294), adjusting only: import `Histogram as SnapshotHistogram` from `metriken_exposition`, and have `build_metadata`-style provenance match this file's `self.source`/`self.endpoint` (Strings, not Options). The arm:
```rust
                prometheus_parse::Value::Histogram(ref buckets) => {
                    let labels_for_meta = labels.clone();
                    if let Some((h, metadata)) = convert_histogram(
                        buckets, &sample.metric, &labels_for_meta, &self.source, &self.endpoint,
                    ) {
                        let id = self.get_or_assign_id(&sample.metric, &labels);
                        histograms.push(metriken_exposition::Histogram { name: id, value: h, metadata });
                    }
                }
```
Declare `let mut histograms = Vec::new();` next to `counters`/`gauges`, and put `histograms` into the `SnapshotV2 { ..., histograms }` (replacing the empty `Vec::new()`). Paste `convert_histogram` (signature taking `source: &str, endpoint: &str`) and `compute_generic_scale` from the reference, with `metadata.insert("source"...)`/`("endpoint"...)` unconditional.

- [ ] **Step 4: Run tests.**
Run: `cargo test -p llm-perf --lib server_metrics::tests`
Expected: all converter tests pass including `converts_histogram_to_snapshot_histogram`. If the `metriken-exposition::Histogram` field types differ from rezolus's (`value: histogram::Histogram`), adapt; if irreconcilable with the pinned version, STOP and report — this task can be deferred.

- [ ] **Step 5: Commit.**
```bash
git add Cargo.toml Cargo.lock src/server_metrics.rs
git commit -m "feat(server-metrics): histogram + summary conversion (vLLM/TGI)"
```

---

## Task 6: Lint, full build, e2e validation

**Files:** none (verification)

- [ ] **Step 1: Format + clippy + tests.**
Run: `cargo fmt --all && cargo fmt --check`
Run: `cargo clippy --all-targets -- -D warnings`
Run: `cargo test -p llm-perf`
Expected: all clean / green.

- [ ] **Step 2: e2e against live ferallm (manual).** Start a `ferallm serve` (with its `/metrics` endpoint), then run a short llm-perf load with a config containing `server_metrics_url = "http://127.0.0.1:4242/metrics"`:
```bash
llm-perf bench <config-with-server_metrics_url-and-metrics-output>.toml
```
Expected: after the run, the `*.server.parquet` exists and is non-empty; spot-check it carries `ferallm_decode_steps_total`/`prefill`/`pad` series (e.g. read it with DuckDB/pandas) with wall-clock timestamps overlapping the client parquet's. This is the loop-closing demonstration of one run yielding aligned client + server metrics.

- [ ] **Step 3: Finish the branch.** Use superpowers:finishing-a-development-branch.

---

## Notes for the implementer
- **Port reference:** `/Users/brian/workspace/brayniac/rezolus/src/recorder/prometheus.rs` is the proven source for the converter (incl. the histogram helpers in Task 5). Read it.
- **API risk:** rezolus targets `metriken-exposition 0.16.0`; llm-perf pins the **git** version. Verify `Counter`/`Gauge`/`Histogram`/`SnapshotV2` fields against the pinned source (Task 2 Step 1) before assuming the pasted code compiles.
- **Crate path:** `use llm_perf::...` in `tests/` — confirm the package name in `Cargo.toml` (`llm-perf` → path `llm_perf`).
- **YAGNI:** histograms (Task 5) exist for vLLM/TGI; ferallm (the current consumer) has none — Tasks 1–3 deliver full ferallm value standalone.

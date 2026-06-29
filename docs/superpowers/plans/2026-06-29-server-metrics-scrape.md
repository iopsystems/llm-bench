# Server-side Metrics Capture (shell out to `rezolus record`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** When a server `/metrics` URL is configured, llm-perf spawns `rezolus record` as a child process bounded to the benchmark, and SIGINTs it at the end so it finalizes a `server.parquet` time-aligned with the client parquet.

**Architecture:** A thin process launcher (`src/server_metrics.rs`) builds the `rezolus record` arg vector, spawns it as a `tokio::process::Child` beside the existing snapshot task, and on load end sends one `SIGINT` (which `rezolus record`'s `ctrlc` handler turns into a clean parquet-finalizing shutdown), then waits with a SIGKILL backstop.

**Tech Stack:** Rust, tokio (process feature), `libc` (new, for `kill`). No parser/parquet code in llm-perf.

**Spec:** `docs/superpowers/specs/2026-06-29-server-metrics-scrape-design.md`
**Prior state:** the config fields `server_metrics_url`/`server_metrics_output` + `resolved_server_output()` already exist (kept from the reverted integrated attempt); `prometheus-parse` was added then and is now unused.

---

## File Structure
- **Modify** `Cargo.toml` — remove unused `prometheus-parse`; add `libc`; ensure tokio has the `process` feature.
- **Modify** `src/config.rs` — add `rezolus_bin: Option<String>` to `MetricsConfig`.
- **Create** `src/server_metrics.rs` — `record_args`, `ServerMetricsRecorder`, `spawn`, `finish`.
- **Modify** `src/lib.rs` — `pub mod server_metrics;`.
- **Modify** `src/benchmark.rs` — spawn the recorder + `finish().await` at load end.
- **Modify** `examples/config.example.toml` — document the fields.

---

## Task 1: Deps + `rezolus_bin` config field

**Files:** `Cargo.toml`, `src/config.rs`

- [ ] **Step 1: Deps.** In `Cargo.toml`: remove the `prometheus-parse = "0.2.5"` line (now unused). Add under `[dependencies]`:
```toml
libc = "0.2"
```
Ensure `tokio` enables the `process` feature. Find the tokio line (`grep -n '^tokio' Cargo.toml`). If its `features` list lacks `"process"`, add it (e.g. `features = ["...", "process"]`). If tokio uses `features = ["full"]`, no change needed.

- [ ] **Step 2: Add the config field.** In `src/config.rs`, `MetricsConfig` currently has `output`, `interval`, `batch_size`, `server_metrics_url`, `server_metrics_output`. Add:
```rust
    /// Binary to invoke for server-metrics capture. Default `"rezolus"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rezolus_bin: Option<String>,
```

- [ ] **Step 3: Write the failing test.** Add to the existing `server_metrics_config_tests` module in `src/config.rs` (or create it if the revert removed it):
```rust
    #[test]
    fn rezolus_bin_defaults_to_none_and_parses() {
        let toml = r#"
            output = "run.parquet"
            server_metrics_url = "http://x/metrics"
        "#;
        let cfg: MetricsConfig = toml::from_str(toml).unwrap();
        assert!(cfg.rezolus_bin.is_none());

        let toml2 = r#"
            output = "run.parquet"
            rezolus_bin = "/opt/rezolus"
        "#;
        let cfg2: MetricsConfig = toml::from_str(toml2).unwrap();
        assert_eq!(cfg2.rezolus_bin.as_deref(), Some("/opt/rezolus"));
    }
```

- [ ] **Step 4: Run + build.**
Run: `cargo test rezolus_bin_defaults_to_none_and_parses` (passes) and `cargo build` (confirms `prometheus-parse` removal didn't break anything — nothing references it after the T2 revert).

- [ ] **Step 5: Commit.**
```bash
cargo fmt
git add Cargo.toml Cargo.lock src/config.rs
git commit -m "feat(config): rezolus_bin field; drop unused prometheus-parse; add libc"
```

---

## Task 2: `ServerMetricsRecorder` launcher

**Files:** Create `src/server_metrics.rs`; modify `src/lib.rs`.

- [ ] **Step 1: Create the module + tests.** Create `src/server_metrics.rs`:
```rust
//! Launches `rezolus record` as a child process to capture the server-under-test's
//! Prometheus /metrics during a benchmark, writing a parquet time-aligned with the
//! client metrics. A thin launcher — rezolus does the scraping/parsing/parquet.

use crate::config::MetricsConfig;
use std::time::Duration;
use tokio::process::{Child, Command};

/// Build the `rezolus record` argument vector (after the binary name).
/// `mc.server_metrics_url` MUST be Some (callers gate on it).
pub fn record_args(mc: &MetricsConfig) -> Vec<String> {
    let url = mc.server_metrics_url.clone().unwrap_or_default();
    let source = reqwest::Url::parse(&url)
        .ok()
        .and_then(|u| u.host_str().map(|h| match u.port() {
            Some(p) => format!("{h}:{p}"),
            None => h.to_string(),
        }))
        .unwrap_or_else(|| "server".to_string());
    let endpoint = format!("{url},source={source},protocol=prometheus");
    let output = mc.resolved_server_output();
    vec![
        "record".to_string(),
        "--endpoint".to_string(),
        endpoint,
        "--interval".to_string(),
        mc.interval.clone(),
        output.to_string_lossy().into_owned(),
    ]
}

/// A running `rezolus record` child.
pub struct ServerMetricsRecorder {
    child: Child,
}

impl ServerMetricsRecorder {
    /// Spawn `rezolus record` for the configured server endpoint. Returns `None`
    /// (with a warning) if the binary can't be started, so the benchmark proceeds
    /// without server metrics.
    pub fn spawn(mc: &MetricsConfig) -> Option<Self> {
        let bin = mc.rezolus_bin.clone().unwrap_or_else(|| "rezolus".to_string());
        let args = record_args(mc);
        match Command::new(&bin).args(&args).spawn() {
            Ok(child) => {
                log::info!("started `{bin} record` for server metrics -> {:?}",
                    mc.resolved_server_output());
                Some(Self { child })
            }
            Err(e) => {
                log::warn!("could not start `{bin} record` for server metrics: {e} \
                    (continuing without server metrics)");
                None
            }
        }
    }

    /// Stop the recorder cleanly: one SIGINT (rezolus finalizes its parquet on the
    /// first SIGINT), then wait up to 30s; SIGKILL as a backstop.
    pub async fn finish(mut self) {
        match self.child.id() {
            Some(pid) => {
                // SAFETY: pid is a live child PID from tokio; SIGINT is well-defined.
                unsafe { libc::kill(pid as libc::pid_t, libc::SIGINT); }
                match tokio::time::timeout(Duration::from_secs(30), self.child.wait()).await {
                    Ok(Ok(status)) => log::info!("rezolus record finished ({status})"),
                    Ok(Err(e)) => log::warn!("rezolus record wait failed: {e}"),
                    Err(_) => {
                        log::warn!("rezolus record didn't exit 30s after SIGINT; killing \
                            (server parquet may be incomplete)");
                        let _ = self.child.kill().await;
                    }
                }
            }
            None => log::warn!("rezolus record already exited before finish()"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn mc(url: &str, out: Option<&str>) -> MetricsConfig {
        MetricsConfig {
            output: PathBuf::from("run.parquet"),
            interval: "1s".to_string(),
            batch_size: None,
            server_metrics_url: Some(url.to_string()),
            server_metrics_output: out.map(PathBuf::from),
            rezolus_bin: None,
        }
    }

    #[test]
    fn record_args_builds_endpoint_interval_output() {
        let args = record_args(&mc("http://localhost:4242/metrics", None));
        assert_eq!(args[0], "record");
        assert_eq!(args[1], "--endpoint");
        assert_eq!(args[2], "http://localhost:4242/metrics,source=localhost:4242,protocol=prometheus");
        assert_eq!(args[3], "--interval");
        assert_eq!(args[4], "1s");
        assert_eq!(args[5], "run.server.parquet"); // derived default
    }

    #[test]
    fn record_args_uses_explicit_output() {
        let args = record_args(&mc("http://h/metrics", Some("srv.parquet")));
        assert_eq!(args.last().unwrap(), "srv.parquet");
    }

    // Lifecycle: a stub "rezolus" that traps SIGINT, writes its last arg (the
    // output path), and exits — proving spawn + SIGINT + clean wait + output.
    #[tokio::test]
    async fn spawn_and_finish_signals_child_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let stub = dir.path().join("fake_rezolus.sh");
        std::fs::write(&stub,
            "#!/bin/sh\nfor a in \"$@\"; do out=\"$a\"; done\n\
             trap 'echo done > \"$out\"; exit 0' INT\n\
             while true; do sleep 0.05; done\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let out = dir.path().join("run.server.parquet");
        let mut config = mc("http://127.0.0.1:9/metrics", Some(out.to_str().unwrap()));
        config.rezolus_bin = Some(stub.to_string_lossy().into_owned());

        let rec = ServerMetricsRecorder::spawn(&config).expect("spawn stub");
        tokio::time::sleep(Duration::from_millis(200)).await;
        rec.finish().await;

        assert!(out.exists(), "stub should have written its output on SIGINT");
        assert_eq!(std::fs::read_to_string(&out).unwrap().trim(), "done");
    }
}
```

- [ ] **Step 2: Register the module.** In `src/lib.rs` add `pub mod server_metrics;` (alphabetically between `saturation` and `snapshot`).

- [ ] **Step 3: Run the tests.**
Run: `cargo test server_metrics`
Expected: `record_args_builds_endpoint_interval_output`, `record_args_uses_explicit_output`, and `spawn_and_finish_signals_child_cleanly` all pass. If `MetricsConfig` has additional fields beyond those in the `mc()` helper, add them to the struct literal (check `src/config.rs`). If `tokio::process` is unavailable, confirm the `process` feature from Task 1.

- [ ] **Step 4: Lint + commit.**
```bash
cargo clippy --all-targets -- -D warnings
cargo fmt
git add src/server_metrics.rs src/lib.rs
git commit -m "feat(server-metrics): rezolus record launcher (spawn + SIGINT finish)"
```

---

## Task 3: Benchmark wiring + docs + e2e

**Files:** `src/benchmark.rs`, `examples/config.example.toml`

- [ ] **Step 1: Wire into the run loop.** In `src/benchmark.rs`, near the existing snapshot-task spawn (`let snapshot_handle = if self.config.metrics.is_some() { ... };`), add a recorder:
```rust
        // Launch `rezolus record` for server-side metrics if a URL is configured.
        let server_recorder = self
            .config
            .metrics
            .as_ref()
            .filter(|m| m.server_metrics_url.is_some())
            .and_then(crate::server_metrics::ServerMetricsRecorder::spawn);
```
Then, after `crate::metrics::RUNNING.store(false, ...)` and BEFORE awaiting `snapshot_handle` (so the recorder finalizes while the client snapshot also wraps up), add:
```rust
        if let Some(rec) = server_recorder {
            rec.finish().await;
        }
```
(`ServerMetricsRecorder::spawn` takes `&MetricsConfig`; `.and_then(...spawn)` passes the `&MetricsConfig` from `filter`. If the borrow checker complains about the closure capturing, bind `let mc = self.config.metrics.as_ref();` and call `ServerMetricsRecorder::spawn(mc)` inside an explicit `if let Some(mc) = ... { if mc.server_metrics_url.is_some() { ServerMetricsRecorder::spawn(mc) } else { None } }`.)

- [ ] **Step 2: Build + full tests.**
Run: `cargo build && cargo test`
Expected: green. (No new test here — Task 2's lifecycle test covers the recorder; this step confirms the wiring compiles and nothing regressed.)

- [ ] **Step 3: Document the fields.** In `examples/config.example.toml`, in the `[metrics]` area, add (commented, matching the file's style):
```toml
# Server-side metrics capture (optional): spawns `rezolus record` to scrape the
# server-under-test's Prometheus /metrics on the same interval and write a
# parquet time-aligned with the client metrics (join on the wall-clock timestamp).
# Requires the `rezolus` binary on PATH (or set rezolus_bin).
# server_metrics_url = "http://localhost:4242/metrics"
# server_metrics_output = "run.server.parquet"  # default: <output-stem>.server.parquet
# rezolus_bin = "rezolus"                          # binary to invoke
```

- [ ] **Step 4: Lint + commit.**
```bash
cargo fmt && cargo fmt --check
cargo clippy --all-targets -- -D warnings
git add src/benchmark.rs examples/config.example.toml
git commit -m "feat(server-metrics): spawn/stop rezolus record around the load + docs"
```

- [ ] **Step 5: e2e (manual).** With a `ferallm serve` running (exposing `/metrics`) and `rezolus` on PATH, run `llm-perf bench` with a config containing `[metrics] output=... server_metrics_url="http://127.0.0.1:4242/metrics"`. Confirm: the run logs "started `rezolus record`" and "rezolus record finished"; `run.server.parquet` exists and is non-empty; spot-check it carries `ferallm_*` series (e.g. with DuckDB) whose timestamps overlap the client `run.parquet`.

- [ ] **Step 6: Finish the branch.** Use superpowers:finishing-a-development-branch.

---

## Notes for the implementer
- `rezolus record` finalizes its parquet on the **first** SIGINT (verified in `rezolus/src/recorder/mod.rs` — `ctrlc` handler flips `STATE` to `TERMINATING`, the loop breaks, then it flushes+converts). Do not SIGKILL first.
- The endpoint spec format is `url[,source=name][,protocol=prometheus]`; CLI is `rezolus record --endpoint <spec> --interval <i> <output>` (confirm with `rezolus record --help` if anything mismatches).
- Unix-only (`libc::kill`); the benchmark box is Unix. Guard the test's chmod with `#[cfg(unix)]` (done above).

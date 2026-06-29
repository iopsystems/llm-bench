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
        .and_then(|u| {
            u.host_str().map(|h| match u.port() {
                Some(p) => format!("{h}:{p}"),
                None => h.to_string(),
            })
        })
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
        let bin = mc
            .rezolus_bin
            .clone()
            .unwrap_or_else(|| "rezolus".to_string());
        let args = record_args(mc);
        match Command::new(&bin).args(&args).spawn() {
            Ok(child) => {
                log::info!(
                    "started `{bin} record` for server metrics -> {:?}",
                    mc.resolved_server_output()
                );
                Some(Self { child })
            }
            Err(e) => {
                log::warn!(
                    "could not start `{bin} record` for server metrics: {e} \
                     (continuing without server metrics)"
                );
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
                unsafe {
                    libc::kill(pid as libc::pid_t, libc::SIGINT);
                }
                match tokio::time::timeout(Duration::from_secs(30), self.child.wait()).await {
                    Ok(Ok(status)) => log::info!("rezolus record finished ({status})"),
                    Ok(Err(e)) => log::warn!("rezolus record wait failed: {e}"),
                    Err(_) => {
                        log::warn!(
                            "rezolus record didn't exit 30s after SIGINT; killing \
                             (server parquet may be incomplete)"
                        );
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
        assert_eq!(
            args[2],
            "http://localhost:4242/metrics,source=localhost:4242,protocol=prometheus"
        );
        assert_eq!(args[3], "--interval");
        assert_eq!(args[4], "1s");
        assert_eq!(args[5], "run.server.parquet");
    }

    #[test]
    fn record_args_uses_explicit_output() {
        let args = record_args(&mc("http://h/metrics", Some("srv.parquet")));
        assert_eq!(args.last().unwrap(), "srv.parquet");
    }

    #[tokio::test]
    async fn spawn_and_finish_signals_child_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let stub = dir.path().join("fake_rezolus.sh");
        std::fs::write(
            &stub,
            "#!/bin/sh\nfor a in \"$@\"; do out=\"$a\"; done\n\
             trap 'echo done > \"$out\"; exit 0' INT\n\
             while true; do sleep 0.05; done\n",
        )
        .unwrap();
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

        assert!(
            out.exists(),
            "stub should have written its output on SIGINT"
        );
        assert_eq!(std::fs::read_to_string(&out).unwrap().trim(), "done");
    }
}

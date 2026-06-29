# Server-side metrics scraping — design

**Date:** 2026-06-29. **Status:** design, pending implementation plan. **Repo:** llm-perf.

## Goal

While benchmarking an OpenAI-compatible server, scrape that server's Prometheus
`/metrics` endpoint on the same interval as llm-perf's existing client-side
metrics snapshots, and write the server-side time-series to a parquet file that
is **time-aligned** (shared wall-clock timestamp grid) with the client parquet.
This lets a run attribute client-observed behavior (TTFT/ITL/throughput) to a
*server-side cause* (occupancy, prefill-admission share, KV pressure, queue
depth, …) instead of inferring it.

Concrete motivating case: the `ferallm-serve` engine exposes a Prometheus
`/metrics` endpoint (occupancy, prefill-step share, pad-efficiency,
cache-exhausted, KV blocks, …). A serving profile found that client TTFT collapse
under load was caused by prefill-admission starvation — visible only in the
server-side metrics. This feature captures those server metrics alongside the
client metrics in one run.

## Context: relationship to `rezolus record`

`rezolus record` already scrapes Prometheus endpoints (via `prometheus_parse`)
and writes parquet, with provenance tagging and `--duration`. This design
deliberately **does not depend on rezolus** — llm-perf is distributed standalone
(deb/rpm) and should capture server metrics self-contained. However, it **borrows
rezolus's proven approach**: convert scraped Prometheus into a
`metriken-exposition` `Snapshot` and run it through the *same*
`MsgpackToParquet` pipeline llm-perf already uses for client snapshots. That
reuse means: no new parquet dependency, and the server parquet is the **same
schema family** as the client parquet (consistent analysis tooling, native
timestamp join).

## Non-goals

- Merging server metrics into the *client* parquet (a separate file, joined on
  timestamp, keeps schemas clean and avoids metric-ID collisions).
- Authentication / TLS to the scrape target (local benchmarking; add later if a
  real need appears).
- A live dashboard or alerting (this writes parquet for offline analysis).
- Multiple scrape endpoints in one run (single `server_metrics_url`; the
  provenance field is forward-compatible with multi-endpoint if needed later).

## Architecture

A sidecar tokio task mirroring the existing client snapshot task, decoupled from
the metriken client registry.

```
benchmark.rs run(): RUNNING=true
  ├─ spawn stats task              (existing)
  ├─ spawn snapshot task           (existing: client metriken → client parquet)
  ├─ spawn capture_server_metrics  (NEW, only if server_metrics_url set)
  ├─ run workers (concurrent / QPS / saturation)
  └─ RUNNING=false → await snapshot_handle AND server_metrics_handle
```

- **`MetricsConfig` (src/config.rs)** gains two optional fields (below). The
  struct is `#[serde(deny_unknown_fields)]`; additive optional fields are safe.
- **New `src/server_metrics.rs`**, two units:
  - `PrometheusConverter` — converts Prometheus text → `metriken-exposition`
    `Snapshot`, tagging each series with metadata (`metric` name, the series
    labels, `source`, `endpoint`) and assigning **stable numeric metric IDs
    across scrapes** within a session (so parquet column identity is consistent).
    Mirrors `rezolus/src/recorder/prometheus.rs` (~150 lines); uses
    `prometheus_parse` + `metriken-exposition` (the latter is already a dep).
  - `pub async fn capture_server_metrics(config: Config) -> Result<()>` — the
    sidecar task, structured exactly like `snapshot.rs::capture_snapshots`.
- **Wiring (src/benchmark.rs ~668):** spawn alongside the snapshot task when
  `server_metrics_url` is set; `await` its handle after `RUNNING.store(false)`.
- **New dependency:** `prometheus-parse` (the crate rezolus uses). No new parquet
  dep — `MsgpackToParquet` from `metriken-exposition` is already used.

## Data flow & timestamp alignment

1. `capture_server_metrics` parses `interval` (`humantime`, as `snapshot.rs`
   does) and computes the **same second-aligned start** as `snapshot.rs` (line
   ~32: `Instant::now() - Duration::from_nanos(Utc::now().nanosecond())`), so
   server ticks land on the same boundaries as client snapshots.
2. Build `PrometheusConverter::with_provenance(source, endpoint)` where `source`
   is derived from the URL host:port and `endpoint` is the full URL.
3. `reqwest::Client` with a 5 s per-scrape timeout (reqwest is already a dep).
4. Loop while `RUNNING`: `timeout(Duration::from_secs(1), interval.tick())`; on a
   tick, GET the URL → body text → `converter.convert(&text)` → `Snapshot` (its
   `systemtime` field is `SystemTime::now()`, wall-clock) → `Snapshot::to_msgpack`
   → append to the msgpack buffer (same buffering as `snapshot.rs`).
5. On exit (`RUNNING=false`): flush the buffer and
   `MsgpackToParquet::with_options(...).convert_file_path(temp, output)`.
6. **Alignment:** both client and server snapshots use wall-clock `SystemTime`
   on the same aligned interval, so they join on nearest timestamp. The client
   parquet already carries the snapshot wall-clock timestamp; the server parquet
   carries its own — a simple equi/nearest join on the timestamp column.

## Config (`MetricsConfig`)

```rust
pub struct MetricsConfig {
    pub output: PathBuf,
    pub interval: String,
    pub batch_size: Option<usize>,
    /// Prometheus /metrics URL of the server under test. None ⇒ feature off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_metrics_url: Option<String>,
    /// Output parquet for server metrics. Defaults to `<output-stem>.server.parquet`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_metrics_output: Option<PathBuf>,
}
```

Validation at config load: if `server_metrics_url` is `Some` but unparseable as a
URL, error; if `Some` and `server_metrics_output` is `None`, derive the default
(`output` stem + `.server.parquet`). An example block is added to
`examples/config.example.toml`.

## Error handling

- A scrape failure (connection refused, timeout, non-200) or a parse failure:
  `log::warn!` and **skip that tick**, then continue — the server may be starting
  or restarting, and a missed sample must not abort the load.
- The sidecar's `tokio::spawn` wraps any returned error in a `log::error!`,
  exactly like the existing snapshot task, so a scraper fault cannot take down
  the benchmark.
- If zero scrapes succeeded by the end: `log::warn!("no server metrics captured")`
  and skip writing the parquet (no empty file).

## Testing

- **Unit — `PrometheusConverter::convert`:** feed a fixed Prometheus text sample
  containing (a) ferallm's label-free counters/gauges, (b) a labeled series, and
  (c) a histogram. Assert the resulting `Snapshot` contains the expected metric
  names, values, labels, and types; assert **stable IDs** across two `convert()`
  calls (same `(name, labels)` ⇒ same id); assert provenance metadata
  (`source`, `endpoint`, `metric`) is attached.
- **Integration — round trip:** start a tiny local `TcpListener`/hyper mock that
  serves a fixed `/metrics` text; run `capture_server_metrics` for ~2 ticks at a
  short interval with the `RUNNING` flag toggled off after; read back the parquet
  and assert the metrics are present with ≥2 distinct wall-clock timestamps. No
  model or real server required.
- **e2e (manual):** run `llm-perf bench` with `server_metrics_url` pointed at a
  live `ferallm serve` during a load; confirm `server_metrics.parquet` carries
  `ferallm_decode_steps_total` / `prefill` / `pad` rows whose wall-clock
  timestamps align to the client parquet (spot-check the join).

## Files

- Modify: `src/config.rs` (two `MetricsConfig` fields + validation/default).
- Create: `src/server_metrics.rs` (`PrometheusConverter`, `capture_server_metrics`).
- Modify: `src/benchmark.rs` (spawn + await the sidecar task).
- Modify: `src/lib.rs` (or `main.rs` module list) — register `mod server_metrics;`.
- Modify: `Cargo.toml` (add `prometheus-parse`).
- Modify: `examples/config.example.toml` (document the two fields).

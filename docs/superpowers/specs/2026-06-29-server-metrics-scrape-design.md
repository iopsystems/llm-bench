# Server-side metrics capture — design (shell out to `rezolus record`)

**Date:** 2026-06-29. **Status:** design (revised — pivoted from an integrated
scraper to shelling out). **Repo:** llm-perf.

## Goal

While benchmarking a server, capture that server's Prometheus `/metrics`
time-series alongside llm-perf's client-side metrics, so a run can attribute
client behavior (TTFT/ITL/throughput) to a server-side cause (occupancy,
prefill-admission share, KV pressure). The two parquets join on a shared
wall-clock timestamp.

## Approach: launch `rezolus record` as a child process

`rezolus record` already scrapes Prometheus endpoints (via `prometheus_parse`)
and writes parquet, with provenance, histogram/label support, stable metric IDs,
and a clean `ctrlc` shutdown that finalizes the parquet. Rather than reimplement
that in llm-perf, when a server `/metrics` URL is configured llm-perf **spawns
`rezolus record` as a child for the benchmark's lifetime** and stops it cleanly
at the end.

**Why this over an integrated scraper:** an in-tree converter would re-derive
rezolus's parser, histogram-bucket conversion, and metriken-exposition coupling
— the bulk of the work — for a strictly less capable result. Shelling out reuses
the mature tool; new llm-perf code is a thin process launcher. Trade-off: a
runtime dependency on the `rezolus` binary (graceful: if it's not found, warn
and run the benchmark without server metrics).

(An earlier revision of this spec designed an integrated scraper. The
`prometheus_parse`-based converter from that attempt was reverted; the config
fields `server_metrics_url`/`server_metrics_output` were kept.)

## Non-goals

- Reimplementing Prometheus parsing / parquet writing in llm-perf (that is
  `rezolus record`'s job).
- Auto-joining the two parquets (they share a wall-clock timestamp; join offline).
- Bundling/installing rezolus (the user provides it on `PATH`, or sets a path).

## Architecture

```
benchmark.rs run(): RUNNING=true
  ├─ spawn stats task            (existing)
  ├─ spawn snapshot task         (existing: client metrics → client parquet)
  ├─ ServerMetricsRecorder::spawn(...)  (NEW, if server_metrics_url set):
  │     rezolus record --endpoint '<url>,source=<host>,protocol=prometheus'
  │                    --interval <metrics.interval> <server_output>
  ├─ run workers
  └─ RUNNING=false → recorder.finish().await  (SIGINT → rezolus finalizes parquet)
                   → await snapshot_handle (existing)
```

- **`MetricsConfig` (src/config.rs)** — keeps `server_metrics_url: Option<String>`
  and `server_metrics_output: Option<PathBuf>` (+ `resolved_server_output()`, all
  already present); adds `rezolus_bin: Option<String>` (default `"rezolus"`).
- **New `src/server_metrics.rs`** — a thin launcher (NOT a parser):
  - `fn record_args(mc: &MetricsConfig) -> Vec<String>` — a **pure** function that
    builds the `rezolus record` argument vector (endpoint spec, `--interval`,
    output). Unit-tested without spawning.
  - `struct ServerMetricsRecorder` holding the `tokio::process::Child`.
  - `fn spawn(mc, bin) -> Option<ServerMetricsRecorder>` — spawns the child; on
    spawn failure (binary missing) logs `warn!` and returns `None`.
  - `async fn finish(self)` — send `SIGINT` to the child (`libc::kill`), then
    `child.wait()` with a timeout fallback to `child.kill()`.
- **Wiring (src/benchmark.rs):** create the recorder beside the snapshot-task
  spawn; call `finish().await` after `RUNNING=false`, before awaiting the snapshot
  handle.
- **New dep:** `libc` (for `kill(pid, SIGINT)`). No `prometheus-parse` /
  parquet code in llm-perf.

## `rezolus record` invocation (verified against live ferallm)

CLI: `rezolus record <URL> <OUTPUT> --interval <i>`. llm-perf builds the **plain
positional** form: `record <server_metrics_url> <resolved_server_output>
--interval <metrics.interval>`. rezolus **auto-detects the Prometheus protocol**
via its startup probe (confirmed: a 6 s scrape of ferallm `/metrics` produced a
7-snapshot parquet with a `timestamp` column + per-metric columns). No
`--duration` — the load length is variable (e.g. `total_requests` mode); llm-perf
bounds the recorder by signalling it at load end.

**Why plain positional, not annotated** (both discovered to fail in e2e): the
`--endpoint 'url,source=,protocol=prometheus'` form requires the output as the
`[OUTPUT]` positional, which clap fills *after* the `[URL]` positional (whose
`Url` value-parser then rejects the output path); and putting the annotated
string in the positional `[URL]` makes rezolus take the commas literally and the
probe fails. The plain positional URL + probe auto-detection is the working form.

## Lifecycle & shutdown

`rezolus record` installs a `ctrlc` handler: the **first** SIGINT flips its state
to `TERMINATING`, breaks its collection loop, flushes msgpack, and converts to
parquet (verified in `rezolus/src/recorder/mod.rs`). So:
1. `spawn` launches the child at load start.
2. `finish` sends one `SIGINT` (`libc::kill(child.id(), SIGINT)`), then
   `tokio::time::timeout(30s, child.wait())`. On timeout, `child.kill()` (SIGKILL)
   as a backstop and `warn!` that the server parquet may be incomplete.

## Error handling

- `rezolus` not on `PATH` / spawn fails: `warn!` once and return `None` — the
  benchmark runs normally without server metrics.
- `child.id()` is `None` (already exited): `warn!`, skip the signal.
- SIGINT/wait timeout: SIGKILL backstop + `warn!`.
- None of these abort the load.

## Testing

- **Unit — `record_args`:** assert the built arg vector for a sample
  `MetricsConfig` (endpoint spec with `source=` host + `protocol=prometheus`,
  `--interval`, the resolved output path). Pure, no process spawned.
- **Integration — lifecycle:** point `rezolus_bin` at a **stub script** (written
  by the test) that traps SIGINT, writes a marker file, and exits 0. Run
  `spawn` → sleep briefly → `finish().await`; assert the marker file exists (the
  stub received SIGINT and shut down cleanly) and the child was reaped. This
  tests the spawn/signal/wait machinery without depending on rezolus.
- **e2e (manual):** real `rezolus record` against a live `ferallm serve` during
  an `llm-perf bench` run with `server_metrics_url` set; confirm
  `<output>.server.parquet` is written and carries `ferallm_*` series with
  timestamps overlapping the client parquet.

## Files

- Modify: `src/config.rs` (add `rezolus_bin` field; `server_metrics_url/output`
  already present).
- Create: `src/server_metrics.rs` (`record_args`, `ServerMetricsRecorder`,
  `spawn`, `finish`).
- Modify: `src/lib.rs` (`pub mod server_metrics;`).
- Modify: `src/benchmark.rs` (spawn + finish wiring).
- Modify: `Cargo.toml` (add `libc`; the `prometheus-parse` dep added earlier may
  be removed since the integrated parser was reverted).
- Modify: `examples/config.example.toml` (document the fields).

//! Replay of public LLM inference traces against an OpenAI-compatible server.
//!
//! Production LLM serving research is driven by a handful of openly published
//! *inference traces* — request logs that record, for every request, its arrival
//! time and its input/output token counts (and, for Mooncake, its prefix-cache
//! structure). None of them ship prompt text (they are metadata-only for privacy
//! reasons), so replaying a trace means synthesizing a prompt of the recorded
//! input length and asking the server to generate the recorded output length,
//! dispatched on the recorded arrival schedule.
//!
//! Supported formats (auto-detected from the file, or forced via config):
//!
//! * **Azure LLM Inference Trace** (2023 / 2024, a.k.a. the Splitwise & DynamoLLM
//!   traces) — CSV `TIMESTAMP,ContextTokens,GeneratedTokens`. `TIMESTAMP` is an
//!   absolute wall-clock time with 100 ns ticks; inter-arrival times are derived
//!   as deltas. The 2024 week-long conversation trace is the best default for
//!   realistic diurnal replay.
//! * **BurstGPT** — CSV `Timestamp,[Session ID,Elapsed time,]Model,Request tokens,
//!   Response tokens,Total tokens,Log Type`. `Timestamp` is relative seconds.
//!   Columns are located by header name so both the 6- and 8-column releases parse.
//! * **Mooncake** — JSONL `{timestamp, input_length, output_length, hash_ids}`.
//!   `timestamp` is relative milliseconds; `hash_ids` are 512-token cumulative
//!   prefix-block IDs, so requests sharing IDs share a reusable KV-cache prefix.
//!   This is the only public trace with real prefix-cache structure.
//! * **Generic JSONL** — llm-perf's own lowest-common-denominator format, one JSON
//!   object per line with `arrival_ms` (or `timestamp`), `input_tokens` (or
//!   `input_length`) and `output_tokens` (or `output_length`).

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use crate::benchmark::{Prompt, Workload};
use crate::synthetic::generate_fixed_text;
use crate::tokenizer::Tokenizer;

/// A cumulative prefix-cache block is 512 tokens in the Mooncake trace.
const MOONCAKE_BLOCK_TOKENS: usize = 512;

/// A single request extracted from an inference trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEntry {
    /// Arrival time relative to the first request in the trace.
    pub arrival: Duration,
    /// Input / context / prompt token count.
    pub input_tokens: usize,
    /// Output / generated token count (replayed as the request's `max_tokens`).
    pub output_tokens: usize,
    /// Prefix-cache block IDs (Mooncake only). Requests that share leading IDs
    /// share a reusable KV-cache prefix; empty for traces without this structure.
    pub hash_ids: Vec<u64>,
}

/// Trace file formats understood by the replay engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceFormat {
    /// Auto-detect from the filename/extension and a header/first-line sniff.
    Auto,
    /// Azure LLM Inference Trace (2023 & 2024): `TIMESTAMP,ContextTokens,GeneratedTokens`.
    Azure,
    /// BurstGPT: header-keyed CSV with `Request tokens` / `Response tokens`.
    BurstGpt,
    /// Mooncake JSONL: `timestamp` (ms), `input_length`, `output_length`, `hash_ids`.
    Mooncake,
    /// Generic llm-perf JSONL: `arrival_ms`/`timestamp`, `input_tokens`, `output_tokens`.
    Jsonl,
}

impl FromStr for TraceFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "auto" => Ok(TraceFormat::Auto),
            "azure" => Ok(TraceFormat::Azure),
            "burstgpt" | "burst-gpt" | "burst" => Ok(TraceFormat::BurstGpt),
            "mooncake" => Ok(TraceFormat::Mooncake),
            "jsonl" | "generic" => Ok(TraceFormat::Jsonl),
            other => anyhow::bail!(
                "unknown trace format '{other}' (expected: auto, azure, burstgpt, mooncake, jsonl)"
            ),
        }
    }
}

/// Filters and limits applied while parsing a trace.
#[derive(Debug, Clone, Default)]
pub struct TraceOptions {
    /// Skip this many leading entries (after sorting by arrival).
    pub skip: usize,
    /// Keep at most this many entries (after `skip`).
    pub max_requests: Option<usize>,
    /// BurstGPT only: keep rows whose `Model` matches (case-insensitive), e.g. "GPT-4".
    pub model_filter: Option<String>,
    /// Drop entries with zero output tokens (failed requests in BurstGPT's fail files).
    pub drop_zero_output: bool,
    /// Clamp input token counts to at most this value (protects against replaying
    /// pathologically large contexts). `None` = no clamp.
    pub max_input_tokens: Option<usize>,
    /// Clamp output token counts to at most this value. `None` = no clamp.
    pub max_output_tokens: Option<usize>,
}

/// A known trace that can be auto-downloaded, with its canonical format and URL.
struct KnownTrace {
    name: &'static str,
    format: TraceFormat,
    url: &'static str,
    description: &'static str,
}

/// Traces we can fetch by name. Azure raw CSVs and the 2024 release assets, plus
/// the three Mooncake variants. BurstGPT is distributed as large release assets;
/// point `trace` at a locally downloaded file for it.
const KNOWN_TRACES: &[KnownTrace] = &[
    KnownTrace {
        name: "azure-conv-2023",
        format: TraceFormat::Azure,
        url: "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv",
        description: "Azure LLM Inference 2023, conversation workload (~19K reqs, ~1h; Splitwise)",
    },
    KnownTrace {
        name: "azure-code-2023",
        format: TraceFormat::Azure,
        url: "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv",
        description: "Azure LLM Inference 2023, code workload (~8.8K reqs, ~1h; Splitwise)",
    },
    KnownTrace {
        name: "azure-conv-2024",
        format: TraceFormat::Azure,
        url: "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv",
        description: "Azure LLM Inference 2024, conversation workload (1 week; DynamoLLM)",
    },
    KnownTrace {
        name: "azure-code-2024",
        format: TraceFormat::Azure,
        url: "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv",
        description: "Azure LLM Inference 2024, code workload (1 week; DynamoLLM)",
    },
    KnownTrace {
        name: "mooncake-conversation",
        format: TraceFormat::Mooncake,
        url: "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl",
        description: "Mooncake conversation trace (~12K reqs, prefix-cache hash_ids; FAST'25)",
    },
    KnownTrace {
        name: "mooncake-toolagent",
        format: TraceFormat::Mooncake,
        url: "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl",
        description: "Mooncake tool-agent trace (~23K reqs, prefix-cache hash_ids; FAST'25)",
    },
    KnownTrace {
        name: "mooncake-synthetic",
        format: TraceFormat::Mooncake,
        url: "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl",
        description: "Mooncake synthetic trace (~4K reqs, prefix-cache hash_ids; FAST'25)",
    },
];

fn known_trace(name: &str) -> Option<&'static KnownTrace> {
    let n = name.to_lowercase();
    KNOWN_TRACES.iter().find(|t| t.name == n)
}

/// One-line descriptions of every auto-downloadable trace, for help/error text.
pub fn known_trace_help() -> String {
    KNOWN_TRACES
        .iter()
        .map(|t| format!("  {:22} — {}", t.name, t.description))
        .collect::<Vec<_>>()
        .join("\n")
}

fn cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let dir = PathBuf::from(home)
        .join(".cache")
        .join("llm-perf")
        .join("traces");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Resolve a `trace` value to a local file path and its format.
///
/// If `spec` is an existing path it is used directly (format auto-detected unless
/// overridden). Otherwise `spec` is looked up in [`KNOWN_TRACES`] and downloaded to
/// the on-disk cache on first use; the known trace's canonical format wins unless
/// the caller forced one.
pub async fn resolve_trace(spec: &str, format: TraceFormat) -> Result<(PathBuf, TraceFormat)> {
    let path = Path::new(spec);
    if path.exists() {
        let fmt = if format == TraceFormat::Auto {
            detect_format(path)?
        } else {
            format
        };
        return Ok((path.to_path_buf(), fmt));
    }

    if let Some(known) = known_trace(spec) {
        let cache = cache_dir()?;
        let filename = known
            .url
            .rsplit('/')
            .next()
            .unwrap_or(known.name)
            .to_string();
        let cached = cache.join(&filename);
        if !cached.exists() {
            download_to(known.url, &cached)
                .await
                .with_context(|| format!("downloading known trace '{}'", known.name))?;
        } else {
            log::info!("Using cached trace: {}", cached.display());
        }
        let fmt = if format == TraceFormat::Auto {
            known.format
        } else {
            format
        };
        return Ok((cached, fmt));
    }

    anyhow::bail!(
        "trace '{}' not found on disk and is not a known trace.\n\nKnown traces (auto-downloaded):\n{}\n\n\
         For BurstGPT, download a CSV from https://github.com/HPMLL/BurstGPT/releases and point `trace` at it.",
        spec,
        known_trace_help()
    );
}

async fn download_to(url: &str, dest: &Path) -> Result<()> {
    log::info!("Downloading trace from {url}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("GET {url} returned HTTP {}", resp.status());
    }
    let bytes = resp.bytes().await?;
    // Write via a temp file then rename so an interrupted download never leaves a
    // truncated file that looks complete to the next run.
    let tmp = dest.with_extension("partial");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, dest)?;
    log::info!(
        "Downloaded {:.1} MB to {}",
        bytes.len() as f64 / 1_048_576.0,
        dest.display()
    );
    Ok(())
}

/// Detect a trace format from the filename and a peek at the first line.
fn detect_format(path: &Path) -> Result<TraceFormat> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if name.contains("azurellminference") || name.contains("azure") {
        return Ok(TraceFormat::Azure);
    }
    if name.contains("burstgpt") || name.contains("burst") {
        return Ok(TraceFormat::BurstGpt);
    }
    if name.contains("mooncake") {
        return Ok(TraceFormat::Mooncake);
    }

    // Sniff the first non-empty line.
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading trace {}", path.display()))?;
    let first = content.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let trimmed = first.trim_start();
    if trimmed.starts_with('{') {
        // JSON object per line — Mooncake if it has hash_ids/input_length, else generic.
        if trimmed.contains("hash_ids") || trimmed.contains("input_length") {
            return Ok(TraceFormat::Mooncake);
        }
        return Ok(TraceFormat::Jsonl);
    }
    // CSV header sniff.
    let header = trimmed.to_lowercase();
    if header.contains("contexttokens") && header.contains("generatedtokens") {
        return Ok(TraceFormat::Azure);
    }
    if header.contains("request tokens") || header.contains("response tokens") {
        return Ok(TraceFormat::BurstGpt);
    }
    anyhow::bail!(
        "could not auto-detect trace format for {}; set replay.format explicitly \
         (azure, burstgpt, mooncake, jsonl)",
        path.display()
    );
}

/// Parse a trace file into a list of [`TraceEntry`], sorted by arrival and with
/// `arrival` normalized so the first request is at t=0.
pub fn parse_trace(
    path: &Path,
    format: TraceFormat,
    opts: &TraceOptions,
) -> Result<Vec<TraceEntry>> {
    let format = if format == TraceFormat::Auto {
        detect_format(path)?
    } else {
        format
    };
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading trace {}", path.display()))?;

    // Raw entries carry an absolute-ish time in nanoseconds; we normalize afterward.
    let mut raw: Vec<RawEntry> = match format {
        TraceFormat::Azure => parse_azure(&content, opts)?,
        TraceFormat::BurstGpt => parse_burstgpt(&content, opts)?,
        TraceFormat::Mooncake => parse_mooncake(&content, opts)?,
        TraceFormat::Jsonl => parse_jsonl(&content, opts)?,
        TraceFormat::Auto => unreachable!("format resolved above"),
    };

    if raw.is_empty() {
        anyhow::bail!("no usable entries parsed from {}", path.display());
    }

    // Sort by time so inter-arrival deltas are non-negative even if the file is
    // slightly out of order, then normalize to a zero-based offset.
    raw.sort_by_key(|e| e.t_nanos);
    let base = raw[0].t_nanos;

    let mut entries: Vec<TraceEntry> = raw
        .into_iter()
        .map(|e| {
            let offset = (e.t_nanos - base).max(0) as u128;
            TraceEntry {
                arrival: Duration::from_nanos(offset.min(u64::MAX as u128) as u64),
                input_tokens: opts
                    .max_input_tokens
                    .map(|m| e.input.min(m))
                    .unwrap_or(e.input),
                output_tokens: opts
                    .max_output_tokens
                    .map(|m| e.output.min(m))
                    .unwrap_or(e.output),
                hash_ids: e.hash_ids,
            }
        })
        .collect();

    // Apply skip/limit after normalization so offsets stay relative to the kept window.
    if opts.skip > 0 {
        if opts.skip >= entries.len() {
            anyhow::bail!(
                "replay.skip ({}) >= trace length ({})",
                opts.skip,
                entries.len()
            );
        }
        entries.drain(0..opts.skip);
        // Re-zero arrivals to the new first entry.
        if let Some(first) = entries.first().map(|e| e.arrival) {
            for e in &mut entries {
                e.arrival = e.arrival.saturating_sub(first);
            }
        }
    }
    if let Some(max) = opts.max_requests {
        entries.truncate(max);
    }

    if entries.is_empty() {
        anyhow::bail!("no entries remain after applying skip/max_requests/filters");
    }
    Ok(entries)
}

/// Intermediate parse result with a nanosecond timestamp, normalized later.
struct RawEntry {
    t_nanos: i128,
    input: usize,
    output: usize,
    hash_ids: Vec<u64>,
}

fn keep(_input: usize, output: usize, opts: &TraceOptions) -> bool {
    !(opts.drop_zero_output && output == 0)
}

/// Split a CSV line on commas. The trace formats we support have no quoted fields
/// or embedded commas (Azure timestamps use spaces, token counts are integers), so
/// a plain split is correct and avoids pulling in a CSV dependency.
fn csv_fields(line: &str) -> Vec<&str> {
    line.split(',').map(|f| f.trim()).collect()
}

fn parse_azure(content: &str, opts: &TraceOptions) -> Result<Vec<RawEntry>> {
    let mut out = Vec::new();
    let mut header_ctx = None;
    let mut header_gen = None;
    for (i, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = csv_fields(line);
        if i == 0 || (header_ctx.is_none() && line.to_lowercase().contains("contexttokens")) {
            // Header row: locate the token columns by name (handles the multimodal
            // superset `TIMESTAMP,NumImages,ContextTokens,GeneratedTokens` too).
            for (idx, f) in fields.iter().enumerate() {
                match f.to_lowercase().as_str() {
                    "contexttokens" => header_ctx = Some(idx),
                    "generatedtokens" => header_gen = Some(idx),
                    _ => {}
                }
            }
            continue;
        }
        let ctx_idx = header_ctx.unwrap_or(1);
        let gen_idx = header_gen.unwrap_or(2);
        if fields.len() <= ctx_idx.max(gen_idx) {
            continue;
        }
        let ts = fields[0];
        let t_nanos = match parse_azure_timestamp(ts) {
            Some(n) => n,
            None => continue,
        };
        let input: usize = fields[ctx_idx].parse().unwrap_or(0);
        let output: usize = fields[gen_idx].parse().unwrap_or(0);
        if !keep(input, output, opts) {
            continue;
        }
        out.push(RawEntry {
            t_nanos,
            input,
            output,
            hash_ids: Vec::new(),
        });
    }
    Ok(out)
}

/// Parse an Azure `TIMESTAMP` such as `2023-11-16 18:15:46.6805900` into
/// nanoseconds since the Unix epoch.
fn parse_azure_timestamp(ts: &str) -> Option<i128> {
    let ts = ts.trim().trim_matches('"');
    // Try with fractional seconds first, then without.
    let naive = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S"))
        .ok()?;
    Some(naive.and_utc().timestamp_nanos_opt()? as i128)
}

fn parse_burstgpt(content: &str, opts: &TraceOptions) -> Result<Vec<RawEntry>> {
    // Locate columns by header name so both the 6-column and 8-column releases work.
    let mut lines = content.lines();
    let header = loop {
        match lines.next() {
            Some(l) if l.trim().is_empty() => continue,
            Some(l) => break l,
            None => anyhow::bail!("BurstGPT trace is empty"),
        }
    };
    let cols: HashMap<String, usize> = csv_fields(header)
        .iter()
        .enumerate()
        .map(|(i, f)| (f.to_lowercase(), i))
        .collect();
    let ts_idx = *cols
        .get("timestamp")
        .context("BurstGPT header missing 'Timestamp' column")?;
    let req_idx = *cols
        .get("request tokens")
        .context("BurstGPT header missing 'Request tokens' column")?;
    let resp_idx = *cols
        .get("response tokens")
        .context("BurstGPT header missing 'Response tokens' column")?;
    let model_idx = cols.get("model").copied();

    let mut out = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = csv_fields(line);
        let max_idx = ts_idx.max(req_idx).max(resp_idx);
        if fields.len() <= max_idx {
            continue;
        }
        if let (Some(want), Some(mi)) = (opts.model_filter.as_ref(), model_idx)
            && mi < fields.len()
            && !fields[mi].eq_ignore_ascii_case(want)
        {
            continue;
        }
        // Timestamp is relative seconds (float or int).
        let secs: f64 = match fields[ts_idx].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let input: usize = fields[req_idx].parse().unwrap_or(0);
        let output: usize = fields[resp_idx].parse().unwrap_or(0);
        if !keep(input, output, opts) {
            continue;
        }
        out.push(RawEntry {
            t_nanos: (secs * 1_000_000_000.0) as i128,
            input,
            output,
            hash_ids: Vec::new(),
        });
    }
    Ok(out)
}

fn parse_mooncake(content: &str, opts: &TraceOptions) -> Result<Vec<RawEntry>> {
    let mut out = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // timestamp is relative milliseconds.
        let ts_ms = v.get("timestamp").and_then(|t| t.as_f64()).unwrap_or(0.0);
        let input = v.get("input_length").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
        let output = v.get("output_length").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
        let hash_ids = v
            .get("hash_ids")
            .and_then(|h| h.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        if !keep(input, output, opts) {
            continue;
        }
        out.push(RawEntry {
            t_nanos: (ts_ms * 1_000_000.0) as i128,
            input,
            output,
            hash_ids,
        });
    }
    Ok(out)
}

fn parse_jsonl(content: &str, opts: &TraceOptions) -> Result<Vec<RawEntry>> {
    let mut out = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // Arrival: prefer explicit ms, then seconds, then a bare timestamp (ms).
        let t_nanos = if let Some(ms) = v.get("arrival_ms").and_then(|t| t.as_f64()) {
            (ms * 1_000_000.0) as i128
        } else if let Some(s) = v.get("arrival_s").and_then(|t| t.as_f64()) {
            (s * 1_000_000_000.0) as i128
        } else if let Some(ms) = v.get("timestamp").and_then(|t| t.as_f64()) {
            (ms * 1_000_000.0) as i128
        } else {
            0
        };
        let input = field_usize(&v, &["input_tokens", "input_length", "prompt_tokens"]);
        let output = field_usize(&v, &["output_tokens", "output_length", "completion_tokens"]);
        let hash_ids = v
            .get("hash_ids")
            .and_then(|h| h.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        if input == 0 && output == 0 {
            continue;
        }
        if !keep(input, output, opts) {
            continue;
        }
        out.push(RawEntry {
            t_nanos,
            input,
            output,
            hash_ids,
        });
    }
    Ok(out)
}

fn field_usize(v: &serde_json::Value, keys: &[&str]) -> usize {
    for k in keys {
        if let Some(n) = v.get(*k).and_then(|x| x.as_u64()) {
            return n as usize;
        }
    }
    0
}

/// Build replayable workloads and their arrival schedule from parsed trace entries.
///
/// Each entry becomes a single-turn [`Workload`] whose prompt is synthesized to the
/// recorded input length and whose `max_tokens` is the recorded output length.
///
/// When an entry carries Mooncake `hash_ids`, the prompt is assembled from
/// per-block text keyed by block ID, so two requests that share leading IDs share a
/// literal prompt prefix and therefore a reusable server-side KV cache — faithfully
/// reproducing the trace's prefix-cache behavior. Entries without `hash_ids` get a
/// unique prompt (seeded per index) so unrelated requests don't collide in the cache.
///
/// Returns `(workloads, arrivals)` with `arrivals[i]` the offset of `workloads[i]`
/// from the start of the replay.
pub fn build_replay_workloads(
    entries: &[TraceEntry],
    tokenizer: Arc<Tokenizer>,
    seed: u64,
) -> (Vec<Workload>, Vec<Duration>) {
    let mut block_cache: HashMap<u64, String> = HashMap::new();
    let mut workloads = Vec::with_capacity(entries.len());
    let mut arrivals = Vec::with_capacity(entries.len());

    for (idx, entry) in entries.iter().enumerate() {
        let input_tokens = entry.input_tokens.max(1);
        let prompt_text = if entry.hash_ids.is_empty() {
            // Unique per-request text; vary the seed by index to avoid accidental
            // cross-request prefix sharing.
            let s = seed.wrapping_add((idx as u64).wrapping_mul(2_654_435_761));
            generate_fixed_text(input_tokens, Arc::clone(&tokenizer), s)
        } else {
            // Assemble from cumulative prefix blocks so shared IDs share a prefix.
            let mut text = String::new();
            for hid in &entry.hash_ids {
                let block = block_cache.entry(*hid).or_insert_with(|| {
                    generate_fixed_text(MOONCAKE_BLOCK_TOKENS, Arc::clone(&tokenizer), *hid)
                });
                text.push_str(block);
                text.push(' ');
            }
            tokenizer.truncate_to_tokens(&text, input_tokens)
        };

        // Force the server to emit the recorded number of tokens (capped at u32).
        let max_tokens = Some(entry.output_tokens.max(1).min(u32::MAX as usize) as u32);
        workloads.push(Workload::SingleTurn(Prompt {
            prompt: prompt_text,
            max_tokens,
        }));
        arrivals.push(entry.arrival);
    }

    (workloads, arrivals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> TraceOptions {
        TraceOptions::default()
    }

    #[test]
    fn format_from_str() {
        assert_eq!("azure".parse::<TraceFormat>().unwrap(), TraceFormat::Azure);
        assert_eq!(
            "burstgpt".parse::<TraceFormat>().unwrap(),
            TraceFormat::BurstGpt
        );
        assert_eq!(
            "mooncake".parse::<TraceFormat>().unwrap(),
            TraceFormat::Mooncake
        );
        assert_eq!("JSONL".parse::<TraceFormat>().unwrap(), TraceFormat::Jsonl);
        assert!("nope".parse::<TraceFormat>().is_err());
    }

    #[test]
    fn azure_timestamp_parses_with_ticks() {
        let a = parse_azure_timestamp("2023-11-16 18:15:46.6805900").unwrap();
        let b = parse_azure_timestamp("2023-11-16 18:15:47.6805900").unwrap();
        assert_eq!(b - a, 1_000_000_000, "one second apart in nanoseconds");
    }

    #[test]
    fn parse_azure_derives_relative_arrivals() {
        let csv = "TIMESTAMP,ContextTokens,GeneratedTokens\n\
                   2023-11-16 18:15:46.0000000,374,44\n\
                   2023-11-16 18:15:47.5000000,100,200\n";
        let path = write_tmp("azure_test.csv", csv);
        let entries = parse_trace(&path, TraceFormat::Auto, &opts()).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].arrival, Duration::ZERO);
        assert_eq!(entries[0].input_tokens, 374);
        assert_eq!(entries[0].output_tokens, 44);
        assert_eq!(entries[1].arrival, Duration::from_millis(1500));
        assert_eq!(entries[1].input_tokens, 100);
        assert_eq!(entries[1].output_tokens, 200);
    }

    #[test]
    fn parse_azure_multimodal_superset_columns() {
        // TIMESTAMP,NumImages,ContextTokens,GeneratedTokens — token columns are
        // located by header name, not position.
        let csv = "TIMESTAMP,NumImages,ContextTokens,GeneratedTokens\n\
                   2024-10-15 00:00:00.0000000,2,900,120\n";
        let path = write_tmp("azure_mm.csv", csv);
        let entries = parse_trace(&path, TraceFormat::Azure, &opts()).unwrap();
        assert_eq!(entries[0].input_tokens, 900);
        assert_eq!(entries[0].output_tokens, 120);
    }

    #[test]
    fn parse_burstgpt_locates_columns_by_header() {
        // 8-column v2.0 layout.
        let csv = "Timestamp,Session ID,Elapsed time,Model,Request tokens,Response tokens,Total tokens,Log Type\n\
                   0,1,2.3,GPT-4,100,50,150,Conversation log\n\
                   5,2,1.1,ChatGPT,200,0,200,API log\n";
        let path = write_tmp("burst.csv", csv);
        let entries = parse_trace(&path, TraceFormat::BurstGpt, &opts()).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].arrival, Duration::ZERO);
        assert_eq!(entries[0].input_tokens, 100);
        assert_eq!(entries[0].output_tokens, 50);
        assert_eq!(entries[1].arrival, Duration::from_secs(5));
    }

    #[test]
    fn burstgpt_model_filter_and_zero_output_drop() {
        let csv = "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n\
                   0,GPT-4,100,50,150,Conversation log\n\
                   1,ChatGPT,200,30,230,API log\n\
                   2,GPT-4,300,0,300,API log\n";
        let path = write_tmp("burst2.csv", csv);
        let o = TraceOptions {
            model_filter: Some("gpt-4".to_string()),
            drop_zero_output: true,
            ..Default::default()
        };
        let entries = parse_trace(&path, TraceFormat::BurstGpt, &o).unwrap();
        // Only the first GPT-4 row survives (the second GPT-4 row has zero output).
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].input_tokens, 100);
    }

    #[test]
    fn parse_mooncake_reads_hash_ids_and_ms() {
        let jsonl = "{\"timestamp\": 0, \"input_length\": 1024, \"output_length\": 100, \"hash_ids\": [0,1]}\n\
                     {\"timestamp\": 2500, \"input_length\": 512, \"output_length\": 50, \"hash_ids\": [0]}\n";
        let path = write_tmp("mooncake.jsonl", jsonl);
        let entries = parse_trace(&path, TraceFormat::Mooncake, &opts()).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].input_tokens, 1024);
        assert_eq!(entries[0].hash_ids, vec![0, 1]);
        assert_eq!(entries[1].arrival, Duration::from_millis(2500));
        assert_eq!(entries[1].hash_ids, vec![0]);
    }

    #[test]
    fn parse_generic_jsonl_flexible_keys() {
        let jsonl = "{\"arrival_ms\": 0, \"input_tokens\": 10, \"output_tokens\": 20}\n\
                     {\"arrival_ms\": 1000, \"input_length\": 30, \"output_length\": 40}\n";
        let path = write_tmp("generic.jsonl", jsonl);
        let entries = parse_trace(&path, TraceFormat::Jsonl, &opts()).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].output_tokens, 20);
        assert_eq!(entries[1].input_tokens, 30);
        assert_eq!(entries[1].arrival, Duration::from_secs(1));
    }

    #[test]
    fn skip_and_max_requests_rewindow_arrivals() {
        let csv = "TIMESTAMP,ContextTokens,GeneratedTokens\n\
                   2023-01-01 00:00:00.0,1,1\n\
                   2023-01-01 00:00:01.0,2,2\n\
                   2023-01-01 00:00:02.0,3,3\n\
                   2023-01-01 00:00:03.0,4,4\n";
        let path = write_tmp("skip.csv", csv);
        let o = TraceOptions {
            skip: 1,
            max_requests: Some(2),
            ..Default::default()
        };
        let entries = parse_trace(&path, TraceFormat::Azure, &o).unwrap();
        assert_eq!(entries.len(), 2);
        // After skipping the first row, the new first row is re-zeroed.
        assert_eq!(entries[0].arrival, Duration::ZERO);
        assert_eq!(entries[0].input_tokens, 2);
        assert_eq!(entries[1].arrival, Duration::from_secs(1));
    }

    #[test]
    fn token_clamps_apply() {
        let jsonl = "{\"arrival_ms\":0,\"input_tokens\":10000,\"output_tokens\":9000}\n";
        let path = write_tmp("clamp.jsonl", jsonl);
        let o = TraceOptions {
            max_input_tokens: Some(2048),
            max_output_tokens: Some(1024),
            ..Default::default()
        };
        let entries = parse_trace(&path, TraceFormat::Jsonl, &o).unwrap();
        assert_eq!(entries[0].input_tokens, 2048);
        assert_eq!(entries[0].output_tokens, 1024);
    }

    #[test]
    fn build_workloads_sets_max_tokens_and_shares_prefixes() {
        let tokenizer = Arc::new(Tokenizer::new("gpt-3.5-turbo").unwrap());
        let entries = vec![
            TraceEntry {
                arrival: Duration::ZERO,
                input_tokens: 700,
                output_tokens: 128,
                hash_ids: vec![0, 1],
            },
            TraceEntry {
                arrival: Duration::from_millis(10),
                input_tokens: 400,
                output_tokens: 64,
                hash_ids: vec![0],
            },
        ];
        let (workloads, arrivals) = build_replay_workloads(&entries, Arc::clone(&tokenizer), 42);
        assert_eq!(workloads.len(), 2);
        assert_eq!(arrivals[1], Duration::from_millis(10));

        let (p0, mt0) = match &workloads[0] {
            Workload::SingleTurn(p) => (p.prompt.clone(), p.max_tokens),
            _ => panic!("expected single-turn"),
        };
        let (p1, mt1) = match &workloads[1] {
            Workload::SingleTurn(p) => (p.prompt.clone(), p.max_tokens),
            _ => panic!("expected single-turn"),
        };
        assert_eq!(mt0, Some(128));
        assert_eq!(mt1, Some(64));
        // Both share block 0, so the shorter prompt is a prefix of the longer one's
        // opening block region — at minimum their first block-worth of text matches.
        let head0: String = p0.chars().take(64).collect();
        let head1: String = p1.chars().take(64).collect();
        assert_eq!(
            head0, head1,
            "shared hash_id 0 should yield a shared prefix"
        );
    }

    #[test]
    fn build_workloads_unique_when_no_hash_ids() {
        let tokenizer = Arc::new(Tokenizer::new("gpt-3.5-turbo").unwrap());
        let entries = vec![
            TraceEntry {
                arrival: Duration::ZERO,
                input_tokens: 128,
                output_tokens: 32,
                hash_ids: vec![],
            },
            TraceEntry {
                arrival: Duration::ZERO,
                input_tokens: 128,
                output_tokens: 32,
                hash_ids: vec![],
            },
        ];
        let (workloads, _) = build_replay_workloads(&entries, tokenizer, 42);
        let p0 = match &workloads[0] {
            Workload::SingleTurn(p) => p.prompt.clone(),
            _ => unreachable!(),
        };
        let p1 = match &workloads[1] {
            Workload::SingleTurn(p) => p.prompt.clone(),
            _ => unreachable!(),
        };
        assert_ne!(p0, p1, "distinct indices should produce distinct prompts");
    }

    fn write_tmp(name: &str, content: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("llm-perf-trace-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }
}

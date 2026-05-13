# llm-calc

A roofline-style performance calculator for dense decoder-only LLM inference.
Answers three questions:

1. **What's the theoretical limit** for a given (GPU, model, quantization, workload)?
2. **How does measured performance compare** to that limit? (Gap analysis is manual today — see the JSON output.)
3. **What changes when I change a knob?** (Live recompute as inputs change.)

Two front-ends share one engine:

- A **static web app** (Svelte 5, Vite-built) for interactive exploration.
- A **CLI** for pipeable JSON, scripts, and `jq` workflows.

## Quick start

```bash
npm install
npm run dev        # web app at http://localhost:5173
npm run cli -- -g h100 -V sxm-80 -m llama-3.3-70b --format table
```

## CLI

The CLI lives at `bin/llm-calc.mjs`. After `npm install` it's also available via
the package's `bin` entry. Default output is one-line JSON for piping; pass
`--format table` for human-readable output.

### Synopsis

```
llm-calc [OPTIONS]              calculate performance metrics (default)
llm-calc list gpus              list GPUs and their variants
llm-calc list models            list models with arch info
llm-calc --help                 show help
```

### Calc flags

| Short | Long              | Default  | Notes                          |
| ----- | ----------------- | -------- | ------------------------------ |
| `-g`  | `--gpu`           | required | GPU id (see `list gpus`)       |
| `-V`  | `--variant`       | first    | GPU variant id                 |
| `-m`  | `--model`         | required | Model id (see `list models`)   |
| `-p`  | `--prompt`        | 2048     | Prompt tokens                  |
| `-o`  | `--output`        | 512      | Output tokens                  |
| `-c`  | `--concurrency`   | 1        | Concurrent requests            |
| `-w`  | `--weights`       | fp16     | Weight dtype                   |
| `-k`  | `--kv`            | fp16     | KV cache dtype                 |
| `-a`  | `--activations`   | fp16     | Activation/compute dtype       |
|       | `--format`        | json     | `json` or `table`              |

Dtypes: `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4`.

### Examples

```bash
# JSON to jq
llm-calc -g h100 -V sxm-80 -m llama-3.3-70b -p 8192 -c 16 | jq .memory.fits

# Table for humans
llm-calc -g mi300x -m mistral-large-2 -c 8 --format table

# Filter the model list
llm-calc list models | grep qwen

# Sweep concurrency
for c in 1 4 16 64; do
  llm-calc -g h100 -V sxm-80 -m qwen3-32b -c $c \
    | jq -r "[$c, .perf.peak.decode.regime, .perf.peak.outputTokenRate] | @tsv"
done
```

### Exit codes

| Code | Meaning                                                |
| ---- | ------------------------------------------------------ |
| 0    | success (including OOM — check `memory.fits` in JSON)  |
| 1    | invalid input (unknown GPU/variant/model/dtype combo)  |
| 64   | usage error (missing required flag, unknown flag)      |

OOM stays exit 0 deliberately so the JSON pipes cleanly to downstream tools.

## Design

### Engine

Pure functions, no DOM. Single entry point: `calculate(input) → CalcResult`.

```ts
type CalcInput = {
  gpu: GpuSpec
  gpuVariantId: string
  model: ModelArch
  quant: { weights, kv, activations: Dtype }
  workload: { promptTokens, outputTokens, concurrency }
}
```

The math is honest roofline analysis:

| Phase   | Time                                                | Regime               |
| ------- | --------------------------------------------------- | -------------------- |
| Prefill | `max(prefill_flops / tflops, prefill_bytes / bw)`   | `compute` or `memory`|
| Decode  | `max(decode_flops/step / tflops, decode_bytes / bw)`| `compute` or `memory`|

Prefill FLOPs include both the MLP term (`2·params·p`) and the attention
quadratic (`2·layers·p²·hidden`) — kept separate so long-context behavior is
visible. KV cache is GQA-aware (`2·layers·num_kv_heads·head_dim·bytes(kv_dtype)`).
Per-component quantization: weights / KV / activations are independent dtype
knobs. Activations dtype selects which `tflops` entry of the GPU operating
point is used.

### Data model

GPUs nest: `GpuSpec → GpuVariant[] → GpuOperatingPoint[]`.

- A **variant** is a physical SKU (H100 SXM-80 vs PCIe-80 vs NVL-188 — different
  silicon, different capacity, often different bandwidth).
- An **operating point** is a runtime configuration of a variant (`peak`, and
  later `achievable`, `power-capped`, etc.). v1 ships `peak` only.

Models store `paramCount` explicitly rather than deriving from shape — saves us
from arch-specific param formulas drifting across families.

The property database lives as typed TS modules in `src/data/`:
- `gpus.ts` — 19 accelerators across NVIDIA Hopper/Ampere/Ada/Blackwell,
  AMD CDNA3, Intel Gaudi 2/3, Google TPU v5p and Trillium, Cerebras WSE-3,
  and Apple Silicon (M3–M5).
- `models.ts` — 11 dense / GQA-class models (Qwen3, Llama 3.x, Gemma 3,
  Mistral, Phi-4).

Adding a GPU or model is a PR that appends to the relevant array.

### What's in v1, what isn't

**In:** dense decoder-only transformers, batching, GQA/MQA/MLA, per-component
quantization, single-GPU, roofline regime classification, derivation steps for
"show the math."

**Out (deferred to v2+):** tensor / pipeline / expert parallelism, multi-GPU
topology and interconnect, disaggregated prefill/decode, MoE, speculative
decoding, prefix caching, `achievable` operating points sourced from
microbenchmarks, library-calibrated efficiency factors.

### Apple Silicon caveats

Unified-memory: capacity is shared with the OS and other workloads, so usable
headroom is materially lower than the figure shown. Bandwidth is system-wide,
shared with the CPU. No FP8/INT8/INT4 tensor acceleration on the GPU — those
dtype keys are omitted from operating points, and selecting them via the CLI
or UI surfaces a clear error. TFLOPS values are best-effort estimates of GPU
shader-core throughput and should be cross-checked before relying on absolute
decode rates.

## Project layout

```
src/
  engine/    # pure-TS math, no DOM
    types.ts dtypes.ts memory.ts roofline.ts prefill.ts decode.ts
    derivation.ts calc.ts index.ts
  data/      # property database (typed TS modules)
    gpus.ts models.ts index.ts
  ui/        # Svelte 5 components, depend on engine + data
    App.svelte InputPanel.svelte MemoryPanel.svelte
    PerfPanel.svelte DerivationDrawer.svelte stores.ts
  cli.ts     # node:util parseArgs, JSON/table output
test/
  fixtures.ts engine/*.test.ts cli.test.ts
bin/
  llm-calc.mjs    # shell wrapper that runs src/cli.ts via tsx
docs/superpowers/
  specs/  plans/  # design spec and implementation plan
```

## Scripts

```
npm run dev          # vite dev server
npm run build        # static build to dist/
npm run preview      # serve dist/
npm run check        # svelte-check (type-check Svelte + TS)
npm test             # vitest
npm run cli -- ...   # run CLI via tsx
```

## Deployment

Static site, deployed to **`calc.inference.systems`** via
**Cloudflare Workers + Static Assets** (not Pages — Workers is CF's
forward path that unifies static asset serving with optional dynamic
handlers).

### Config

- [`wrangler.toml`](wrangler.toml) declares the worker. No handler code —
  just an `[assets]` binding pointing at `./dist`, plus SPA fallback so
  unknown routes serve `index.html`.
- [`package.json`](package.json) has `deploy` (`vite build && wrangler
  deploy`) and `deploy:dry` (build + plan, no upload) scripts.

### First deploy

```bash
cd calc
npx wrangler login                # one-time browser auth to your CF account
npm run deploy:dry                # sanity check the build + upload plan
npm run deploy                    # actual upload
```

After the first deploy, the worker is reachable at
`llm-calc.<account>.workers.dev`. Verify the calculator loads there.

### Custom domain

DNS for `inference.systems` is already on Cloudflare, so:

- Cloudflare dashboard → Workers & Pages → `llm-calc` → Settings → Domains
  & Routes → **Add Custom Domain** → `calc.inference.systems`.
- Cloudflare auto-creates the CNAME record (DNS proxied through CF) and
  provisions the cert.

Alternatively, uncomment the `routes` block in `wrangler.toml` and run
`wrangler deploy` again to set the custom domain via config.

### Auto-deploy

Either:
- Tie deploys to GitHub via Cloudflare's Workers GitHub integration
  (dashboard → Connect to Git), or
- Add a GitHub Actions workflow that runs `npm run deploy` on push to
  `main` with `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` secrets.

Node version is pinned via `.nvmrc` (22) and `package.json` `engines`
(`>=20`) so the build environment stays consistent across local, CI, and
the Cloudflare runner.

## Status

v1. Engine math complete and tested (33 vitest cases). UI functional but
unpolished — Tier-3 input filtering (hiding unsupported dtypes per GPU) is
deferred. Model arch fields should be cross-checked against HuggingFace
`config.json` before relying on numbers.

Spec: [`docs/superpowers/specs/2026-05-11-llm-calculator-design.md`](docs/superpowers/specs/2026-05-11-llm-calculator-design.md)
Plan: [`docs/superpowers/plans/2026-05-11-llm-calculator-v1.md`](docs/superpowers/plans/2026-05-11-llm-calculator-v1.md)

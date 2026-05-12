# LLM Performance Calculator — v1 Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-11
**Scope:** v1 (L0–L2) — dense decoder-only transformer, batching, GQA, per-component quantization, monolithic single-GPU

## Motivation

A web tool that answers three questions about LLM inference:

1. **What is the theoretical performance limit given the inputs?** (Pedagogical: roofline math for the chosen GPU, model, quantization, workload.)
2. **How does a measured result compare to the theoretical limit?** (Diagnostic: gap analysis for `llm-perf` benchmark results, manual comparison in v1.)
3. **What are the deciding factors of performance, and how does changing one (e.g., concurrency) affect the outcome?** (Sensitivity: regime indicator, derivation steps, live recompute as inputs change.)

The calculator is **decoupled from `llm-perf`**. They share no code. Gap-to-theoretical analysis happens by manual comparison (or external JSON ingest later).

## Approach

Build a clean, layered modeling engine first. UI grows incrementally on top. The math evolves historically: start from the simplest model (dense, FP16, single GPU, single request) and add layers for batching, GQA, quantization, parallelism, MoE, disagg, etc. Each layer is additive — earlier behavior remains valid as a special case.

### v1 layers (in scope)

| Layer | Adds |
|---|---|
| **L0** | Dense transformer, FP16 throughout, single GPU, single request. KV size, prefill/decode roofline, TTFT, decode tok/s. |
| **L1** | Batching: per-step batched roofline. Decode regime crosses from memory-bound to compute-bound as batch grows. Aggregate throughput. |
| **L2** | GQA / MQA / MLA (via independent `num_kv_heads`). Independent dtypes for weights, KV cache, and activations. |

### Deferred (v2+)

- Tensor / pipeline / expert parallelism, multi-GPU topology, interconnect bandwidth (NVLink / PCIe / network)
- Disaggregated prefill/decode
- MoE (active vs total params)
- `achievable` operating points sourced from microbenchmarks
- Library-calibrated efficiency factors (the "L6" overlay)
- Speculative decoding, prefix caching, paged-attention waste factor
- Multi-node, multi-host
- Custom GPU/model input UI; JSON import; URL-shareable state
- Roofline plots, sweep mode, side-by-side comparison

## Architecture

Single Vite project, TypeScript, deploys as a static site. No backend.

```
calc/
  src/
    engine/        # pure functions, no DOM, heavily tested
    data/          # property database: gpus.ts, models.ts, dtypes.ts
    ui/            # thin Svelte shell
  test/            # vitest, TDD
  index.html
  vite.config.ts
  package.json
```

**Engine is decoupled from UI.** No DOM references in `engine/` or `data/`. UI imports engine. This is what lets the UI be swapped or skinned without touching the math.

**Framework:** Svelte. The interaction model — many inputs all feeding into all outputs that recompute live — is what Svelte is built for, and the v1 UI stays small (~200-300 lines).

**Testing:** Vitest. TDD per project convention: write failing test, implement minimum to pass, refactor. Engine functions are pure, easy to test against hand-computed reference values.

## Engine API

### Input types

```ts
type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4'

interface GpuOperatingPoint {
  id: string                    // 'peak'; v2 adds 'achievable', 'tdp-350w', etc.
  label: string
  tflops: Partial<Record<Dtype, number>>
  hbmBandwidthGBs: number
}

interface GpuVariant {
  id: string                    // 'sxm-80', 'pcie-94', 'oam-192'
  label: string                 // 'SXM 80GB'
  hbmCapacityGB: number
  operatingPoints: GpuOperatingPoint[]   // at least 'peak'
}

interface GpuSpec {
  id: string                    // 'h100', 'mi300x'
  name: string                  // 'NVIDIA H100'
  vendor: string
  family?: string
  variants: GpuVariant[]
}

interface ModelArch {
  id: string                    // 'llama-3-8b'
  name: string                  // 'Llama 3 8B'
  family: string                // 'llama-3'
  layers: number
  hiddenDim: number
  intermediateDim: number
  numHeads: number
  numKvHeads: number            // = numHeads for MHA; smaller for GQA/MQA
  headDim: number
  vocabSize: number
  paramCount: number            // stored explicitly, not re-derived
}

interface Quantization {
  weights: Dtype
  kv: Dtype
  activations: Dtype            // also determines compute dtype
}

interface Workload {
  promptTokens: number
  outputTokens: number
  concurrency: number           // batch size for decode
}

interface CalcInput {
  gpu: GpuSpec
  gpuVariantId: string
  model: ModelArch
  quant: Quantization
  workload: Workload
}
```

### Output types

```ts
interface MemoryResult {
  weights: number               // bytes
  kvCachePerRequest: number     // bytes, sized at end-of-generation
  kvCacheTotal: number          // bytes, × concurrency
  activationsPeak: number       // bytes, coarse estimate (see below)
  total: number
  hbmCapacityGB: number         // echoed for UI
  headroom: number              // capacity_bytes − total; signed
  fits: boolean
}

interface PerfTier {
  prefill: { flops: number; bytes: number; timeS: number;
             regime: 'compute' | 'memory' }
  decode:  { flopsPerStep: number; bytesPerStep: number; timePerTokenS: number;
             regime: 'compute' | 'memory'; aggregateTokensPerS: number }
  ttftS: number
  inputTokenRate: number        // prompt tokens / prefill time
  outputTokenRate: number       // == decode.aggregateTokensPerS
}

interface DerivationStep {
  label: string                 // 'KV per token'
  expression: string            // '2 × layers × kv_heads × head_dim × bytes(kv_dtype)'
  value: number
  unit: string                  // 'bytes', 'FLOPs', 's', etc.
}

interface CalcResult {
  memory: MemoryResult                 // operating-point-independent
  perf: Record<string, PerfTier>       // keyed by operating point id
  derivation: DerivationStep[]
}
```

### Math (L0–L2)

`bytes(dtype)` looks up the dtype table. `effective_tflops` and `effective_bw` come from the operating point — for v1 there's only `peak`, so they equal advertised numbers. The operating-point loop produces one `PerfTier` per id.

| Quantity | Formula |
|---|---|
| KV per token per request | `2 × layers × numKvHeads × headDim × bytes(quant.kv)` |
| KV total | `kvPerToken × (promptTokens + outputTokens) × concurrency` |
| Weight bytes | `paramCount × bytes(quant.weights)` |
| Prefill FLOPs | `2 × paramCount × promptTokens + 2 × layers × promptTokens² × hiddenDim` |
| Prefill bytes | `weightBytes + activationsPeak` |
| Prefill time | `max(flops / effective_tflops, bytes / effective_bw)` |
| Decode FLOPs/step | `(2 × paramCount + 2 × layers × seqlen × hiddenDim) × concurrency`, where `seqlen` = average over the output window (approximation: `promptTokens + outputTokens/2`) |
| Decode bytes/step | `weightBytes + kvReadBytes × concurrency` |
| Decode time/step | `max(flops / effective_tflops, bytes / effective_bw)` |
| Aggregate decode tok/s | `concurrency / decodeTimePerStep` |
| TTFT | `prefillTime` (v1: no queueing) |

**`activationsPeak`** approximation:

```
activationsPeak ≈ concurrency × promptTokens × (hiddenDim + intermediateDim) × bytes(quant.activations) × k
```

where `k ≈ 2`, covering one layer's attention output buffer + FFN intermediate. Assumes FlashAttention-style kernels (no materialized `S × S` matrix). Flagged in UI as a coarse estimate. The other three memory components (weights, KV, capacity) are exact given inputs.

**`regime`** is whichever side of `max(...)` is binding. This is the single most useful diagnostic the engine produces.

### Design choices worth flagging

- **`paramCount` is stored, not derived.** Param-count formulas drift across architectures (gate vs no-gate FFN, tied embeddings, etc.). Storing it explicitly keeps the database honest and removes arch-specific code paths.
- **Attention quadratic term is kept separate from the MLP term.** At long context, the `promptTokens²` term dominates prefill. Folding it into a single `2 × params × tokens` formula would hide exactly the effect users need to see.
- **Operating points and variants are orthogonal.** Variants are SKUs (different silicon binning, form factor, memory size); operating points are runtime configurations of a variant (peak vs achievable vs power-capped). Both feed a `(tflops, bw)` tuple to the roofline.
- **Memory is operating-point-independent.** Computed once. The `perf` map is what varies per operating point.

## Property Database

Format: TS modules in `src/data/`, typed against engine interfaces. Compile-time checked.

```
src/data/
  gpus.ts        // GpuSpec[]
  models.ts      // ModelArch[]
  dtypes.ts      // bytes-per-element table
  index.ts       // re-exports
```

### Dtype table

```ts
{ fp32: 4, fp16: 2, bf16: 2, fp8: 1, int8: 1, int4: 0.5 }
```

Mixed-precision micro-formats (mxfp4, nvfp4, fp6) → v2.

### GPU seed

Each variant gets a `peak` operating point sourced from datasheets. `achievable` operating points are punted to v2 once we have microbench data.

| GPU | Variants |
|---|---|
| NVIDIA H100 | SXM-80, PCIe-80, PCIe-94, NVL-188 |
| NVIDIA H200 | SXM-141 |
| NVIDIA A100 | SXM-40, SXM-80, PCIe-40, PCIe-80 |
| NVIDIA L40S | PCIe-48 |
| NVIDIA RTX 5090 | 32 |
| NVIDIA RTX 4090 | 24 |
| AMD MI300X | OAM-192 |

### Model seed

Dense / GQA-class only; MoE deferred. Current as of early 2026:

- Qwen3: 1.7B, 4B, 8B, 14B, 32B
- Llama 3.3: 70B
- Llama 3.1: 405B
- Gemma 3: 12B, 27B
- Mistral Small 3.1: 24B
- Mistral Large 2: 123B
- Phi-4: 14B

Arch fields come from each model's HuggingFace `config.json`. `paramCount` taken from official sources, not re-derived.

Notes:
- **Llama 4** is MoE-only; deferred with the other MoE families (DeepSeek-V3/R1, Qwen3-MoE, Mixtral).
- **Gemma 3** uses logit soft-capping and a mixed sliding-window attention pattern. Roofline math is unaffected; users comparing to measured Gemma numbers may see efficiency oddities.

### Extension path

Adding a GPU or model is a PR that appends to the relevant TS array. No upload UI, no "paste your own spec" feature for v1.

## UI Shell (v1)

Single page, no routing. **Top-down flow:**

1. **Inputs (top):** all fields visible.
   - GPU selector → variant selector (cascading)
   - Model selector
   - Quantization: three dropdowns (weights / KV / activations)
   - Workload: three number inputs (prompt tokens / output tokens / concurrency)
2. **Outputs (below):**
   - **Memory panel** — segmented bar chart (weights / KV / activations / headroom), OOM rendered red.
   - **Performance panel** — for each operating point of the chosen variant, a table with TTFT, prefill regime, decode time/token, decode regime, input tok/s, output tok/s. Regime badges are visually distinct.
3. **Derivation panel (side drawer, collapsed by default):** labeled steps with formulas and computed values, in order. Expanding the drawer overlays main content without reflowing it.

Every input change triggers a synchronous recompute. No debounce.

**Total UI:** ~200-300 lines of Svelte.

## Testing

Vitest, TDD. Engine tests are load-bearing.

1. **Reference fixtures.** A handful of canonical (gpu, variant, model, quant, workload) tuples with hand-computed expected values for every field of `CalcResult`. e.g., Llama-3-8B FP16 on H100 SXM-80 with `{ prompt: 2048, output: 512, concurrency: 16 }`.
2. **Per-quantity tests.** Separate test for KV math, weight bytes, prefill FLOPs, decode FLOPs, roofline crossover, regime classification. Each is a small pure function with its own test.
3. **Derivation parity test.** For each fixture, assert that the steps in the `derivation` array actually compute to the final result. "Show the math" can't drift from the answer.
4. **Boundary cases:** OOM (`fits: false`), batch=1, very long context (attention-quadratic regime).

UI tests are not v1. The engine is what can go silently wrong with bad numbers; the UI is checkable by eye.

## Evolution Path

v2+ layers slot in as additive changes:

- **Parallelism (TP/PP/EP).** Adds optional `parallelism: { tp, pp, ep }` to `CalcInput`. Engine divides per-GPU compute/memory; adds communication cost terms.
- **Multi-GPU topology.** Adds `topology: { intraNodeBw, interNodeBw }`. Communication cost terms become topology-aware.
- **Disagg P/D.** Adds a second GPU pool to `CalcInput` and a KV-transfer cost term to results.
- **MoE.** Adds `numExperts`, `numExpertsPerToken` to `ModelArch`. Active vs total params distinction in math.
- **`achievable` operating points.** Pure data addition; no code change.
- **Library-calibrated tier.** New operating point types (per-library efficiency factors); pure data + lookup change.

The `CalcResult` shape stays stable: `memory` and `perf[id]` records gain fields, never lose them.

# DeepSeek V4 Attention + MTP + fp4 — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-13
**Scope:** Add support for DeepSeek V4 (V4-Pro, V4-Flash). Three orthogonal additions, all shipped together because they're inseparable in the V4 release:

1. **`'csa-hca-hybrid'`** attention variant — captures V4's three-layer-type scheme (sliding + CSA + HCA) with token-chunking KV compression, per-CSA-layer indexer (not modeled, footnoted), and per-CSA/HCA-layer sliding side branch.
2. **`numNextnLayers`** field on `ModelArch` — Multi-Token Prediction depth. V4 ships with depth=1 (≈ 2 tokens per forward pass).
3. **`'fp4'`** added to `Dtype` — Blackwell-class GPUs have native fp4 tensor cores at a distinct TFLOPS rate from int4.

## Motivation

DeepSeek V4 (released April 2026, 1.6T total / 49B active for V4-Pro, 284B / 13B for V4-Flash) abandons MLA in favor of a token-chunking compression scheme:

- **CSA (Compressed Sparse Attention)**: every `m=4` consecutive tokens compressed into 1 KV entry via softmax-weighted pooling; per-query sparse attention selects `topK` compressed entries (`topK=512` Flash, `topK=1024` Pro). Includes a 64×128 indexer that scores all compressed entries to pick the top-K — same architectural role as DSA's lightning indexer.
- **HCA (Heavily Compressed Attention)**: every `m'=128` tokens compressed into 1 entry, then **dense** attention over all compressed entries.
- **Sliding side branch**: every CSA/HCA layer also runs a small (`n_win=128`) sliding window attention in parallel; outputs combined.
- Plus V4-Flash has 2 dedicated sliding-only layers at the start.

Roofline-relevant effects:

- **KV cache at 1M context**: V4-Pro stores ≈ 8 GB vs V3.2's ≈ 70 GB at fp16 — **~11×** smaller. Paper claims "10% of V3.2 KV"; the slight gap is likely production V4 using fp8 KV.
- **Per-token attention compute**: V4's HCA dense-over-compressed-entries scales with `seqlen / 128` per layer, which is larger than V3.2's DSA top-K cap. Paper claims "27% of V3.2 single-token inference FLOPs" but that's hardware-throughput-specific (MTP-amortized + lower-precision compute); our strict FLOPs counting will likely show V4 with *more* attention FLOPs than V3.2 at the same seqlen. This is honest, footnoted.
- **MTP doubles effective decode throughput**: each forward pass produces 2 tokens instead of 1.
- **fp4 weights** halve weight bandwidth vs fp8 on Blackwell; storage halves correspondingly.

This is the **10th** architectural-evolution feature (sliding → MoE → MLA → hybrid → shared experts → DSA → MLA dim fields → linear-MLA → V4 attention/MTP/fp4).

## Schema

### 1. `'csa-hca-hybrid'` attention variant

```ts
type AttentionConfig =
  | ...existing 6 variants...
  | { type: 'csa-hca-hybrid';
      // Layer counts (sum must equal model.layers)
      numSlidingLayers: number;
      numCsaLayers: number;
      numHcaLayers: number;

      // Sliding-window size — applies to dedicated sliding-only layers
      // AND to the per-layer side-branch on CSA/HCA layers.
      slidingWindow: number;

      // CSA params
      csaCompressionM: number;     // V4: 4
      csaTopK: number;             // V4-Flash: 512, V4-Pro: 1024
      csaIndexerHeads: number;     // V4: 64 (footnoted — not used by math)
      csaIndexerHeadDim: number;   // V4: 128 (footnoted)

      // HCA params
      hcaCompressionM: number;     // V4: 128
    }
```

Nine fields. Layer-count invariant `numSlidingLayers + numCsaLayers + numHcaLayers === model.layers` enforced by runtime check in `attendedSeqlenSummedOverLayers`, same pattern as `'hybrid'` and `'linear-mla-hybrid'`.

### 2. MTP — `numNextnLayers` on `ModelArch`

```ts
interface ModelArch {
  // ... existing
  numNextnLayers: number;  // MTP depth — 0 for non-MTP models
}
```

Required field. 22 existing entries retrofit to `numNextnLayers: 0`. Forces every model author to think about MTP — same discipline that drove `numSharedExperts`.

### 3. `'fp4'` Dtype

```ts
export type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'fp4' | 'int8' | 'int4'
```

`bytesOf('fp4') = 0.5`. Separate from `int4` because Blackwell fp4 tensor cores have their own TFLOPS rate distinct from int4. (Operating-point TFLOPS records are keyed by Dtype, so fp4 needs to be its own value to receive its own throughput entries.)

## Math

### Three existing helpers — new `csa-hca-hybrid` branch

```ts
// kvBytesPerTokenPerLayer: returns the per-compressed-entry rate. The compression
// factor and side-branch buffer are accounted for in attendedSeqlenSummedOverLayers.
if (att.type === 'csa-hca-hybrid') {
  return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

// attentionDim: per-query-head × head_dim. MQA semantics — full Q heads, 1 KV head.
if (att.type === 'csa-hca-hybrid') {
  return model.numHeads * model.headDim
}

// attendedSeqlenSummedOverLayers: KV-storage vs attention-compute formulas differ
// for CSA (storage = seqlen/m_csa per layer; compute = topK per layer).
if (att.type === 'csa-hca-hybrid') {
  const slidingContrib = att.numSlidingLayers * Math.min(seqlen, att.slidingWindow)
  const csaCount = forKv ? (seqlen / att.csaCompressionM) : att.csaTopK
  const csaContrib = att.numCsaLayers * (csaCount + att.slidingWindow)
  const hcaContrib = att.numHcaLayers * (seqlen / att.hcaCompressionM + att.slidingWindow)
  if (att.numSlidingLayers + att.numCsaLayers + att.numHcaLayers !== model.layers) {
    throw new Error(`csa-hca-hybrid layer counts must sum to model.layers: ` +
      `${att.numSlidingLayers} + ${att.numCsaLayers} + ${att.numHcaLayers} ≠ ${model.layers}`)
  }
  return slidingContrib + csaContrib + hcaContrib
}
```

For uniform-attention models the `forKv` flag is irrelevant; for `mla-dsa` it already distinguishes storage (full seqlen) from compute (topK); for `csa-hca-hybrid` it does the same per-CSA-layer thing plus accounts for sliding side branches on every CSA/HCA layer.

### Decode-step adjustment for MTP

In `computeDecode`, after computing `timeS`:

```ts
const mtpFactor = 1 + model.numNextnLayers
return {
  flopsPerStep,    // per forward pass — unchanged
  bytesPerStep,    // per forward pass — unchanged
  timePerTokenS: timeS / mtpFactor,                              // effective per-token time
  regime,
  aggregateTokensPerS: workload.concurrency * mtpFactor / timeS  // effective throughput
}
```

Per-pass `flopsPerStep` and `bytesPerStep` are unchanged — they describe the work per forward pass. The `mtpFactor` only adjusts the user-visible throughput / per-token-time. Models with `numNextnLayers: 0` see no change (factor = 1).

### What's NOT modeled (intentionally footnoted)

- **CSA lightning indexer compute and state**: fp16 indexer would dominate FLOPs at long context; production V4 quantizes to fp8/int8. Same dodge as V3.2's DSA. Schema stores the indexer params for data integrity, math ignores them.
- **Query compression (`d_c`)**: V4's Q is projected through a `d_c`-dim bottleneck (1024 Flash, 1536 Pro). Per-token Q-projection FLOPs are folded into `activeParams`.
- **Output projection groups (`g`, `d_g`)**: same — folded into `activeParams`.
- **Hash MoE routing for first 3 MoE layers**: small per-layer overhead; not modeled.
- **Manifold-Constrained Hyper-Connections (mHC)**: residual variant; no compute or bytes impact at roofline level.
- **MTP verification overhead**: the `(1 + depth)` throughput gain assumes all predicted tokens are accepted by speculative decoding. Production V4 sees ~75-85% acceptance; our model gives an upper bound on speedup.

### Numerical impact

**V4-Pro KV cache at 1M context, fp16**:

Per-compressed-entry bytes = `2 × numKvHeads × headDim × bytes(fp16) = 2 × 1 × 512 × 2 = 2048` (K and V both stored per entry).

- 30 CSA layers × `(1M/4 + 128) × 2048 = ~512 MB/layer` = **15.36 GB**
- 31 HCA layers × `(1M/128 + 128) × 2048 = ~16.26 MB/layer` = **504 MB**
- **Total ≈ 15.86 GB** vs V3.2's `61 × 1152 × 1M ≈ 70.3 GB` → **~4.4× smaller** at apples-to-apples fp16

The paper's "10% of V3.2 KV" claim assumes V4 deployments use **fp8 KV** while V3.2 uses fp16 (paper-consistent). At fp8, V4-Pro KV at 1M ≈ 7.93 GB ≈ 11% of V3.2 at fp16 ≈ paper's 10×. Our calc supports `quant.kv = 'fp8'` already; users can model V4 production by selecting fp8 KV.

**V4-Pro attention FLOPs per token at 1M decode** (this PR's math): ≈ 36 B ops vs V3.2's ≈ 9 B ops — V4 will appear *higher* in our roofline. The paper's "27% FLOPs" claim incorporates MTP amortization (V4 = 2 tokens/pass) and lower-precision compute (fp4 weights × fp8 indexer × fp8 activations) that strict FLOPs counting doesn't capture. Spec documents this discrepancy honestly.

**V4-Pro decode throughput with MTP=1**: 2× the without-MTP equivalent (modulo MTP acceptance rate).

## Data

### V4-Flash entry

```ts
{
  id: 'deepseek-v4-flash', name: 'DeepSeek-V4-Flash', family: 'deepseek',
  layers: 43, hiddenDim: 4096, intermediateDim: 2048, vocabSize: 129280,
  numHeads: 64, numKvHeads: 1, headDim: 512,
  paramCount: 284_000_000_000,
  numNextnLayers: 1,
  attention: {
    type: 'csa-hca-hybrid',
    numSlidingLayers: 2, numCsaLayers: 21, numHcaLayers: 20,
    slidingWindow: 128,
    csaCompressionM: 4, csaTopK: 512,
    csaIndexerHeads: 64, csaIndexerHeadDim: 128,
    hcaCompressionM: 128
  },
  architecture: {
    type: 'moe',
    numExperts: 256, numExpertsActive: 6,
    numSharedExperts: 1,
    activeParamCount: 13_000_000_000
  }
}
```

`intermediateDim: 2048` uses `moe_intermediate_size` (V4-Flash is all-MoE — no dense intermediate field in the HF config). The activation-memory estimate (`workload.concurrency × prompt × (hidden + intermediate) × bytes`) becomes slightly conservative for V4 since per-expert MoE width is smaller, but the result is in the right ballpark.

Layer split derived from V4-Flash's `compress_ratios` `[0, 0, 4, 128, 4, 128, ..., 4, 128, 4, 0]`:
- Index 0, 1: `0` → sliding-only (2 layers)
- Indices 2-41: alternating `4, 128, 4, 128, ...` starting with `4` → 21 CSA + 20 HCA
- Index 42: `0` → terminal entry, not a separate layer (MTP layer accounted for via `numNextnLayers`)

### V4-Pro entry

```ts
{
  id: 'deepseek-v4-pro', name: 'DeepSeek-V4-Pro', family: 'deepseek',
  layers: 61, hiddenDim: 7168, intermediateDim: 3072, vocabSize: 129280,
  numHeads: 128, numKvHeads: 1, headDim: 512,
  paramCount: 1_600_000_000_000,
  numNextnLayers: 1,
  attention: {
    type: 'csa-hca-hybrid',
    numSlidingLayers: 0, numCsaLayers: 30, numHcaLayers: 31,
    slidingWindow: 128,
    csaCompressionM: 4, csaTopK: 1024,
    csaIndexerHeads: 64, csaIndexerHeadDim: 128,
    hcaCompressionM: 128
  },
  architecture: {
    type: 'moe',
    numExperts: 384, numExpertsActive: 6,
    numSharedExperts: 1,
    activeParamCount: 49_000_000_000
  }
}
```

Layer split from V4-Pro's `compress_ratios` `[128, 128, 4, 128, 4, ..., 4, 0]`:
- Index 0, 1: `128` → HCA (the paper's "first two layers we use HCA")
- Indices 2-60: alternating `4, 128, ...` starting with `4` → 30 CSA + 29 HCA
- Index 61: `0` → terminal (MTP, accounted for via `numNextnLayers`)
- Total: 30 CSA + (2 + 29) HCA = 30 CSA + 31 HCA = 61 layers ✓

### Retrofit — 22 entries gain `numNextnLayers: 0`

Every existing model entry gains the required field. Mechanical change; TypeScript catches missing entries at compile time.

## UI

No required changes. The model selector still shows the model name. Visible downstream effects:

- V4 entries show dramatically smaller KV cache than V3.2 at long context (paper-validated ~10× reduction).
- V4 decode `aggregateTokensPerS` is 2× what raw compute alone would give (MTP throughput gain).
- The roofline plot's decode marker for V4 moves rightward relative to V3.2 (more compute per byte loaded — V4 stores less KV per token).

Optional future caption: "CSA/HCA hybrid: 2 sliding / 21 CSA (top-K 512) / 20 HCA; MTP=1; fp4 weights".

## Testing

- **Helper unit tests** for the `'csa-hca-hybrid'` branch in `kvBytesPerTokenPerLayer`, `attentionDim`, `attendedSeqlenSummedOverLayers` (storage vs compute via `forKv`), plus the layer-count invariant.
- **Memory / prefill / decode regression tests** with a synthetic `'csa-hca-hybrid'` fixture exercising all three layer types.
- **MTP regression test**: a synthetic dense model with `numNextnLayers: 1` confirms `aggregateTokensPerS` doubles vs `numNextnLayers: 0` (the math change is isolated and testable independently).
- **`bytesOf('fp4') === 0.5`** unit test.
- **Retrofit regression**: all 88+ current tests pass byte-for-byte. Math doesn't read `numNextnLayers` when it's 0, and existing models all get 0.
- **Integration tests**:
  - V4-Pro at 1M context — `kvCachePerRequest` matches the per-layer-type sum; ratio vs V3.2 in `(8, 15)` range.
  - V4-Flash at 128k — basic sanity on KV / decode bytes.
  - V4-Pro with MTP — `aggregateTokensPerS` is exactly 2× the same calc with `numNextnLayers` set to 0.

Net additions: ~14 cases on top of current 88+.

## Evolution path

- **CSA indexer overhead modeling**: needs an "auxiliary-precision FLOPs" abstraction in `Quantization`. Same future feature that would handle DSA's indexer. Defer until precision regimes are a first-class concept.
- **MTP acceptance rate**: introduce a `mtpAcceptanceRate: number` field (default 1.0) on `ModelArch`. Multiply the `mtpFactor` by acceptance. Production V4 uses ~0.75-0.85 acceptance — modeling this lets the calc give realistic throughput estimates rather than upper bounds.
- **fp4 KV cache**: V4 production may use fp4 or fp8 KV cache (paper's "10% KV vs V3.2" implies low-precision KV). Our `Quantization.kv` already supports fp8; fp4 is now available too. Data entries can set `quant.kv = 'fp4'` to model this.
- **Hash MoE routing overhead**: if first-3-layer hash routing becomes interesting for precision, add a small fixed-FLOPs term per hash-routed MoE layer. Not currently warranted.
- **GPU operating points for fp4**: Blackwell-class GPUs (B100, B200, GB200) and AMD MI355X+ support fp4 natively. Adding fp4 TFLOPS entries to those operating points is data-only work for follow-up.

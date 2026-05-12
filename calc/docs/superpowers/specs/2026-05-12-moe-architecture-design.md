# Mixture of Experts (MoE) Architecture — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-12
**Scope:** Add support for uniform MoE models (Mixtral 8x7B canonical user). Routed experts only — DeepSeek-style shared experts, fine-grained expert variants, expert-capacity factors, and routing-overhead modeling are out of scope and tracked as separate follow-up features.

## Motivation

The calculator currently assumes dense FFN — every parameter contributes to every per-token compute step, every weight is read from memory per token. This is wrong for MoE models where each layer's FFN is split into N experts and only K are active per token. The numerical impact is:

- **Compute** scales with active params, not total params. Mixtral 8x7B at ~12.9B active does decode FLOPs like a 13B dense model, not a 47B one.
- **Decode memory bandwidth** scales similarly — only active expert weights are loaded per step.
- **Storage** remains at total params — all experts live in VRAM whether or not they're routed to this token. MoE solves compute, not capacity.

These three asymmetries are the core MoE insight, and modeling them gives the calculator correct numbers for Mixtral and the wider MoE family.

This is the **second** architectural evolution feature in the planned sequence:

1. Sliding window attention (shipped)
2. MoE (this spec)
3. MLA (Multi-head Latent Attention, DeepSeek-V2/V3) — separate spec
4. Hybrid attention layers (Gemma 3's pattern) — separate spec

## Schema

A second axis on `ModelArch`, mirroring the discriminated-union pattern established by `attention`:

```ts
type ArchitectureConfig =
  | { type: 'dense' }
  | { type: 'moe'; numExperts: number; numExpertsActive: number; activeParamCount: number }

interface ModelArch {
  // ... existing fields including attention ...
  architecture: ArchitectureConfig
}
```

`activeParamCount` is stored explicitly rather than derived from arch fields. Computing active params from layer / hidden / intermediate / expert geometry requires knowing per-architecture details (gate vs no-gate FFN, embedding sharing, attention parameter share). Storing the value from the model card matches the existing approach for `paramCount` and avoids per-family formulas.

### Retrofit of existing models

All 13 current entries get `architecture: { type: 'dense' }` added. TS refuses to compile until each entry is tagged — same mechanical retrofit pattern as the sliding-window feature.

### Composition with existing axes

`attention` and `architecture` are independent axes. Mixtral 8x7B happens to use both: `attention: { type: 'sliding', window: 4096 }` AND `architecture: { type: 'moe', ... }`. Future MLA + MoE combinations (DeepSeek-V3) compose the same way. The two unions don't reference each other.

## Math

A single helper that abstracts the dense / MoE distinction for compute-related uses:

```ts
function activeParams(model: ModelArch): number {
  return model.architecture.type === 'moe'
    ? model.architecture.activeParamCount
    : model.paramCount
}
```

Used in three places, intentionally NOT in `memory.weights`.

### 1. Prefill FLOPs — `prefill.ts`

```
prefill.flops = 2 × activeParams(model) × prompt        (MLP — was 2 × paramCount × prompt)
              + 2 × layers × prompt × effP × hidden     (attention, unchanged)
```

The attention term is untouched — attention weights are non-FFN and apply uniformly regardless of expert routing.

### 2. Decode FLOPs per step — `decode.ts`

```
decode.flopsPerStep =
  (2 × activeParams(model) + 2 × layers × effAvg × hidden) × concurrency
```

Per-token routing: only active experts contribute to per-token compute.

### 3. Decode bytes per step — `decode.ts`

```
decode.bytesPerStep = activeParams(model) × bytes(quant.weights)   ← was: paramCount × bytes
                    + memory.kvCachePerRequest × concurrency
```

This is the **core MoE insight** in the math. Per-token weight bandwidth is the active subset, not the total. For Mixtral 8x7B at 12.9B active params on H100 SXM-80, decode bytes/step ≈ 25.8 GB (FP16), giving ~7.7 ms/token (memory-bound) — versus a hypothetical dense 47B model that would burn ~28 ms/step reading all 94 GB.

### Memory storage — UNCHANGED

```
memory.weights = paramCount × bytes(quant.weights)         (total, all experts loaded)
memory.kvCachePerRequest = ...                             (unchanged from sliding window)
memory.total = weights + kvCacheTotal + activationsPeak    (still includes all experts)
```

Mixtral 8x7B's 93 GB total weight load still won't fit on a single H100 SXM-80. That's the right output — MoE doesn't help capacity sizing.

### Prefill bytes — UNCHANGED

```
prefill.bytes = memory.weights + memory.activationsPeak    (still total)
```

Over a prompt of any meaningful length, statistical routing touches every expert at least once. Using total `paramCount` here is a tight upper bound. Prefill is usually compute-bound anyway, so the slight over-count rarely changes the regime.

## Data

New entry in `src/data/models.ts`:

```ts
{
  id: 'mixtral-8x7b', name: 'Mixtral 8x7B v0.1', family: 'mistral',
  layers: 32, hiddenDim: 4096, intermediateDim: 14336,
  numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
  paramCount: 46_702_792_704,                  // total, from model card
  attention: { type: 'sliding', window: 4096 },
  architecture: {
    type: 'moe',
    numExperts: 8,
    numExpertsActive: 2,
    activeParamCount: 12_879_204_352           // from model card
  }
}
```

All arch fields must be verified against `mistralai/Mixtral-8x7B-v0.1/config.json` on HuggingFace using the `verifying-achievable-perf-numbers` discipline (read the actual config, not a paraphrase). Specifically: `num_local_experts`, `num_experts_per_tok`, plus the standard architecture fields. The paramCount and activeParamCount values come from the model card / safetensors metadata, not the config.

Mixtral 8x7B was chosen as the canonical first MoE because:

- It's the first major open-weights MoE LLM (Dec 2023)
- It also uses sliding window, which validates that the two axes compose cleanly
- DeepSeek-V3 (the natural next MoE target) wants MLA as well, which is feature #3 — pulling it in here would over-scope

Other models that could be added later as data-only PRs: Mixtral 8x22B (same shape, larger), Qwen3-30B-A3B, Qwen3-235B-A22B, Llama 4 Scout/Maverick.

### Retrofit pass

The 13 existing model entries each gain `architecture: { type: 'dense' }` as a new required field. TS refuses to compile the codebase until every entry is tagged — same change-detection mechanism as the sliding-window retrofit.

## UI

No required UI changes. The model selector still shows the model name; the dense / MoE distinction is invisible at the UI level for v1 of this feature.

Visible downstream effects:

- A Mixtral 8x7B run shows a memory bar dominated by weights (no different from a dense 47B model would), with `fits = false` on most consumer cards.
- Decode time/token is similar to a dense 13B (the active size), not a dense 47B — the perf table shows the asymmetry.
- The roofline plot's prefill/decode markers sit at the same arithmetic-intensity as a 13B-dense workload, even though the model has 47B params total. This is the right diagnostic visual for MoE.

Optional follow-up (deferred): a small caption near the model selector showing `MoE: 8 experts, 2 active` when the model uses MoE. Punt to a follow-up PR if useful.

## Testing

- **Helper unit test** for `activeParams(model)`: returns `paramCount` for dense, returns `activeParamCount` for MoE.
- **Prefill test** — synthetic MoE model variant with explicit numExperts/numExpertsActive/activeParamCount. Assert FLOPs use `activeParams`, not `paramCount`.
- **Decode test** — same fixture. Assert `flopsPerStep` uses `activeParams`. Assert `bytesPerStep` uses `activeParams × dtype` + KV.
- **Memory regression** — assert `memory.weights` STILL uses `paramCount` (total), not activeParams, regardless of architecture type.
- **Regression** — all 41 current tests pass byte-for-byte (existing models all dense + retrofitted).
- **Integration test** — Mixtral 8x7B at a realistic workload (e.g., prompt 2k, output 512, batch 1) on H100 SXM-80. Assert:
  - `memory.weights ≈ 93 GB` → `memory.fits === false`
  - `decode.bytesPerStep` matches the activeParams formula (not paramCount)
  - `decode.regime === 'memory'`

Net additions: ~5 cases on top of the current 41 (~46 total).

## Evolution path

When MLA lands as feature #3: extend `AttentionConfig` union with `'mla'` (separate axis from `architecture`). DeepSeek-V3 entries combine `attention: { type: 'mla', ... }` with `architecture: { type: 'moe', ... }`. The two axes remain orthogonal.

When shared-expert MoE variants are sourced: extend `ArchitectureConfig` with an `'moe-shared'` variant carrying `numSharedExperts`, or extend the `'moe'` variant with an optional `numSharedExperts` field. Math layer adds the shared-expert FLOPs and bytes as always-active contributions.

When expert capacity / drop-token modeling becomes important: add a capacity-factor field. The current `activeParamCount`-based model is the no-drop-tokens baseline.

Each future MoE refinement slots in without disturbing the dense path.

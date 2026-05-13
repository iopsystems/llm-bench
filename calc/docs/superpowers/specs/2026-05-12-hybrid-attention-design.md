# Hybrid Attention Layers — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-12
**Scope:** Add support for hybrid-attention models that interleave sliding-window (local) and full (global) attention layers within a single model. Gemma 3 is the canonical user. Per-layer attention dim variation (different head counts across layer types) is footnoted, not implemented — Gemma 3's two layer types share the same Q/K/V head shape.

## Motivation

Gemma 3 mixes two attention types within a single model: most layers use a small (1024-token) sliding window, every sixth layer uses full global attention. This pattern keeps long-context KV cache and attention compute mostly bounded by the small window while preserving global information flow through the periodic full-attention layers.

The roofline-relevant effects:

- **KV cache**: sliding layers cap at window size, global layers grow linearly with seq. Combined cache is dominated by global layers once seq exceeds the window — asymptotic reduction `layers / numGlobalLayers` (6.2× for Gemma 3 27B with 62 layers / 10 global).
- **Per-step attention compute**: same shape — bounded contribution from sliding layers, linear contribution from global layers.
- **Both effects are baked-in per layer** — they don't simplify to a single scalar `effectiveAttentionLength` the way uniform-attention models do.

The calculator currently models both Gemma 3 12B and 27B as `attention: { type: 'full' }` — a known under-modeling, called out in the MLA spec's evolution-path section. This feature fixes it.

This is the **fourth** architectural evolution feature in the planned sequence:

1. Sliding window attention (shipped, PR #89)
2. MoE (shipped, PR #90)
3. MLA (shipped, PR #92)
4. Hybrid attention layers (this spec)

## Schema

A new variant on the `AttentionConfig` discriminated union:

```ts
type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; kvLoraRank: number; qkRopeHeadDim: number }
  | { type: 'hybrid'; slidingWindow: number; numSlidingLayers: number; numGlobalLayers: number }
```

Three required paired fields on the hybrid variant:

- `slidingWindow` — window size in tokens for sliding-window layers (Gemma 3: 1024)
- `numSlidingLayers` — how many of the model's layers are sliding-window (Gemma 3 27B: 52)
- `numGlobalLayers` — how many are global / full attention (Gemma 3 27B: 10)

**Invariant**: `numSlidingLayers + numGlobalLayers === model.layers`. Enforced by an `assert` in the math helper. The TypeScript type system can't express this constraint directly without coupling the variant to `ModelArch.layers`, so a runtime check is the practical option; the constructor-equivalent (the entry in `models.ts`) is where the check would fire if violated.

### Why store counts rather than a pattern

For roofline math we only need *how many* of each layer type the model has, not *which* layers are which. The order matters for actual inference (interleaving affects which tokens see global context first) but the per-token compute and KV bytes only depend on totals. Storing counts keeps the schema flat and matches the established pattern of `numExperts` / `numExpertsActive` for MoE.

**Footnote on the actual pattern**: Gemma 3's `sliding_window_pattern: 6` means every 6th layer is global (`layer_idx where (idx+1) % 6 == 0`). For 62 layers: indices 5, 11, …, 59 → 10 global, 52 sliding. For 48 layers: 8 global, 40 sliding. The schema doesn't represent this pattern; it'd add complexity without changing any roofline output.

### Composition with existing axes

`attention: { type: 'hybrid', ... }` composes with `architecture: { type: 'dense' | 'moe' }` orthogonally, same as the other attention variants. Future hybrid-MoE models (Llama 4 family, others) would slot in with no schema change.

### Per-layer attention dim variation (not implemented)

Some future model could have different head counts / head dims across its layer types. The schema doesn't currently express this — `attentionDim(model)` returns a single value. If/when needed, the variant gains per-type dim fields. Gemma 3 uses the same Q/K/V geometry on both layer types, so this is moot here.

## Math

The current per-layer-aware helpers in `memory.ts` get refactored so the `layers ×` factor lives in exactly one place — the new layer-aggregating helper. This is required for hybrid (where different layers contribute different seqlens) and is a cleanup for the uniform-attention cases.

```ts
// Per-token, per-layer KV bytes. No `layers ×` factor inside.
export function kvBytesPerTokenPerLayer(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

// Σ over all layers of "effective seqlen for that layer"
export function attendedSeqlenSummedOverLayers(model: ModelArch, seqlen: number): number {
  const att = model.attention
  if (att.type === 'hybrid') {
    if (att.numSlidingLayers + att.numGlobalLayers !== model.layers) {
      throw new Error(`hybrid layer counts must sum to model.layers`)
    }
    return att.numSlidingLayers * Math.min(seqlen, att.slidingWindow)
         + att.numGlobalLayers * seqlen
  }
  const perLayer = att.type === 'sliding' ? Math.min(seqlen, att.window) : seqlen
  return model.layers * perLayer
}

// Unchanged — attention dim is per-head, same for all layer types in Gemma 3.
export function attentionDim(model: ModelArch): number { /* as today */ }
```

`effectiveAttentionLength` is removed (no remaining callers after the refactor).

### Uses

**1. KV cache (`memory.ts`)**

```
kvCachePerRequest = kvBytesPerTokenPerLayer(model, quant.kv)
                  × attendedSeqlenSummedOverLayers(model, prompt + output)
```

For sliding-window-layer-only segments the inner term is `min(seq, W)`; for global-layer segments it's `seq`. The sum collapses the two contributions into a single byte count.

**2. Prefill attention term (`prefill.ts`)**

```
prefill.flops = 2 × activeParams(model) × prompt                                          (MLP)
              + 2 × prompt × attendedSeqlenSummedOverLayers(model, prompt) × attentionDim(model)   (attention)
```

The explicit `model.layers ×` factor that appears today is absorbed into the helper. For uniform-attention models the result is byte-identical: `attendedSeqlenSummedOverLayers = layers × effLen_per_layer`.

**3. Decode attention term (`decode.ts`)**

```
decode.flopsPerStep =
  (2 × activeParams(model) + 2 × attendedSeqlenSummedOverLayers(model, avgSeqlen) × attentionDim(model)) × concurrency
```

Same substitution.

### Numerical impact

For Gemma 3 27B at prompt=8192, output=0, FP16:

- **Current (full attention)**: `2 × 62 × 16 × 128 × 2 × 8192 = 4.16 GB / request`
- **Corrected (hybrid)**: `2 × 16 × 128 × 2 × (52 × 1024 + 10 × 8192) = 8192 × 135168 ≈ 1.10 GB / request`
- **Ratio**: ~3.8× smaller. The sliding-layer contribution is small (~0.43 GB) and the global-layer contribution (~0.67 GB) dominates.

Asymptotic behavior at seqlen → ∞: hybrid uses `numGlobalLayers × seq` for the dominant term vs the dense full-attention `layers × seq`. Asymptotic reduction = `layers / numGlobalLayers = 6.2×` for Gemma 3 27B.

## Data

Two existing model entries in `src/data/models.ts` get their `attention` field rewritten:

```ts
{
  id: 'gemma-3-27b', name: 'Gemma 3 27B', family: 'gemma-3',
  layers: 62, hiddenDim: 5376, intermediateDim: 21504,
  numHeads: 32, numKvHeads: 16, headDim: 128, vocabSize: 262144,
  paramCount: 27_009_000_000,
  attention: { type: 'hybrid', slidingWindow: 1024, numSlidingLayers: 52, numGlobalLayers: 10 },
  architecture: { type: 'dense' }
},
{
  id: 'gemma-3-12b', name: 'Gemma 3 12B', family: 'gemma-3',
  layers: 48, hiddenDim: 3840, intermediateDim: 15360,
  numHeads: 16, numKvHeads: 8, headDim: 256, vocabSize: 262144,
  paramCount: 12_187_000_000,
  attention: { type: 'hybrid', slidingWindow: 1024, numSlidingLayers: 40, numGlobalLayers: 8 },
  architecture: { type: 'dense' }
}
```

All values verified against `unsloth/gemma-3-27b-it/config.json` and `unsloth/gemma-3-12b-it/config.json` (the Google originals are gated; unsloth's mirrors are unauthenticated and load the same `gemma3` architecture in `transformers`).

Confirmed from HF configs:
- 27B: `num_hidden_layers: 62`, `sliding_window: 1024`, `sliding_window_pattern: 6`, `num_attention_heads: 32`, `num_key_value_heads: 16`, `head_dim: 128`, `hidden_size: 5376`, `intermediate_size: 21504`
- 12B: same except `num_hidden_layers: 48`, `num_attention_heads: 16`, `num_key_value_heads: 8`, `head_dim: 256`, `hidden_size: 3840`, `intermediate_size: 15360`

Layer-count derivation from `sliding_window_pattern: 6` (every 6th layer is global):
- 27B: `floor(62/6) = 10` global, `62 - 10 = 52` sliding
- 12B: `48/6 = 8` global, `48 - 8 = 40` sliding

Note: `config.json` reports `vocab_size: 262208`, the seed entry has `262144`. The 64-token delta is multi-modal vocabulary additions (vision tokens) absorbed into the embedding table. Not load-bearing for this feature (attention-side change only); leaving the existing seed values untouched.

## UI

No required UI changes. The model selector still shows the model name; the hybrid distinction is invisible at the UI level.

Visible downstream effects:

- Gemma 3 27B at long prompt shows a KV cache ~4× smaller than the previous (incorrectly full-attention) model. At very long prompts the reduction approaches 6.2×.
- The roofline plot's decode marker moves toward the compute-bound side compared to the previous full-attention model (less per-step KV bytes loaded for the sliding layers).
- The perf table's "Decode time/tok" for Gemma 3 reflects the smaller per-step attention work.

Optional follow-up (deferred): a small caption near the model selector indicating `Hybrid: 1024-tok window, 52 sliding / 10 global layers` when the model uses hybrid attention. Punt to a follow-up PR if useful.

## Testing

- **Helper unit tests** for `attendedSeqlenSummedOverLayers` (full / sliding / mla / hybrid branches) and `kvBytesPerTokenPerLayer` (MLA / non-MLA branches). Added to `test/engine/sliding.test.ts` alongside the existing helper tests.
- **Memory test** — synthetic hybrid fixture. Assert `kvCachePerRequest = kvBytesPerTokenPerLayer × (numSlidingLayers × min(seq, W) + numGlobalLayers × seq)`.
- **Prefill test** — synthetic hybrid fixture. Assert attention term uses the summed value.
- **Decode test** — synthetic hybrid fixture. Assert attention term in `flopsPerStep` uses the summed value.
- **Invariant test** — constructing an `attention: { type: 'hybrid', ... }` whose counts don't sum to `model.layers` throws on first use.
- **Regression** — all 57 current tests pass byte-for-byte. The refactor moves the `layers ×` factor inside the helper; the result is identical for uniform-attention models.
- **Integration test** — Gemma 3 27B at prompt=8192 on H100 SXM-80. Assert:
  - `memory.kvCachePerRequest = 8192 × 135168 ≈ 1.10 GB` (verifiable by hand)
  - Sanity: ratio vs the would-have-been full-attention value (`62 × 8192 × 8192 ≈ 4.16 GB`) is `> 3.5`, asymptote `≤ 6.2`
  - `decode.regime` — sanity check it falls where expected after the attention-term shrinkage

Net additions: ~6 cases on top of current 57.

## Evolution path

When **per-layer attention dim variation** becomes interesting (a model whose layer types have different head counts or head dims): the hybrid variant gains per-type dim fields, and `attentionDim(model)` extends to a summed-over-layers form analogous to `attendedSeqlenSummedOverLayers`. Until then, the single-dim assumption holds across all current variants.

When more **hybrid-MoE** models are sourced (Llama 4 Scout / Maverick claim a hybrid attention pattern combined with MoE): no schema change required, just data entries. The two `AttentionConfig` and `ArchitectureConfig` axes are orthogonal.

When `sliding_window_pattern` becomes a degree of freedom worth modeling (a future model with a non-trivial pattern that affects compute, e.g., asymmetric sliding windows): the variant gains a `pattern` field. The current count-only schema is the no-pattern-dependence baseline.

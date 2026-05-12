# Multi-head Latent Attention (MLA) — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-12
**Scope:** Add support for MLA-attention models (DeepSeek-V2 as the canonical user) so KV cache size and the attention compute term reflect the latent-projection trade-off instead of standard GQA / MHA. Full attention-mechanism math (explicit latent down/up-projection terms) is footnoted, not implemented — at roofline level the simplified model captures the dominant effect.

## Motivation

DeepSeek-V2 introduced Multi-head Latent Attention (May 2024): instead of caching K and V tensors per head, MLA caches a single compressed latent per token per layer. K and V are reconstructed at runtime from the latent (or, via an absorption trick, never explicitly materialized).

The roofline-relevant effects:

- **KV cache shrinks dramatically.** DeepSeek-V2 with 128 heads × 192 dim would be ~5 MB / token under GQA; MLA brings this to ~70 KB / token (~70× reduction).
- **Per-step attention compute also shrinks**, because the absorbed-Q-dot-latent operation works in the small latent dimension (`kv_lora_rank + rope`) rather than the full `numHeads × headDim`. For DeepSeek-V2 (where the equivalent full-attention dim would be `128 × 192 = 24576`): 576 vs 24576 (~43× reduction).
- **Compute / memory trade-off direction inverts**: MLA gives up some compute (extra projections) for much less memory bandwidth. At the roofline level this trade is captured by replacing the attention-term dimension factor.

This is the **third** architectural evolution feature in the planned sequence:

1. Sliding window attention (shipped)
2. MoE (shipped)
3. MLA (this spec)
4. Hybrid attention layers (Gemma 3) — separate spec

## Schema

A new variant on the existing `AttentionConfig` discriminated union:

```ts
type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; kvLoraRank: number; qkRopeHeadDim: number }
```

Two required paired fields on the MLA variant:

- `kvLoraRank` — the compressed latent dimension (DeepSeek-V2/V3: 512)
- `qkRopeHeadDim` — the rotary positional dim stored alongside the latent (V2/V3: 64)

Adding a variant to the union does **not** invalidate existing entries — every current model declares `{ type: 'full' }` or `{ type: 'sliding', ... }` already, and TS continues to enforce paired-field correctness per variant.

### Vestigial fields for MLA entries

`ModelArch` retains its top-level `numKvHeads` and `headDim` fields. For MLA the math doesn't reference them; they're populated with the underlying model's `num_attention_heads` and combined Q-head dim from the HF config for informational fidelity. A more aggressive refactor would push these fields into the attention-config variants (so each variant carries only the fields it actually needs), but that's a wide-reaching change for marginal cleanup and is out of scope here. `notes` on the model entry documents the situation.

## Math

Two helpers, placed next to the existing `effectiveAttentionLength` and `activeParams` in `memory.ts`:

```ts
function kvBytesPerToken(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla') {
    return model.layers * (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  return 2 * model.layers * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

function attentionDim(model: ModelArch): number {
  const att = model.attention
  if (att.type === 'mla') return att.kvLoraRank + att.qkRopeHeadDim
  return model.numHeads * model.headDim
}
```

### Uses

**1. KV cache size (`memory.ts`)**

```
kvCachePerRequest = kvBytesPerToken(model, quant.kv)
                  × effectiveAttentionLength(prompt + output, model.attention)
```

The sliding-window bound still applies (in principle MLA + sliding window can compose, though no model in the current seed does this).

**2. Prefill attention term (`prefill.ts`)**

```
prefill.flops = 2 × activeParams(model) × prompt                      (MLP)
              + 2 × layers × prompt × effP × attentionDim(model)      (attention)
```

For MLA models `attentionDim(model)` is `kvLoraRank + qkRopeHeadDim`; for dense / sliding-window / GQA models it's `numHeads × headDim` (per PR #91, which correctly handles models where `numHeads × headDim ≠ hiddenDim` such as Mistral Small 3.1) — i.e., unchanged from current behavior.

**3. Decode attention term (`decode.ts`)**

```
decode.flopsPerStep =
  (2 × activeParams(model) + 2 × layers × effAvg × attentionDim(model)) × concurrency
```

Same substitution. Per-step weight bandwidth (the term that uses `activeParams × bytesOf(weights)`) is unchanged — the MLA up-projection matrices are part of `paramCount` and therefore implicitly accounted for in the per-step weight read.

### Footnote: full MLA math (not implemented)

The simplified model above replaces the attention-term dimension factor with `kvLoraRank + qkRopeHeadDim`. A more precise model would have separate terms:

- **Latent down-projection per token per layer**: `hidden × kvLoraRank` FLOPs to compress the per-token activations into a latent before storage.
- **Per-step latent retrieval / up-projection compute** (or its absorbed form): for each layer, computing the attention score against all `seq` cached latents involves `seq × (kvLoraRank + qkRopeHeadDim)` MACs. The DeepSeek reference implementation absorbs the K up-projection into Q, so this is the dominant cost. The non-absorbed form would have an extra `seq × num_heads × qk_nope_head_dim × kvLoraRank` term, but it's not how production decode runs.
- **V up-projection at attention output**: `seq × num_heads × v_head_dim × kvLoraRank` per layer per step (also typically absorbed into the output projection).

For a roofline calculator targeting order-of-magnitude regime diagnosis, the simplified single-substitution model captures the dominant effect (the attention term shrinks ~9× for DeepSeek-V2). Adding the explicit projections wouldn't change which side of the roofline ridge the workload lands on, only nudge the absolute number by a few percent. Schema fields `qkNopeHeadDim` and `vHeadDim` are *not* added here — if/when this refinement becomes interesting, those fields slot into the MLA variant.

## Data

New entry in `src/data/models.ts`:

```ts
{
  id: 'deepseek-v2', name: 'DeepSeek-V2', family: 'deepseek',
  layers: 60, hiddenDim: 5120, intermediateDim: 12288,
  numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 102400,
  paramCount: 236_000_000_000,
  attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 },
  architecture: {
    type: 'moe',
    numExperts: 160,                    // routed-only count
    numExpertsActive: 6,
    activeParamCount: 21_000_000_000    // per model card; includes shared-expert contribution
  }
}
```

All arch fields and the MLA latent dimensions must be verified against `deepseek-ai/DeepSeek-V2/config.json` on HuggingFace using the verifying-achievable-perf-numbers skill. `paramCount` and `activeParamCount` come from the model card metadata.

### Shared-expert approximation

DeepSeek-V2 has 160 routed experts + 2 shared experts (always active). The current `architecture: 'moe'` schema doesn't have a `numSharedExperts` field — shared-experts modeling is a separate deferred feature.

The approximation:

- `numExperts: 160` and `numExpertsActive: 6` count **routed experts only**.
- `activeParamCount: 21_000_000_000` is the model-card value, which **already includes the always-active shared experts' parameters**. So compute math (prefill MLP, decode MLP, decode bytes-per-step) comes out correctly without explicitly modeling shared experts.
- `notes` on the model entry calls this out so future readers / refiners understand.

When a `numSharedExperts` field lands, the entry gets updated and the compute math will reproduce the same totals with cleaner field semantics.

### Why DeepSeek-V2 (and not V3)

V2 is historically first (May 2024 vs V3's Dec 2024) and matches the established "evolution by historical order" pattern of the previous features. Both have the same shared-expert approximation issue; V2 wins on chronology.

V3 can be added as a data-only PR once V2 is in (same MLA dimensions, larger expert count, more shared experts).

## UI

No required UI changes. The model selector still shows the model name; the MLA distinction is invisible at the UI level. Visible downstream effects:

- DeepSeek-V2 at long prompts shows a tiny KV cache compared to a hypothetical full-attention 236B model. The memory bar makes this concrete.
- The roofline plot's decode marker moves rightward (higher arithmetic intensity) compared to a same-size dense model with full attention, because per-step compute drops less than per-step bytes — MLA shifts toward the compute-bound side of the ridge.
- The perf table's "Decode time/tok" for DeepSeek-V2 reflects the combined MoE-active-params + MLA-small-attention-term — much faster than the 236B total params would suggest under dense + full-attention modeling.

Optional follow-up (deferred): a small caption near the model selector indicating `MLA: latent 512, rope 64` when the model uses MLA. Punt to a follow-up PR if useful.

## Testing

- **Helper unit tests** for `kvBytesPerToken` (both GQA and MLA branches) and `attentionDim` (both branches). Added to `test/engine/sliding.test.ts` alongside the existing helper tests.
- **Memory test** — synthetic MLA fixture; assert `kvCachePerRequest` uses `layers × (kvLoraRank + rope) × bytes × seqlen`, NOT the GQA formula.
- **Prefill test** — synthetic MLA fixture; assert attention term uses `(kvLoraRank + rope)` instead of `hidden`.
- **Decode test** — synthetic MLA fixture; assert attention term in `flopsPerStep` uses `(kvLoraRank + rope)`.
- **Regression** — all 47 current tests pass byte-for-byte (no existing model uses MLA).
- **Integration test** — DeepSeek-V2 at long prompt (e.g., 32k tokens) on H100 SXM-80. Assert:
  - `memory.kvCachePerRequest` = `60 × (512 + 64) × bytes(fp16) × 32768` (verifiable by hand)
  - Sanity: this number is ~115× smaller than a full-attention 128-head equivalent would produce. The integration test asserts both the formula directly and the order-of-magnitude reduction vs the GQA formula.
  - `decode.regime` — sanity check it falls where expected given the dimensional reduction in attention.

Net additions: ~5 cases on top of the current 47.

## Evolution path

When **shared experts** lands as its own feature: `ArchitectureConfig`'s `'moe'` variant gains a `numSharedExperts` field. DeepSeek-V2's entry gets the field populated (`numSharedExperts: 2`); `numExperts` continues to mean routed-only by convention. Compute math distributes the shared-expert contribution explicitly instead of relying on the activeParamCount aggregate.

When **hybrid attention layers** lands as feature #4: another variant on `AttentionConfig` (`'hybrid'`) describes a per-layer attention pattern. Gemma 3 entries adopt it. MLA remains its own variant — none of the current models combine MLA with hybrid layering, but the schema doesn't preclude future combinations.

When **the full MLA math** becomes interesting (cluster-scale planning, careful kernel sizing): `qkNopeHeadDim` and `vHeadDim` get added to the MLA variant, and the explicit projection terms from the footnote get implemented as separate FLOPs contributions. Until then, the simplified single-substitution model is the documented level of accuracy.

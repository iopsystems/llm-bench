# MLA Dimension Fields — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-12
**Scope:** Split `qk_nope_head_dim` and `v_head_dim` out as required schema fields on the existing `'mla'` and `'mla-dsa'` variants. Retrofit four existing MLA entries (V2 / V3 / V3.2 / Kimi K2 — all `128 / 128`). Add GLM-5 as the canonical user (`qk_nope=192, v=256`). Per-token math is unchanged — the new fields are metadata that captures architectural truth for MLA models whose Q-K head and V-output dims diverge.

## Motivation

DeepSeek V2 introduced MLA with `qk_nope_head_dim = v_head_dim = 128`. Every MLA model we've added since (V3, V3.2, Kimi K2) follows the same convention. Our schema folds this into a single `headDim = qk_nope_head_dim + qk_rope_head_dim` field, and treats Q-K and V as symmetric.

GLM-5 (Feb 2026) breaks the symmetry: `qk_nope_head_dim: 192`, `v_head_dim: 256`. Our current schema can't faithfully represent this — a single `headDim` field collapses the distinction. Today we'd be forced to either lie about the geometry or refuse to add GLM-5.

The roofline-relevant story is mild:

- **Per-token attention compute** in MLA is dominated by the absorbed Q·latent dot product, which works in `kvLoraRank + qkRopeHeadDim` dim. `qkNopeHeadDim` and `vHeadDim` don't appear in the absorbed-form per-token math.
- **KV cache size** is bounded by the compressed latent; neither of the new fields affects KV bytes per token.
- **Parameter count** depends on `vHeadDim × numHeads × hiddenDim` (V-projection size), but we store `paramCount` as a single number from the model card. The new fields are consistent with the model card without changing the totals.

So this feature doesn't change any roofline number. It captures architectural truth so GLM-5 can be modeled honestly, and prepares the schema for any future MLA refinement (e.g., per-V-head compute breakdown) that might exercise the fields.

This is the **8th** architectural-evolution feature in the planned sequence:

1. Sliding window (#89)
2. MoE (#90)
3. MLA (#92)
4. Hybrid attention (#93)
5. Shared experts (#94)
6. (Model expansion data PR #95)
7. DSA (#96)
8. MLA dimension fields (this spec)

## Schema

Add two required fields to both `'mla'` and `'mla-dsa'` variants:

```ts
type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla';
      kvLoraRank: number;
      qkRopeHeadDim: number;
      qkNopeHeadDim: number;    // NEW — non-rope per-head Q-K dim
      vHeadDim: number;         // NEW — per-head V projection dim
    }
  | { type: 'hybrid'; slidingWindow: number; numSlidingLayers: number; numGlobalLayers: number }
  | { type: 'mla-dsa';
      kvLoraRank: number;
      qkRopeHeadDim: number;
      qkNopeHeadDim: number;    // NEW
      vHeadDim: number;         // NEW
      topK: number
    }
```

TypeScript refuses to compile any `'mla'` or `'mla-dsa'` entry without the two new fields. The retrofit (4 existing entries + 4 inline test fixtures) is forced by the type-checker.

### Why required, not optional

Matches the precedent set by `numSharedExperts` (#94): required + explicit `0` (or in this case explicit `128 / 128`) is more honest than optional-with-default-from-elsewhere. Forces every author of an MLA entry to think about the geometry. The retrofit is mechanical.

### Convention

- `qkNopeHeadDim`: per-head dim of the non-rope portion of Q and K. Most MLA models so far: 128.
- `vHeadDim`: per-head dim of the V projection output. Most MLA models so far: 128.
- The model-level `headDim` field stays at `qkNopeHeadDim + qkRopeHeadDim` (= 192 for V2/V3/V3.2/Kimi K2, = 256 for GLM-5). This is the legacy convention — `headDim` for MLA models has been "the equivalent full-attention head dim", which is the Q-K combined dim. It's vestigial for MLA math (the helpers use `kvLoraRank + qkRopeHeadDim`) but still meaningful for non-MLA paths.

## Math

**No changes.**

- `kvBytesPerTokenPerLayer` (MLA / MLA-DSA branch) still returns `(kvLoraRank + qkRopeHeadDim) × bytes(kvDtype)`. Doesn't read the new fields.
- `attentionDim` (MLA / MLA-DSA branch) still returns `kvLoraRank + qkRopeHeadDim`. Same.
- `attendedSeqlenSummedOverLayers` (MLA / MLA-DSA branch) still returns `layers × seqlen` (full) or `layers × min(seqlen, topK)` (DSA). Same.

The new fields are read by zero functions in this PR. They are metadata.

### Why the math doesn't change

For MLA's absorbed-form attention (the production form used in DeepSeek's reference implementation and adopted by all MLA models in the wild):

- Q is projected to `kvLoraRank + qkRopeHeadDim` per head and dot-producted with the cached latent. This is the per-token attention work.
- The V up-projection is folded into the output projection, which we approximate via `activeParams`.
- `qkNopeHeadDim` shows up only if we model the non-absorbed form, which production inference doesn't use.
- `vHeadDim` shows up only if we model the V up-projection as a separate term, which we approximate via `activeParams`.

Both fields are honest captures of architectural geometry that wait for future precision refinements.

## Data

### Retrofit

Four existing MLA entries gain `qkNopeHeadDim: 128, vHeadDim: 128`:

- DeepSeek V2 (`'mla'` variant)
- DeepSeek V3 (`'mla'` variant)
- DeepSeek V3.2 (`'mla-dsa'` variant)
- Kimi K2 (`'mla'` variant)

All four share the same MLA geometry. Values verified against each model's `config.json` during earlier feature work (V2 in #92, V3 in #94, V3.2 in #96, K2 in #95).

### New entry — GLM-5

```ts
{
  id: 'glm-5', name: 'GLM-5', family: 'glm',
  layers: 78, hiddenDim: 6144, intermediateDim: 12288,
  numHeads: 64, numKvHeads: 64, headDim: 256, vocabSize: 154880,
  paramCount: 355_000_000_000,
  attention: {
    type: 'mla-dsa',
    kvLoraRank: 512, qkRopeHeadDim: 64,
    qkNopeHeadDim: 192, vHeadDim: 256,
    topK: 2048
  },
  architecture: {
    type: 'moe',
    numExperts: 256, numExpertsActive: 8,
    numSharedExperts: 1,
    activeParamCount: 32_000_000_000
  }
}
```

All architecture fields verified against `zai-org/GLM-5/config.json`:
- `num_hidden_layers: 78`, `hidden_size: 6144`, `intermediate_size: 12288`, `moe_intermediate_size: 2048`
- `num_attention_heads: 64`, `num_key_value_heads: 64`, `vocab_size: 154880`
- `n_routed_experts: 256`, `n_shared_experts: 1`, `num_experts_per_tok: 8`
- MLA: `kv_lora_rank: 512`, `qk_rope_head_dim: 64`, `qk_nope_head_dim: 192`, `v_head_dim: 256`
- DSA: `index_topk: 2048`
- `headDim = qk_nope + qk_rope = 256` (vestigial for MLA — but stored for convention consistency)

`paramCount` and `activeParamCount`: Z.ai's GLM-5 model card cites ~355B total / ~32B active. Will verify exact figures against safetensors metadata during data entry — adjust if needed.

### Test fixture retrofit

Four inline MLA fixtures across the engine test files use synthetic MLA values:
- `test/engine/sliding.test.ts` (3 fixtures — `kvBytesPerTokenPerLayer`, `attentionDim`, `attendedSeqlenSummedOverLayers`)
- `test/engine/memory.test.ts` (MLA + MLA-DSA cases)
- `test/engine/prefill.test.ts` (1 MLA fixture)
- `test/engine/decode.test.ts` (1 MLA fixture)

Each gets `qkNopeHeadDim: <something>, vHeadDim: <something>` added. Values can be anything that doesn't affect the asserted math — pick `qkNopeHeadDim: 8, vHeadDim: 8` (consistent with the existing fixture's tiny shape) or match the implicit `headDim - qkRopeHeadDim` from each fixture.

## UI

No required UI changes. The model selector still shows the model name; the geometry distinction is invisible at the UI level.

Visible downstream effects from adding GLM-5:

- A second `glm` family model alongside GLM-4.5-Air, this one significantly larger (355B vs 106B total) and using MLA+DSA instead of GQA+MoE.
- At long context, GLM-5 shows the same DSA savings shape as V3.2 (compute capped at topK=2048), but with different absolute numbers because of GLM-5's 78 layers (vs V3.2's 61) and different MLA geometry.

Optional follow-up: a caption like "MLA-DSA: latent 512, rope 64, q-nope 192, v-dim 256, top-2048" when GLM-5 is selected. Punt to a future PR.

## Testing

- **Retrofit regression**: all 77 current tests pass byte-for-byte. Math doesn't read the new fields; the only delta is each MLA / MLA-DSA literal in the data + test files gaining `qkNopeHeadDim: 128, vHeadDim: 128` (or fixture-appropriate values).
- **TypeScript compile-time check**: any MLA / MLA-DSA entry missing the new fields fails to compile. The retrofit catches every such literal for free.
- **Integration test**: GLM-5 at prompt=32k on H100 SXM-80. Asserts:
  - `memory.kvCachePerRequest` matches the MLA formula with GLM-5's layer count: `78 × (512 + 64) × bytes(fp16) × 32768`
  - `decode.bytesPerStep` ≈ `activeParamCount × bytes(fp16) = 32B × 2 = 64 GB`
  - `decode.regime === 'memory'` at batch 1

Net additions: ~2 cases on top of current 77.

## Evolution path

When **V-projection output modeling** becomes useful (likely never at roofline level, but could surface for cluster-scale planning): math gains a term involving `vHeadDim × numHeads × hiddenDim`. Schema is ready.

When **GLM-5.1** is wanted: data-only one-liner. Config is byte-identical to GLM-5 (verified) so the entry is a duplicate with a different ID.

When **DeepSeek V4** lands: V4 abandons MLA in favor of MQA + per-layer KV compression. New variant family entirely (`'mqa-compressed'` or similar). The MLA fields added in this PR remain useful for V2/V3/V3.x/K2 entries that stay on MLA.

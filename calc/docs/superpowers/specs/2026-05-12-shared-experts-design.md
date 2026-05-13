# Shared Experts (MoE) — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-12
**Scope:** Add `numSharedExperts` as a required field on the existing `'moe'` architecture variant, capturing the always-active expert count separately from the routed pool. Add DeepSeek V3 as the canonical user. Per-token math is unchanged — this feature is about schema honesty and unblocking data PRs for shared-expert MoE families (DeepSeek V3+, Kimi K2, GLM-4.5+).

## Motivation

DeepSeek V2 (already in the seed) and several frontier MoE models in 2025/2026 use a routing scheme that splits experts into two pools:

- **Routed experts**: each token activates a top-K subset (gated by a router).
- **Shared experts**: always active for every token (no gating).

The current `'moe'` variant only models the routed pool: `{ numExperts, numExpertsActive, activeParamCount }`. The MoE spec (PR #90) and MLA spec (PR #92) both deferred shared-experts modeling, with the latter explicitly noting:

> DeepSeek-V2 has 160 routed experts + 2 shared experts (always active). The current `architecture: 'moe'` schema doesn't have a `numSharedExperts` field — shared-experts modeling is a separate deferred feature. […] `activeParamCount: 21_000_000_000` is the model-card value, which already includes the always-active shared experts' parameters.

The roofline-relevant story is subtle: per-token compute and per-decode-step bytes are already correct for shared-expert models, because the model card's `activeParamCount` aggregates routed-active + shared-always-on. So this feature doesn't fix any numerical bug — it fixes a schema honesty / data integrity gap. What it unlocks:

1. **Verifiable HF config matching** for the routed/shared split (catches silent data errors).
2. **Cleaner data entries** for frontier MoE models (DeepSeek V3 / Kimi K2 / GLM-4.5-Air) — `numSharedExperts` can be populated from the config instead of buried in a `notes` comment.
3. **Future UI captions** like "256 routed + 1 shared experts, 8 active per token" without further schema changes.
4. **Future precision refinements** (per-token cost breakdown showing shared-vs-routed contribution) plug in cleanly when needed.

## Schema

A single required field added to the existing `'moe'` variant:

```ts
type ArchitectureConfig =
  | { type: 'dense' }
  | { type: 'moe';
      numExperts: number;          // routed-only, unchanged
      numExpertsActive: number;
      numSharedExperts: number;    // NEW — always-active expert count
      activeParamCount: number;    // aggregate routed-active + shared, unchanged
    }
```

**Convention** (matches the convention already documented in the MoE / MLA specs):

- `numExperts`: routed-pool size only. For DeepSeek V3: 256.
- `numExpertsActive`: top-K routed per token. For DeepSeek V3: 8.
- `numSharedExperts`: always-active count, **separate** from the routed pool. For DeepSeek V3: 1.
- `activeParamCount`: model-card aggregate including both routed-active and shared contribution. For DeepSeek V3: 37e9.

Total expert count for a model is `numExperts + numSharedExperts` (e.g., V3: 257; Mixtral 8x7B: 8). This isn't stored as a separate field — derivable on demand.

### Retrofit

The two existing MoE entries each gain `numSharedExperts`:

- **Mixtral 8x7B v0.1**: `numSharedExperts: 0` (no shared experts in this design).
- **DeepSeek V2**: `numSharedExperts: 2` (per V2 config / model card).

TS won't compile any `'moe'` entry without the field — the retrofit is forced by the type-checker.

### Why required, not optional

Required is more honest: every author of an MoE entry has to think about it. The Mixtral case (no shared experts) is captured by an explicit `0`, not by field omission. This matches the spirit of the existing `numExperts` field (which is required even though Mixtral could in principle omit it).

## Math

**No changes.**

`activeParams(model)` continues to return `architecture.activeParamCount` for MoE. Every roofline formula that uses it produces the same number as before:

- `prefill.flops` MLP term: `2 × activeParams × prompt` — unchanged.
- `decode.flopsPerStep` MLP term: `2 × activeParams × concurrency` — unchanged.
- `decode.bytesPerStep` weight term: `activeParams × bytes(weights)` — unchanged.
- `memory.weights` storage term: `paramCount × bytes(weights)` — unchanged.

The `numSharedExperts` field is read by zero functions in this PR. It's metadata.

### Why the math is correct without splitting active params

For roofline-level per-token compute:

- Routed experts contribute `numExpertsActive × routedExpertParamCount` per token (via the router).
- Shared experts contribute `numSharedExperts × sharedExpertParamCount` per token (deterministically).
- The model card's `activeParamCount` is the sum: `routed_active + shared + attention/embedding overhead`.

Using the aggregate is the correct per-token compute. Splitting it into routed-vs-shared would let us *display* the breakdown but wouldn't change the number.

For statistical-routing-touches-every-expert prefill bytes: the upper-bound `paramCount × bytes` already includes both pools, so shared-vs-routed labeling doesn't affect it.

## Data

New entry in `src/data/models.ts`:

```ts
{
  id: 'deepseek-v3', name: 'DeepSeek-V3', family: 'deepseek',
  layers: 61, hiddenDim: 7168, intermediateDim: 18432,
  numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
  paramCount: 671_000_000_000,
  attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 },
  architecture: {
    type: 'moe',
    numExperts: 256, numExpertsActive: 8,
    numSharedExperts: 1,
    activeParamCount: 37_000_000_000
  }
}
```

All arch fields verified against `deepseek-ai/DeepSeek-V3/config.json` (public, not gated):

- `num_hidden_layers: 61`
- `hidden_size: 7168`
- `intermediate_size: 18432` (the dense-layer intermediate; see Known Approximations below)
- `num_attention_heads: 128`, `num_key_value_heads: 128` (no GQA; full multi-head — same as V2)
- `vocab_size: 129280`
- `n_routed_experts: 256`, `n_shared_experts: 1`, `num_experts_per_tok: 8`
- `kv_lora_rank: 512`, `qk_rope_head_dim: 64` (MLA — same dimensions as V2)
- `qk_nope_head_dim: 128`, `v_head_dim: 128` → `headDim = qk_nope_head_dim + qk_rope_head_dim = 192` (matches V2's convention)

`paramCount` and `activeParamCount` are from the V3 model card (671B total, 37B active).

### Known approximations (not new — same as V2)

- **First-k-dense-replace**: V3's `first_k_dense_replace: 3` means layers 0-2 are dense FFN (no routing); layers 3-60 are MoE. Our schema doesn't distinguish per-layer FFN type. The dense layers' contribution is folded into the model card's `37B active` aggregate, so the per-token roofline number is still correct. A future per-layer-FFN-type feature would slot in cleanly.
- **MoE intermediate ≠ dense intermediate**: V3 has `moe_intermediate_size: 2048` (per-expert FFN width) vs `intermediate_size: 18432` (dense layers). Our `intermediateDim` field stores the dense one (matches the convention used by every other entry — including V2). Per-expert width is implicit in `activeParamCount`. The activation-peak calculation uses `intermediateDim`, which captures the dense-layer FFN buffer correctly; MoE layers have smaller per-expert activations, so this is a conservative upper bound (consistent with how `intermediateDim` is used elsewhere).

These approximations are inherited from V2 and don't change with this feature.

### Retrofit pass

Two existing entries gain `numSharedExperts`:

- Mixtral 8x7B v0.1: `numSharedExperts: 0`
- DeepSeek V2: `numSharedExperts: 2`

For V2, the shared-experts approximation is currently documented only in the MLA spec (PR #92) — the data entry itself doesn't have a comment. After this PR, the data entry expresses the architecture directly via the `numSharedExperts: 2` field, and the MLA spec's explanatory paragraph is now historical context.

## UI

No required changes. The model selector still shows the model name; the shared-expert presence is invisible at the UI level for v1 of this feature.

Visible downstream effects from adding DeepSeek V3:

- A second model in the `deepseek` family, with the same MLA + MoE shape as V2 but ~3× larger.
- At long prompts, V3 shows the same dramatic MLA KV-cache savings as V2 (per-layer-bytes-per-token = `(512 + 64) × bytes(fp16) = 1152`, vs the GQA-equivalent of `2 × 128 × 192 × 2 = 98304` — ~85× smaller per layer).
- Decode time/token scales with the 37B active, not the 671B total.

Optional follow-up (deferred): a caption "`numExperts` routed + `numSharedExperts` shared, `numExpertsActive` active per token" near the model selector when the model uses MoE. Punt to a follow-up PR.

## Testing

- **Retrofit regression** — all 62 current tests pass byte-for-byte. The math doesn't read the new field, so every existing per-token number is identical. The retrofit only adds `numSharedExperts: 0` (Mixtral) and `numSharedExperts: 2` (DeepSeek V2).
- **Integration test** — DeepSeek V3 at prompt=32768 on H100 SXM-80. Asserts:
  - `memory.kvCachePerRequest` matches the MLA formula with V3's geometry: `61 × (512 + 64) × bytes(fp16) × 32768`.
  - Sanity: ratio vs the would-have-been-GQA-equivalent for V3 (`2 × 61 × 128 × 192 × 2 × 32768`) is ~43× — same MLA reduction factor as V2 (since both share kv_lora_rank=512 and qk_rope_head_dim=64).
  - `decode.bytesPerStep` matches the activeParams formula: `37e9 × 2 + kvCachePerRequest × 1` (fp16, batch 1).
  - `decode.regime === 'memory'` at batch 1.

Net additions: ~2 cases on top of current 62.

## Evolution path

**Data-only follow-ups** (this PR makes them straightforward):
- Kimi K2 (1T total, 32B active, 384 routed + 1 shared, MLA)
- GLM-4.5-Air (~106B total, MoE + MLA)
- Mixtral 8x22B (data-only — same shape as 8x7B but bigger; `numSharedExperts: 0`)
- DeepSeek V3-0324 (V3 update, same arch)
- DeepSeek-V2-Lite, DeepSeek-Coder-V2-Lite (smaller variants of V2)

**Future schema refinements**:
- **First-k-dense-replace** — a `numDenseLayers` field on the MoE variant for models that mix dense-FFN and MoE layers. Only matters at the level below roofline precision.
- **Per-expert geometry** — split `activeParamCount` into routed and shared contributions. Lets the UI display "of 37B active, X is shared". Pure display granularity; doesn't change roofline.
- **DSA (DeepSeek Sparse Attention)** — new attention variant, unlocks V3.2 and GLM-5/5.1. Independent of this feature.
- **Linear / hybrid attention** (Kimi-Linear, Mamba families) — orthogonal schema work.

Each refinement slots in without disturbing the routed/shared field added here.

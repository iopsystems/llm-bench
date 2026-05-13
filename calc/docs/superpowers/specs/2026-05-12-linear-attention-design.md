# Linear Attention (KDA-style) + MLA Hybrid — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-12
**Scope:** Add `'linear-mla-hybrid'` as a new variant on `AttentionConfig` to support Kimi-Linear-48B-A3B and the broader linear+MLA hybrid family. Models capture (a) constant-in-seqlen state size for KDA-style linear-attention layers and (b) the full softmax-attention work over MLA layers. Per-token KDA compute is modeled explicitly. Stacks on top of the MLA dimension fields feature (PR #98) — Kimi-Linear's full-attention layers use the MLA variant with the post-#98 schema.

## Motivation

Kimi-Linear (Moonshot, Oct 2025) introduces a hybrid attention pattern with two layer types:

- **KDA (Kimi Delta Attention) layers**: linear attention. Per-token "state" is a `numHeads × headDim²` matrix per layer that's read and updated per token. Cost shape is *constant in seqlen* on the cache side and per-token-constant on the compute side.
- **Full MLA layers**: standard MLA with absorbed-form attention. Cost shape is *linear in seqlen* per token (softmax-attention).

The roofline-relevant effects:

- **KV cache**: at long context, most layers (20 of 27 for Kimi-Linear) stop growing. Only the 7 MLA layers' KV cache scales with seqlen. Asymptotic reduction `model.layers / numFullLayers ≈ 3.86×` for Kimi-Linear.
- **Compute**: KDA compute is constant per token (`O(d²)`); MLA compute is linear per token (`O(seq × d)`). At very long context MLA dominates; at very short context KDA dominates. Crossover for Kimi-Linear is at seq ≈ 2.6k.

This is the **9th** architectural-evolution feature in the planned sequence:

1. Sliding window (#89)
2. MoE (#90)
3. MLA (#92)
4. Hybrid attention — sliding+global (#93)
5. Shared experts (#94)
6. Model expansion data PR (#95)
7. DSA (#96)
8. MLA dimension fields (#98)
9. Linear attention + MLA hybrid (this spec)

## Schema

A new variant on `AttentionConfig`:

```ts
type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; kvLoraRank: number; qkRopeHeadDim: number; qkNopeHeadDim: number; vHeadDim: number }
  | { type: 'hybrid'; slidingWindow: number; numSlidingLayers: number; numGlobalLayers: number }
  | { type: 'mla-dsa';
      kvLoraRank: number; qkRopeHeadDim: number;
      qkNopeHeadDim: number; vHeadDim: number;
      topK: number
    }
  | { type: 'linear-mla-hybrid';
      // Inner MLA configuration (for the full-attention layers)
      kvLoraRank: number;
      qkRopeHeadDim: number;
      qkNopeHeadDim: number;
      vHeadDim: number;
      // Per-layer counts (must sum to model.layers)
      numLinearLayers: number;
      numFullLayers: number;
      // Linear-attention geometry (state size = numLinearHeads × linearHeadDim² per layer)
      numLinearHeads: number;
      linearHeadDim: number
    }
```

Eight required fields. Invariant: `numLinearLayers + numFullLayers === model.layers`, enforced by a runtime check in `attendedSeqlenSummedOverLayers` (same pattern as the existing `'hybrid'` variant's count invariant).

### Why a dedicated variant (not an extension to existing `'hybrid'`)

The existing `'hybrid'` variant captures "sliding+global" layer mixing — different effective seqlens, but both layer types are softmax attention. Linear+MLA is a genuinely different layer mix: linear-attention layers have constant-state-per-layer and constant-per-token-compute behavior that softmax attention doesn't share. Conflating them into one variant would force complex internal conditionals; separate variants stay self-documenting.

### Naming

`linear-mla-hybrid` uses `linear` (generic, not Kimi-specific `kda`) so future linear-attention variants — Mamba, RWKV, RetNet — that share the `numHeads × headDim²` state shape can reuse this variant. Variant-specific quirks (e.g., short-conv kernel size, gating mechanism) are below roofline level and not in the schema.

## Math

Three existing helpers gain a new branch:

```ts
// kvBytesPerTokenPerLayer: returns the per-FULL-MLA-layer-per-token bytes.
// Linear layers' state bytes are constant and added separately via linearAttentionStateBytes.
if (att.type === 'linear-mla-hybrid') {
  return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
}

// attentionDim: returns the MLA absorbed-form attention dim.
if (att.type === 'linear-mla-hybrid') {
  return att.kvLoraRank + att.qkRopeHeadDim
}

// attendedSeqlenSummedOverLayers: only the full layers do softmax seq-scan.
if (att.type === 'linear-mla-hybrid') {
  if (att.numLinearLayers + att.numFullLayers !== model.layers) {
    throw new Error(
      `linear-mla-hybrid layer counts must sum to model.layers: ` +
      `${att.numLinearLayers} + ${att.numFullLayers} ≠ ${model.layers}`
    )
  }
  return att.numFullLayers * seqlen
}
```

Two new helpers:

```ts
// Constant per-request state bytes from linear-attention layers. Zero otherwise.
export function linearAttentionStateBytes(model: ModelArch, kvDtype: Dtype): number {
  if (model.attention.type !== 'linear-mla-hybrid') return 0
  const a = model.attention
  return a.numLinearLayers * a.numLinearHeads * a.linearHeadDim * a.linearHeadDim * bytesOf(kvDtype)
}

// FLOPs per token from linear-attention layers (constant in seqlen). Zero otherwise.
export function linearAttentionFlopsPerToken(model: ModelArch): number {
  if (model.attention.type !== 'linear-mla-hybrid') return 0
  const a = model.attention
  return 2 * a.numLinearLayers * a.numLinearHeads * a.linearHeadDim * a.linearHeadDim
}
```

### Uses

**`computeMemory.kvCachePerRequest`**:

```
kvCachePerRequest = kvBytesPerTokenPerLayer(model, kv) × attendedSeqlenSummedOverLayers(model, seq, forKv=true)
                  + linearAttentionStateBytes(model, kv)
```

The first term covers the full-MLA layers' KV cache (grows with seq). The second term covers the linear-attention layers' state (constant in seq).

**`prefill.flops`**:

```
prefill.flops = 2 × activeParams × prompt                                                    (MLP)
              + 2 × prompt × attendedSeqlenSummedOverLayers(model, prompt) × attentionDim    (full-MLA attention)
              + prompt × linearAttentionFlopsPerToken(model)                                 (linear-attention compute)
```

Last term: linear-attention compute is per-token constant; over `prompt` tokens it scales linearly.

**`decode.flopsPerStep`**:

```
decode.flopsPerStep =
  (2 × activeParams + 2 × attendedSeqlenSummedOverLayers(model, avgSeq) × attentionDim + linearAttentionFlopsPerToken(model))
  × concurrency
```

**`decode.bytesPerStep`**:

```
decode.bytesPerStep =
  activeParams × bytes(weights)
  + kvCachePerRequest × concurrency                              (state read; includes both MLA KV and KDA state)
  + linearAttentionStateBytes(model, kv) × concurrency           (state write-back for linear layers)
```

The extra `+ linearAttentionStateBytes × concurrency` models the per-step write-back. KV cache is read-only per step (KV appended for new token, but that's small enough to ignore); KDA state is read-and-written. The conservative single-extra-read accounting captures the dominant difference.

### Numerical impact (Kimi-Linear at long context)

At prompt=128k, batch=1, fp16 on H100 SXM-80:

- KDA state bytes: `20 × 32 × 128 × 128 × 2 = 20_971_520` (~20 MB, constant in seq)
- Full-MLA KV cache: `7 × (512 + 64) × 2 × 128k ≈ 1.06 GB`
- **Total `kvCachePerRequest` ≈ 1.08 GB** (MLA dominates; KDA state is ~2% of total)
- Hypothetical all-MLA Kimi-Linear at same workload: `27 × 576 × 2 × 128k ≈ 4.08 GB`
- Ratio: **~3.78×** smaller. Asymptote `27/7 ≈ 3.86×` as seq → ∞.

For decode FLOPs at seq=128k:
- MLA per-token: `7 × 2 × 128k × 576 ≈ 1.03B ops`
- KDA per-token: `20 × 2 × 32 × 128² ≈ 21M ops` (~2% of MLA)

Crossover where KDA compute equals MLA compute: roughly seq ≈ 2.6k. Below that, KDA dominates compute; above, MLA dominates. Kimi-Linear is interesting in the seq >> 2.6k regime.

## Data

New entry in `src/data/models.ts` (after the `glm-5` entry; alongside other `kimi` family entries):

```ts
{
  id: 'kimi-linear', name: 'Kimi-Linear-48B-A3B', family: 'kimi',
  layers: 27, hiddenDim: 2304, intermediateDim: 9216,
  numHeads: 32, numKvHeads: 32, headDim: 192, vocabSize: 163840,
  paramCount: 48_000_000_000,
  attention: {
    type: 'linear-mla-hybrid',
    kvLoraRank: 512, qkRopeHeadDim: 64,
    qkNopeHeadDim: 128, vHeadDim: 128,
    numLinearLayers: 20, numFullLayers: 7,
    numLinearHeads: 32, linearHeadDim: 128
  },
  architecture: {
    type: 'moe',
    numExperts: 256, numExpertsActive: 8,
    numSharedExperts: 1,
    activeParamCount: 3_000_000_000
  }
}
```

All values verified against `moonshotai/Kimi-Linear-48B-A3B-Instruct/config.json` (public):
- 27 layers, hidden 2304, intermediate 9216
- 32 attention heads, 32 KV heads (no GQA on the full layers), vocab 163840
- Full layers: MLA with `kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128`
- `headDim = qk_nope + qk_rope = 192` (legacy convention, vestigial for MLA)
- `linear_attn_config`: 7 full layers at indices `[4, 8, 12, 16, 20, 24, 27]`, 20 KDA layers at the remaining indices, KDA `num_heads=32`, `head_dim=128`
- 256 routed + 1 shared experts, 8 active per token
- `moe_intermediate_size=1024` (smaller than V3's 2048 — captured implicitly in `activeParamCount`)

paramCount / activeParamCount: 48B / 3B from the model card name. Verify against safetensors metadata during data entry.

## UI

No required UI changes. The model selector still shows the model name; the linear/MLA mix is invisible at the UI level.

Visible downstream effects:

- Kimi-Linear at long prompts shows much smaller KV cache than other Kimi-family models. The memory bar makes this visible at very long seqlen workloads.
- Decode bytes per step grow modestly with seq (only 7 layers' KV cache scales); the per-step KDA state read/write contributes a fixed ~40 MB overhead at batch=1.
- The roofline plot's decode marker for Kimi-Linear moves leftward (more compute-bound, less memory-bound) relative to a hypothetical all-MLA equivalent at the same seqlen.

Optional future caption near the model selector: "Linear+MLA hybrid: 20 linear / 7 full attention layers".

## Testing

- **Helper unit tests** for the `'linear-mla-hybrid'` branch in `attendedSeqlenSummedOverLayers`, `kvBytesPerTokenPerLayer`, `attentionDim`, plus dedicated tests for the two new helpers `linearAttentionStateBytes` and `linearAttentionFlopsPerToken`.
- **Layer-count invariant test**: constructing a `'linear-mla-hybrid'` whose layer counts don't sum to `model.layers` throws on first use of the helper.
- **Memory test**: synthetic linear-mla-hybrid fixture. Assert `kvCachePerRequest = mla_term + linear_state_term` with hand-computed values.
- **Prefill test**: assert flops includes `prompt × linearAttentionFlopsPerToken` term.
- **Decode test**: assert `flopsPerStep` includes the KDA compute term; assert `bytesPerStep` includes the KDA state write-back term.
- **Regression**: all 79 current tests (after PR #98) pass byte-for-byte. The new helpers return 0 for non-linear-hybrid models, so all existing call sites are mathematically unchanged.
- **Integration test**: Kimi-Linear at prompt=128k on H100 SXM-80. Asserts:
  - `kvCachePerRequest` formula matches `7 × 576 × 2 × 128k + 20 × 32 × 128² × 2`
  - Ratio vs all-MLA equivalent (`27 × 576 × 2 × 128k`) is `> 3.5` (actual ≈ 3.78×)
  - `decode.regime === 'memory'` at batch 1 (active 3B × 2 = 6 GB weights, well within H100 capacity; memory-bound by weight reads)

Net additions: ~9 cases on top of current 79.

## Evolution path

- **State-write-back precision**: today we add `linearAttentionStateBytes` once to `bytesPerStep`. A more precise model would separate read/write costs (2× factor). For Kimi-Linear at long context the state contribution is sub-5% of bytesPerStep, so the rough modeling suffices.
- **Other linear-attention variants** (Mamba's `d_inner × d_state`, RWKV's per-channel state, RetNet's chunked recurrence): each has its own state shape. The `'linear-mla-hybrid'` variant assumes the canonical "delta rule" form (`numHeads × headDim²` state). If another shape is sourced, add a sibling variant with the appropriate fields.
- **Short conv overhead**: Kimi-Linear's `short_conv_kernel_size: 4` represents a per-KDA-layer 1-D depthwise convolution. Roofline-negligible compared to the `d²` matrix update; punted.
- **Multi-token prediction** (MTP), if a future linear-attention model adds it, is orthogonal — handled by a separate model-level field, not in `AttentionConfig`.
- **DSA-over-linear**: hypothetical future combination. Would warrant a `'linear-mla-dsa-hybrid'` variant or refactoring to a more composable schema.

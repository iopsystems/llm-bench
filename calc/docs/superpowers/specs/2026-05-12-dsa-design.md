# DeepSeek Sparse Attention (DSA) — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-12
**Scope:** Add `'mla-dsa'` as a new variant on `AttentionConfig` to support DeepSeek V3.2 (Dec 2025) and the broader DSA family. Models the sparse-attention compute savings via a `topK` cap on per-query attended seqlen; the lightning indexer overhead is footnoted (not modeled) on the basis that production deployments quantize it. GLM-5 deferred to a follow-up because its MLA dimensions (`qk_nope_head_dim=192`, `v_head_dim=256`) require a separate schema refinement.

## Motivation

DeepSeek V3.2 introduced **DSA** in Dec 2025: instead of attending to all prior tokens, each query selects a sparse top-K subset (typically 2048 tokens) using a small auxiliary "lightning indexer". The roofline-relevant effect:

- **Attention compute drops to `min(seqlen, topK)` per query** — same compute shape as sliding-window attention, but the selection is learned-relevance rather than chronological.
- **KV cache size unchanged** — every past token might be selected later, so all KV entries stay resident. Memory cost mirrors plain MLA.
- **Indexer adds linear-in-seq overhead per query**, but production V3.2 quantizes the indexer to fp8/int8 to amortize it. The published DSA results show net speedup at long context.

The calculator currently has no model for DSA. DeepSeek V3.2 (already historically released) and the GLM-5 family use it. This feature adds the schema and math to capture the dominant effect: the sparse-attention compute savings.

This is the **7th** architectural-evolution feature in the planned sequence:

1. Sliding window attention (#89)
2. MoE (#90)
3. MLA (#92)
4. Hybrid attention layers (#93)
5. Shared experts (#94)
6. (Data-only model expansion: Mixtral 8x22B, Kimi K2, GLM-4.5-Air — #95)
7. DSA (this spec)

## Schema

A new variant on the existing `AttentionConfig` discriminated union:

```ts
type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; kvLoraRank: number; qkRopeHeadDim: number }
  | { type: 'hybrid'; slidingWindow: number; numSlidingLayers: number; numGlobalLayers: number }
  | { type: 'mla-dsa'; kvLoraRank: number; qkRopeHeadDim: number; topK: number }
```

Three required paired fields on the new variant:

- `kvLoraRank` — same meaning as in the `'mla'` variant (compressed latent dimension; V3.2: 512)
- `qkRopeHeadDim` — same meaning as in the `'mla'` variant (rotary positional dim; V3.2: 64)
- `topK` — DSA sparse-attention cap (V3.2: 2048)

Adding a variant doesn't invalidate any existing entry — they all continue to declare `'full'`, `'sliding'`, `'mla'`, or `'hybrid'`.

### Why a dedicated `'mla-dsa'` variant (not an optional field on `'mla'`)

Matches the project's discriminated-union-per-architectural-pattern style (sliding / mla / hybrid each already get their own variant). Keeps the type signature self-documenting and pattern-match-friendly: helper functions branch on `att.type === 'mla-dsa'` to apply DSA-specific math, then delegate to the MLA logic for everything else.

The alternative — an optional `dsaTopK?: number` on the existing `'mla'` variant — would mix two concerns into one variant and force every reader of MLA-related code to consider the optional case. The discriminated-union choice scales better as the schema grows.

### Why not yet a generic DSA wrapper

`{ type: 'dsa'; topK: number; inner: AttentionConfig }` would future-proof DSA over any underlying attention. Punted: no DSA-over-non-MLA model exists in current wild (both V3.2 and GLM-5 are DSA-over-MLA). If a future model ships DSA-over-GQA, we add a sibling variant or refactor at that point. YAGNI.

## Math

All three branched helpers in `memory.ts` gain a new branch:

```ts
export function kvBytesPerTokenPerLayer(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

export function attendedSeqlenSummedOverLayers(model: ModelArch, seqlen: number): number {
  const att = model.attention
  if (att.type === 'hybrid') {
    if (att.numSlidingLayers + att.numGlobalLayers !== model.layers) { throw new Error(...) }
    return att.numSlidingLayers * Math.min(seqlen, att.slidingWindow) + att.numGlobalLayers * seqlen
  }
  if (att.type === 'mla-dsa') return model.layers * Math.min(seqlen, att.topK)
  const perLayer = att.type === 'sliding' ? Math.min(seqlen, att.window) : seqlen
  return model.layers * perLayer
}

export function attentionDim(model: ModelArch): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') return att.kvLoraRank + att.qkRopeHeadDim
  return model.numHeads * model.headDim
}
```

### Uses

The three call sites (memory.ts / prefill.ts / decode.ts) are unchanged — they already route through the helpers. The behavior change for `'mla-dsa'` models:

- KV cache stays full-shape (`kvBytesPerTokenPerLayer × layers × seqlen` — every past token cached because any might be selected later).
- Attention compute (prefill + decode) drops to `topK`-bounded `attendedSeqlen` once `seqlen > topK`.

### Numerical impact (V3.2 vs V3 at long context)

For DeepSeek V3.2 at prompt=32k on H100 SXM-80:

- **KV cache**: identical to V3 (`61 × 576 × 2 × 32768 ≈ 2.3 GB`). Both share MLA dims and layer count.
- **Attention compute (prefill term)**:
  - V3: `attendedSeq = 61 × 32768 = 1_998_848`
  - V3.2: `attendedSeq = 61 × min(32768, 2048) = 61 × 2048 = 124_928`
  - Ratio: **~16× attention FLOPs reduction** at 32k. Approaches `seqlen / topK = 64×` at 128k.

This is the dominant DSA effect at long context. Production V3.2 realizes a smaller speedup in practice because the lightning indexer adds overhead that quantization mitigates but doesn't eliminate — see the footnote below.

### Footnote: lightning indexer overhead (not modeled)

DSA's lightning indexer scores all past tokens against each query, producing the top-K selection. The indexer adds compute that scales linearly with `seq` per token per layer:

- V3.2 indexer config: `index_n_heads: 64`, `index_head_dim: 128` → indexer dim 8192
- Indexer FLOPs per query per layer: `2 × seqlen × indexer_dim`

At seq=32k in fp16, the indexer term (`32k × 8192 ≈ 262M ops per token per layer`) would dwarf the sparse-attention savings (`2k × 576 ≈ 1.2M ops`). The DSA design only yields net speedup when the indexer runs at lower precision (production V3.2 uses fp8 or int8 indexers, sometimes with additional optimizations like quantized key caches).

Our roofline doesn't represent "auxiliary computations at a different precision than the main attention" — adding it would require either:

- An `indexerHeads` / `indexerHeadDim` schema extension + a separate FLOPs term that uses an indexer-specific TFLOPS rate, OR
- A "compute mix" abstraction where different terms can declare different precisions

Both are real refinements but outside the current scope. The α decision (skip indexer modeling) was made on the basis that DSA's published design intent is the sparse-attention savings; unoptimized indexer was never the production target. See Evolution Path.

## Data

New entry in `src/data/models.ts`:

```ts
{
  id: 'deepseek-v3.2', name: 'DeepSeek-V3.2', family: 'deepseek',
  layers: 61, hiddenDim: 7168, intermediateDim: 18432,
  numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
  paramCount: 671_000_000_000,
  attention: { type: 'mla-dsa', kvLoraRank: 512, qkRopeHeadDim: 64, topK: 2048 },
  architecture: {
    type: 'moe',
    numExperts: 256, numExpertsActive: 8,
    numSharedExperts: 1,
    activeParamCount: 37_000_000_000
  }
}
```

All non-DSA fields verified against `deepseek-ai/DeepSeek-V3.2/config.json`:
- `num_hidden_layers: 61`, `hidden_size: 7168`, `intermediate_size: 18432`
- `num_attention_heads: 128`, `num_key_value_heads: 128`, `vocab_size: 129280`
- `n_routed_experts: 256`, `n_shared_experts: 1`, `num_experts_per_tok: 8`
- `kv_lora_rank: 512`, `qk_rope_head_dim: 64`, `qk_nope_head_dim: 128`, `v_head_dim: 128`
  → `headDim = qk_nope + qk_rope = 192` (matches V2/V3 convention)
- DSA fields: `index_topk: 2048` → `topK: 2048`

`paramCount` and `activeParamCount`: V3.2 adds the lightning indexer parameters on top of V3's geometry — roughly `index_n_heads × index_head_dim × hidden × layers = 64 × 128 × 7168 × 61 ≈ 3.6B extra params`. The model card rounds these into the same 671B / 37B figures used for V3 (sub-1% delta); we follow that convention.

## UI

No required UI changes. The model selector still shows the model name; DSA is invisible at the UI level.

Visible downstream effects:

- DeepSeek V3.2 at long prompts shows the same KV-cache memory footprint as V3, but with much smaller attention compute term. The perf table's "TTFT" for V3.2 at 32k is significantly lower than V3's.
- The roofline plot's prefill marker for V3.2 moves leftward (lower arithmetic intensity) relative to V3 — sparse attention means less compute per byte of KV touched.

Optional follow-up (deferred): a caption "MLA-DSA: latent 512, rope 64, sparse top-2048" near the model selector for DSA models.

## Testing

- **Helper unit tests** for the new `'mla-dsa'` branch in each of:
  - `kvBytesPerTokenPerLayer` (returns same as `'mla'`)
  - `attentionDim` (returns same as `'mla'`)
  - `attendedSeqlenSummedOverLayers` (returns `layers × min(seqlen, topK)`)
- **Memory test** with synthetic `'mla-dsa'` fixture — assert `kvCachePerRequest` uses the MLA formula (DSA doesn't affect KV size).
- **Prefill test** — same fixture, assert attention term uses topK-capped attendedSeq.
- **Decode test** — same fixture, assert flopsPerStep attention term uses topK-capped attendedSeq.
- **Regression** — all 69 current tests pass byte-for-byte (no model uses `'mla-dsa'` yet).
- **Integration test** — DeepSeek V3.2 at prompt=32k on H100 SXM-80. Asserts:
  - `memory.kvCachePerRequest` matches `61 × 576 × 2 × 32768` (same as V3)
  - Sanity: V3.2's prefill attention term, divided by V3's at the same workload, is ≈ `topK / seqlen = 2048 / 32768 ≈ 1/16`

Net additions: ~6 cases on top of current 69.

## Evolution path

When **GLM-5** becomes interesting: extend the MLA family of variants with optional / additional fields for `qk_nope_head_dim` and `v_head_dim` (GLM-5: 192 / 256 vs V3's 128 / 128). May warrant generalizing the "MLA head geometry" into a sub-config that both `'mla'` and `'mla-dsa'` reference.

When **indexer overhead modeling** is worth the precision: introduce a `Quantization`-like field for auxiliary computations OR add `indexerHeads` / `indexerHeadDim` schema fields with an explicit FLOPs term. The schema slot is ready (`'mla-dsa'` variant already exists); only the math and TFLOPS plumbing change.

When **DSA-over-non-MLA** appears (e.g., a future GLM-Light with sparse attention over plain GQA): add a sibling variant (`'gqa-dsa'` or generic `'dsa'` wrapper) without disturbing the existing `'mla-dsa'`.

When **Kimi K2.5 / DeepSeek V4** are unblocked: separate spec, since their `kimi_k25` and `deepseek_v4` arch classes likely introduce further changes beyond DSA.

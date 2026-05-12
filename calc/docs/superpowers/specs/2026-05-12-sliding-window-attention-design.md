# Sliding Window Attention — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-12
**Scope:** Add support for uniform sliding-window attention models (Mistral 7B and similar). Hybrid attention patterns (e.g., Gemma 3's interleaved sliding/global layers) and MLA / MoE / other architectural changes are out of scope and tracked as separate follow-up features.

## Motivation

The calculator currently assumes full causal attention for every model — every token attends to every prior token, KV cache grows linearly with sequence length, prefill attention is quadratic in prompt size. This is correct for the existing seed (Llama 3.x, Qwen3, Gemma 3, Mistral Small 3.1, etc.) but wrong for sliding-window models like Mistral 7B v0.1, where:

- KV cache is bounded by the window size, not the full sequence length
- Prefill attention work is `O(prompt × window)` instead of `O(prompt²)`
- Decode attention is constant per step once seqlen exceeds the window

At long context (prompt ≥ 32k tokens) the numerical difference is order-of-magnitude. The calculator should reflect it.

This is the **first** architectural evolution feature in the planned sequence:

1. Sliding window attention (this spec)
2. MoE (FFN sparse activation) — separate spec
3. MLA (Multi-head Latent Attention, DeepSeek-V2/V3) — separate spec
4. Hybrid attention layers (Gemma 3's pattern) — separate spec

Each lands as its own design / plan / PR cycle.

## Schema

A new required field on `ModelArch`, modeled as a discriminated union so TypeScript prevents invalid combinations of attention type and per-type configuration:

```ts
type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }   // window in tokens

interface ModelArch {
  // ... existing fields unchanged ...
  attention: AttentionConfig
}
```

When future features add MLA, hybrid, etc., the union grows:

```ts
type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; /* MLA-specific paired fields */ }
  | { type: 'hybrid'; /* hybrid-specific paired fields */ }
```

Each variant carries exactly the fields it needs. Invalid combinations (e.g., a `'full'` entry with a `window` field) fail compilation.

### Retrofit of existing models

All 12 current entries in `src/data/models.ts` get `attention: { type: 'full' }` added. The build won't compile until every entry is tagged. Mechanical change, no behavior delta for current models.

### Why this shape

Flat alternatives (`attention: 'full' | 'sliding'` + optional `slidingWindow?: number`) admit invalid states like `attention: 'full', slidingWindow: 4096`. As the attention union grows with MLA and hybrid variants, the flat shape's invalid-state surface grows combinatorially. The discriminated union scales cleanly.

## Math

A single helper, scoped to the engine:

```ts
function effectiveAttentionLength(rawSeqlen: number, attention: AttentionConfig): number {
  if (attention.type === 'sliding') return Math.min(rawSeqlen, attention.window)
  return rawSeqlen
}
```

This helper is used in three places. Models with `attention.type === 'full'` see no behavior change; the helper is the identity.

### 1. KV cache size — `memory.ts`

```
kvPerToken         = 2 × layers × kvHeads × headDim × bytes(quant.kv)
effSeqlen          = effectiveAttentionLength(prompt + output, model.attention)
kvCachePerRequest  = kvPerToken × effSeqlen                        ← was (prompt + output)
kvCacheTotal       = kvCachePerRequest × concurrency               ← unchanged
```

For 32k context on Mistral 7B (window 4096): KV cache per request drops from `32k × per-token` to `4k × per-token`. Roughly 8× memory reduction.

### 2. Prefill FLOPs — `prefill.ts`

```
prefill.flops = 2 × params × prompt                              (MLP — unchanged)
              + 2 × layers × prompt × effSeqlen(prompt) × hidden  (attention)
```

Each prefill token attends to up to `min(prompt, window)` prior tokens. The true causal-attention sum is `Σ min(i, window) for i in 1..prompt`, which differs from `prompt × min(prompt, window)` by a `window/2` correction term. We fold that correction into the leading `2×` constant rather than modeling it explicitly — at the precision a roofline calculator targets, this is below noise.

For `prompt ≤ window` this formula is identical to the existing `2 × layers × prompt² × hidden` form, so full-attention models and short-prompt sliding-window cases produce the same prefill FLOPs as today (zero regression).

### 3. Decode FLOPs per step — `decode.ts`

```
decode.flopsPerStep =
  (2 × params + 2 × layers × effSeqlen(avgSeqlen) × hidden) × concurrency
```

where `avgSeqlen = prompt + output/2` as before. The attention term is now bounded by `window` once `avgSeqlen` exceeds it.

### Bytes — unchanged

Per-step decode bytes already come from `memory.kvCachePerRequest`, which the memory math has already corrected. No additional change needed in `decode.ts` for the bytes channel.

### Where the helper lives

The helper is small enough to live at the top of `memory.ts`, exported for `prefill.ts` and `decode.ts` to import. Single responsibility, easy to unit-test.

## Data update

A new model added to `src/data/models.ts`:

```ts
{
  id: 'mistral-7b-v0.1', name: 'Mistral 7B v0.1', family: 'mistral',
  layers: 32, hiddenDim: 4096, intermediateDim: 14336,
  numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
  paramCount: 7_241_732_096,
  attention: { type: 'sliding', window: 4096 }
}
```

The arch fields and `window` value must be verified against `mistralai/Mistral-7B-v0.1/config.json` on HuggingFace using the `verifying-achievable-perf-numbers` discipline (read the actual config, not a paraphrase). If `config.json` reports different numbers, those are authoritative.

Why Mistral 7B v0.1 specifically:

- It's the canonical sliding-window LLM in open-weights history
- Mistral Small 3.1 and Mistral Large 2 (both in the current seed) dropped sliding window in favor of full attention
- Other current-seed models don't use uniform sliding window — Gemma 3 uses *hybrid* sliding/global pattern which is deferred to feature #4

Other models that could be added later (also using uniform sliding window, all out of scope for this spec): Mistral 7B v0.2 (window 32k), Phi-3-medium-128k (sliding 4k), early Codestral variants.

### Retrofit pass

The 12 existing model entries each gain one line: `attention: { type: 'full' }`. TS refuses to compile the rest of the codebase until this is done — the build is the change-detection mechanism.

## UI

No required UI changes. The model selector still shows the model name; the sliding-window distinction is invisible at the UI level for v1 of this feature.

The visible effect is at the perf output level: long-prompt runs on Mistral 7B will show much smaller KV cache, faster prefill (especially at very long context), and consistent decode attention work past the window. These flow through the existing roofline math without any UI changes.

Optional follow-up (deferred): a small caption next to the model selector showing `sliding window: 4096 tok` when a sliding-window model is selected, so users see why the numbers differ from a full-attention equivalent. Punt to a follow-up PR if useful.

## Testing

- **Helper unit test** — `effectiveAttentionLength(100, { type: 'full' })` returns 100; `(100, { type: 'sliding', window: 50 })` returns 50; `(30, { type: 'sliding', window: 50 })` returns 30.
- **Memory test** — extend `test/engine/memory.test.ts` with a synthetic model that has `attention: { type: 'sliding', window: W }`. Assert `kvCachePerRequest = kvPerToken × min(prompt+output, W)` against a hand-computed value.
- **Prefill test** — same fixture. Assert the attention term equals `2 × layers × prompt × min(prompt, W) × hidden`.
- **Decode test** — same fixture. Assert the attention term in `flopsPerStep` uses `min(avgSeqlen, W)`.
- **Regression** — all 27 existing engine fixture tests must continue to pass. They use the existing `testModel` which gains `attention: { type: 'full' }` as part of the retrofit. Behavior should be byte-identical for full-attention paths.
- **Integration** — one end-to-end test using the new Mistral 7B entry on H100 SXM-80 at a long prompt (e.g., 32768 tokens). Assert that `r.memory.kvCachePerRequest` is consistent with the window-bounded formula, not the full-attention formula. Sanity check rather than exact-value assertion.

Net test additions: ~5 cases on top of the current 33. All others must continue to pass unchanged.

## Evolution path

When MoE lands as feature #2: add an `architecture: 'dense' | 'moe'` axis (separate from `attention`), retrofit all current entries with `'dense'`, and `ModelArch` gains MoE-paired fields (active params, expert count, etc.). The attention axis is unaffected.

When MLA lands as feature #3: extend the `AttentionConfig` union with the `'mla'` variant and its paired latent-dim fields. DeepSeek-V2/V3 entries declare `attention: { type: 'mla', ... }`.

When hybrid lands as feature #4: extend the union with `'hybrid'` and a layer-pattern specification. Gemma 3's existing `attention: { type: 'full' }` (a known under-modeling — currently treated as full attention) is corrected to `'hybrid'` with the right pattern.

Each feature stacks on the same schema, with TS enforcing that the right paired fields accompany each attention type.

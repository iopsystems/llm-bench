# MLA Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Multi-head Latent Attention (MLA) as a new variant on `AttentionConfig`, with the KV-cache and attention-compute math branching to use MLA's small latent dimension. Add DeepSeek-V2 as the canonical user.

**Architecture:** New `'mla'` variant on the existing `AttentionConfig` discriminated union (no retrofit needed — existing variants stay valid). Two new helpers in `memory.ts` — `kvBytesPerToken(model, kvDtype)` and `attentionDim(model)` — that branch on attention type. Used by `memory.ts`, `prefill.ts`, and `decode.ts` to swap the GQA-style formula for the MLA-style one when applicable.

**Tech Stack:** TypeScript, Svelte 5, Vitest. No new deps.

**Spec:** `calc/docs/superpowers/specs/2026-05-12-mla-design.md`

---

## File Structure

```
src/engine/
  types.ts        # add `'mla'` variant to AttentionConfig
  memory.ts       # add kvBytesPerToken + attentionDim helpers; use kvBytesPerToken
  prefill.ts      # use attentionDim in attention FLOPs term
  decode.ts       # use attentionDim in attention FLOPs term
src/data/
  models.ts       # add DeepSeek-V2 entry
test/engine/
  sliding.test.ts # extend with kvBytesPerToken + attentionDim helper tests
  memory.test.ts  # add MLA KV cache test
  prefill.test.ts # add MLA attention term test
  decode.test.ts  # add MLA attention term test
  calc.test.ts    # add DeepSeek-V2 integration test
```

---

## Task 1: Schema (add 'mla' variant)

Just one type change — no retrofit. Existing models declare `{ type: 'full' }` or `{ type: 'sliding', ... }`, both of which remain valid when a new variant joins the union.

**Files:**
- Modify: `calc/src/engine/types.ts`

- [ ] **Step 1: Add the MLA variant to AttentionConfig**

Open `calc/src/engine/types.ts`. Locate the existing `AttentionConfig` type:

```ts
export type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
```

Replace with:

```ts
export type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; kvLoraRank: number; qkRopeHeadDim: number }
```

- [ ] **Step 2: Verify compile and tests still pass**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. All 47 existing tests PASS (no model uses the new variant, so behavior is unchanged).

- [ ] **Step 3: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/types.ts
git commit -m "feat(calc): add 'mla' variant to AttentionConfig"
```

No Co-Authored-By footer.

---

## Task 2: Helpers — kvBytesPerToken + attentionDim (TDD)

Both helpers live in `memory.ts` next to the existing `effectiveAttentionLength` / `activeParams`. Both branch on attention type.

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/sliding.test.ts`

- [ ] **Step 1: Append failing tests to `test/engine/sliding.test.ts`**

The file currently imports `effectiveAttentionLength, activeParams`. Replace that import with:

```ts
import {
  effectiveAttentionLength,
  activeParams,
  kvBytesPerToken,
  attentionDim
} from '../../src/engine/memory'
```

At the bottom of the file (after the existing closing `})` of the last describe block), append:

```ts
describe('kvBytesPerToken', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('GQA / full attention: 2 × layers × kv_heads × head_dim × bytes', () => {
    // 2 × 4 × 2 × 8 × 2 (fp16) = 256
    expect(kvBytesPerToken(base, 'fp16')).toBe(256)
  })

  it('sliding window uses same GQA formula', () => {
    const sliding: ModelArch = {
      ...base,
      attention: { type: 'sliding', window: 50 }
    }
    expect(kvBytesPerToken(sliding, 'fp16')).toBe(256)
  })

  it('MLA: layers × (kv_lora + rope) × bytes (no factor of 2)', () => {
    const mla: ModelArch = {
      ...base,
      attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8 }
    }
    // 4 × (32 + 8) × 2 = 320
    expect(kvBytesPerToken(mla, 'fp16')).toBe(320)
  })
})

describe('attentionDim', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('returns numHeads × headDim for full attention', () => {
    // 8 × 8 = 64 (intentionally different from hiddenDim 16, so the test
    // would fail if the helper accidentally returned hiddenDim — c.f. PR #91)
    expect(attentionDim(base)).toBe(64)
  })

  it('returns numHeads × headDim for sliding window', () => {
    const sliding: ModelArch = {
      ...base,
      attention: { type: 'sliding', window: 50 }
    }
    expect(attentionDim(sliding)).toBe(64)
  })

  it('returns kvLoraRank + qkRopeHeadDim for MLA', () => {
    const mla: ModelArch = {
      ...base,
      attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8 }
    }
    expect(attentionDim(mla)).toBe(40)
  })
})
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/sliding.test.ts
```

Expected: 6 new tests FAIL (`kvBytesPerToken is not a function`, `attentionDim is not a function`); existing helper tests still PASS.

- [ ] **Step 3: Implement both helpers in `src/engine/memory.ts`**

First, the existing types import is:
`import type { AttentionConfig, CalcInput, GpuVariant, MemoryResult, ModelArch } from './types'`

Extend with `Dtype`:
`import type { AttentionConfig, CalcInput, Dtype, GpuVariant, MemoryResult, ModelArch } from './types'`

Then, just after the existing `activeParams` function, add:

```ts
export function kvBytesPerToken(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla') {
    return model.layers * (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  return 2 * model.layers * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

export function attentionDim(model: ModelArch): number {
  const att = model.attention
  if (att.type === 'mla') return att.kvLoraRank + att.qkRopeHeadDim
  return model.numHeads * model.headDim
}
```

The existing `bytesOf` import is already in `memory.ts` (used by `computeMemory` for weights). No new import needed beyond `Dtype`.

- [ ] **Step 4: Verify pass**

```bash
npx vitest run test/engine/sliding.test.ts
```

Expected: 12 tests PASS in `sliding.test.ts` (6 existing + 6 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/sliding.test.ts
git commit -m "feat(calc): add kvBytesPerToken and attentionDim helpers for MLA"
```

---

## Task 3: Memory KV cache uses kvBytesPerToken

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/memory.test.ts`

- [ ] **Step 1: Append failing test to `test/engine/memory.test.ts`**

Inside the existing `describe('computeMemory', ...)` block, append before the closing `})`:

```ts
  it('kvCachePerRequest uses MLA formula for MLA models', () => {
    // testModel: layers=2, prompt+output=15.
    // MLA with kvLoraRank=10, rope=2: layers × (10+2) × 2 (fp16) = 48 bytes/token.
    // × 15 tokens = 720 bytes per request.
    // × concurrency 2 = 1440 bytes total.
    const mlaModel = {
      ...testInput.model,
      attention: { type: 'mla' as const, kvLoraRank: 10, qkRopeHeadDim: 2 }
    }
    const input = { ...testInput, model: mlaModel }
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(720)
    expect(m.kvCacheTotal).toBe(1440)
  })
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/memory.test.ts
```

Expected: new test FAILS — `kvCachePerRequest` still uses the inline GQA formula (2 × 2 × 1 × 2 × 2 = 16 bytes/token × 15 = 240). The 9 existing memory tests still PASS.

- [ ] **Step 3: Update `computeMemory` to use `kvBytesPerToken`**

In `calc/src/engine/memory.ts`, find the current `computeMemory` body. Locate the two lines:

```ts
const kvPerTokenPerRequest =
  2 * model.layers * model.numKvHeads * model.headDim * bytesOf(quant.kv)
const effSeqlen = effectiveAttentionLength(seqlen, model.attention)
const kvCachePerRequest = kvPerTokenPerRequest * effSeqlen
```

Replace with:

```ts
const kvPerTokenPerRequest = kvBytesPerToken(model, quant.kv)
const effSeqlen = effectiveAttentionLength(seqlen, model.attention)
const kvCachePerRequest = kvPerTokenPerRequest * effSeqlen
```

(Just one line changes. The variable `kvPerTokenPerRequest` is reassigned to the helper's return; everything downstream is unchanged because the helper's GQA branch returns the exact same value as the old inline formula.)

- [ ] **Step 4: Verify pass**

```bash
npx vitest run test/engine/memory.test.ts
npm test
```

Expected: 10 memory tests PASS (9 existing + 1 new). Full suite: 53 total (47 baseline + 6 helper from Task 2).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/memory.test.ts
git commit -m "feat(calc): route KV cache through kvBytesPerToken for MLA support"
```

---

## Task 4: Prefill attention term uses attentionDim

**Files:**
- Modify: `calc/src/engine/prefill.ts`
- Modify: `calc/test/engine/prefill.test.ts`

- [ ] **Step 1: Append failing test to `test/engine/prefill.test.ts`**

Inside the existing `describe('computePrefill', ...)` block, append:

```ts
  it('attention term uses attentionDim for MLA (kv_lora + rope, not hidden)', () => {
    // testModel: layers=2, hiddenDim=4, prompt=10, paramCount=1000.
    // MLA with kvLoraRank=10, rope=2 → attentionDim = 12 (vs hidden 4).
    // MLP: 2 × 1000 × 10 = 20000
    // Attention: 2 × 2 × 10 × 10 × 12 = 4800 (full attention, no sliding bound)
    // Total = 24800
    const mlaModel = {
      ...testInput.model,
      attention: { type: 'mla' as const, kvLoraRank: 10, qkRopeHeadDim: 2 }
    }
    const input = { ...testInput, model: mlaModel }
    const mlaMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, mlaMemory)
    expect(p.flops).toBe(24800)
  })
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/prefill.test.ts
```

Expected: new test FAILS (flops reports `2×1000×10 + 2×2×10×10×4 = 21600` — still using `hiddenDim` 4 instead of attentionDim 12); 6 existing prefill tests still PASS.

- [ ] **Step 3: Update `computePrefill` in `src/engine/prefill.ts`**

Extend the existing import from `./memory` to include `attentionDim`:

`import { effectiveAttentionLength, activeParams, attentionDim } from './memory'`

Find the existing flops calculation (note: post-#91 the attention term uses `numHeads × headDim`, not `hiddenDim`):

```ts
const effP = effectiveAttentionLength(p, model.attention)
const flops =
  2 * activeParams(model) * p +
  2 * model.layers * p * effP * model.numHeads * model.headDim
```

Replace `model.numHeads * model.headDim` (only in the attention term, the second line) with `attentionDim(model)`:

```ts
const effP = effectiveAttentionLength(p, model.attention)
const flops =
  2 * activeParams(model) * p +
  2 * model.layers * p * effP * attentionDim(model)
```

- [ ] **Step 4: Verify pass**

```bash
npx vitest run test/engine/prefill.test.ts
npm test
```

Expected: 7 prefill tests PASS. Full suite: 54 total.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/prefill.ts calc/test/engine/prefill.test.ts
git commit -m "feat(calc): use attentionDim in prefill attention FLOPs for MLA"
```

---

## Task 5: Decode attention term uses attentionDim

**Files:**
- Modify: `calc/src/engine/decode.ts`
- Modify: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Append failing test to `test/engine/decode.test.ts`**

Inside the existing `describe('computeDecode', ...)` block, append:

```ts
  it('attention term uses attentionDim for MLA (kv_lora + rope, not hidden)', () => {
    // testModel: layers=2, hiddenDim=4, paramCount=1000, concurrency=2.
    // avgSeqlen = 10 + 5/2 = 12.5.
    // MLA with kvLoraRank=10, rope=2 → attentionDim = 12 (vs hidden 4).
    // flopsPerStep = (2×1000 + 2×2×12.5×12) × 2 = (2000 + 600) × 2 = 5200
    const mlaModel = {
      ...testInput.model,
      attention: { type: 'mla' as const, kvLoraRank: 10, qkRopeHeadDim: 2 }
    }
    const input = { ...testInput, model: mlaModel }
    const mlaMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, mlaMemory)
    expect(d.flopsPerStep).toBe(5200)
  })
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/decode.test.ts
```

Expected: new test FAILS (flopsPerStep reports `(2×1000 + 2×2×12.5×4) × 2 = 4400` — still using `hiddenDim`); 7 existing decode tests still PASS.

- [ ] **Step 3: Update `computeDecode` in `src/engine/decode.ts`**

Extend the import from `./memory`:

`import { effectiveAttentionLength, activeParams, attentionDim } from './memory'`

Find the existing flopsPerStep calculation (post-#91 it uses `numHeads × headDim`, not `hiddenDim`):

```ts
const effAvg = effectiveAttentionLength(avgSeqlen, model.attention)
const flopsPerStep =
  (2 * activeParams(model) + 2 * model.layers * effAvg * model.numHeads * model.headDim) *
  workload.concurrency
```

Replace `model.numHeads * model.headDim` with `attentionDim(model)`:

```ts
const effAvg = effectiveAttentionLength(avgSeqlen, model.attention)
const flopsPerStep =
  (2 * activeParams(model) + 2 * model.layers * effAvg * attentionDim(model)) *
  workload.concurrency
```

(`bytesPerStep` already uses `activeParams × bytesOf(quant.weights)` from the MoE feature — no changes needed there.)

Note: the synthetic test fixture's `testModel` has `numHeads × headDim = 2 × 2 = 4 = hiddenDim`, so existing decode regression tests are unaffected by PR #91's change.

- [ ] **Step 4: Verify pass**

```bash
npx vitest run test/engine/decode.test.ts
npm test
```

Expected: 8 decode tests PASS. Full suite: 55 total.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/decode.ts calc/test/engine/decode.test.ts
git commit -m "feat(calc): use attentionDim in decode attention FLOPs for MLA"
```

---

## Task 6: Add DeepSeek-V2 (verified) and integration test

**Files:**
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/engine/calc.test.ts`

- [ ] **Step 1: Verify DeepSeek-V2 config via WebFetch**

Use WebFetch on `https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json` with prompt:

> Return the raw JSON contents. Specifically: num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, vocab_size, n_routed_experts, n_shared_experts, num_experts_per_tok.

If WebFetch fails (404, auth wall, etc.), STOP and report BLOCKED — do not paste values from memory.

Expected values:
- num_hidden_layers: 60
- hidden_size: 5120
- intermediate_size: 12288
- num_attention_heads: 128
- kv_lora_rank: 512
- qk_rope_head_dim: 64
- qk_nope_head_dim: 128
- v_head_dim: 128
- vocab_size: 102400
- n_routed_experts: 160
- n_shared_experts: 2
- num_experts_per_tok: 6

If any value differs, **use the config's value**. `paramCount: 236_000_000_000` and `activeParamCount: 21_000_000_000` come from the model card / safetensors metadata — accept as-is.

- [ ] **Step 2: Add DeepSeek-V2 entry to `src/data/models.ts`**

Insert a new section after the Mistral block, before Phi. Add this new family heading comment block and the entry:

```ts
  // === DeepSeek ===
  // DeepSeek-V2 has 2 shared experts always active in addition to 6 routed
  // experts per token. The current schema doesn't have a numSharedExperts
  // field (deferred to a later feature) — the activeParamCount value below
  // is from the model card and already includes the shared-expert
  // contribution, so compute math comes out correctly.
  {
    id: 'deepseek-v2', name: 'DeepSeek-V2', family: 'deepseek',
    layers: 60, hiddenDim: 5120, intermediateDim: 12288,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 102400,
    paramCount: 236_000_000_000,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 },
    architecture: {
      type: 'moe',
      numExperts: 160,
      numExpertsActive: 6,
      activeParamCount: 21_000_000_000
    }
  },
```

The `headDim: 192` value is `qk_nope_head_dim + qk_rope_head_dim` (combined Q-head dim). It's informational only for MLA — the math uses `kvLoraRank + qkRopeHeadDim` via `attentionDim`.

Adjust any field if Step 1 surfaced a discrepancy.

- [ ] **Step 3: Add integration test to `test/engine/calc.test.ts`**

At the bottom of the file (after the last existing `describe` block), append:

```ts
describe('calculate — MLA integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const dsv2 = MODELS.find(m => m.id === 'deepseek-v2')!

  it('DeepSeek-V2 at 32k prompt: KV cache uses MLA latent formula', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: dsv2,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // MLA KV per token = layers × (kv_lora + rope) × bytes(fp16)
    //                  = 60 × (512 + 64) × 2 = 69_120 bytes per token
    // × 32768 tokens = 2_264_924_160 bytes per request
    expect(r.memory.kvCachePerRequest).toBe(60 * (512 + 64) * 2 * 32768)

    // Sanity vs the GQA equivalent that would apply if attention.type were
    // 'full': 2 × 60 × 128 × 192 × 2 × 32768 = ~120× larger.
    const gqaEquivalent = 2 * 60 * 128 * 192 * 2 * 32768
    expect(gqaEquivalent / r.memory.kvCachePerRequest).toBeGreaterThan(100)
  })
})
```

- [ ] **Step 4: Run all checks**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. 56 total tests pass (55 + 1 new integration).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/data/models.ts calc/test/engine/calc.test.ts
git commit -m "feat(calc): add DeepSeek-V2 (MLA + MoE) with integration test"
```

---

## Self-Review Notes

Spec coverage:

- **Schema (Section "Schema")** — Task 1 adds `'mla'` variant
- **Helpers (`kvBytesPerToken`, `attentionDim`)** — Task 2
- **KV cache (Section "Math" subsection 1)** — Task 3
- **Prefill attention (Section "Math" subsection 2)** — Task 4
- **Decode attention (Section "Math" subsection 3)** — Task 5
- **Data update (Section "Data")** — Task 6
- **Footnote / full math (Section "Math" subsection "Footnote")** — documented in spec; intentionally not implemented
- **Testing (Section "Testing")** — distributed:
  - Helper tests: Task 2 (`kvBytesPerToken` and `attentionDim` in `sliding.test.ts`)
  - Memory: Task 3
  - Prefill: Task 4
  - Decode: Task 5
  - Integration with DeepSeek-V2: Task 6
  - Regression of 47 baseline tests: continuous (`npm test` after each task)
- **UI (no required changes)** — no task needed
- **Evolution path** — addressed by the discriminated-union schema ready for `numSharedExperts` on the MoE variant, the `'hybrid'` variant on AttentionConfig, and `qkNopeHeadDim` / `vHeadDim` on the MLA variant for the full-math refinement

Type / API consistency check:

- `AttentionConfig`'s new `'mla'` variant defined in Task 1, consumed unchanged in Tasks 2–6
- `kvBytesPerToken(model: ModelArch, kvDtype: Dtype): number` defined in Task 2, called in Task 3 (`kvBytesPerToken(model, quant.kv)`)
- `attentionDim(model: ModelArch): number` defined in Task 2, called in Tasks 4 and 5 (`attentionDim(model)`)
- Existing helpers (`effectiveAttentionLength`, `activeParams`) and their use sites unchanged

No placeholders. Every code block contains the exact text to write.

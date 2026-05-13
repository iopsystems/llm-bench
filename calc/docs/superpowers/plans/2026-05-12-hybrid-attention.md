# Hybrid Attention Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hybrid attention (Gemma 3 pattern — sliding-window layers interleaved with full / global layers) as a new variant on `AttentionConfig`, with the math helpers refactored so the `layers ×` factor lives inside a single layer-aggregating helper. Correct the existing Gemma 3 12B / 27B entries from the under-modeled `{ type: 'full' }` to `{ type: 'hybrid', ... }`.

**Architecture:** New `'hybrid'` variant on the existing `AttentionConfig` discriminated union. A new `attendedSeqlenSummedOverLayers(model, seqlen)` helper in `memory.ts` returns Σ over all layers of "effective seqlen for that layer" — for uniform-attention models it equals `model.layers × effLen_per_layer` (behavior-identical to today). The existing `kvBytesPerToken` is renamed to `kvBytesPerTokenPerLayer` (drops the `layers ×` factor, which now lives inside `attendedSeqlenSummedOverLayers`). `effectiveAttentionLength` is removed (no remaining callers). `attentionDim` is unchanged.

**Tech Stack:** TypeScript, Svelte 5, Vitest. No new deps.

**Spec:** `calc/docs/superpowers/specs/2026-05-12-hybrid-attention-design.md`

---

## File Structure

```
src/engine/
  types.ts        # add 'hybrid' variant to AttentionConfig
  memory.ts       # rename kvBytesPerToken → kvBytesPerTokenPerLayer (drop layers factor);
                  # add attendedSeqlenSummedOverLayers helper; remove effectiveAttentionLength
  prefill.ts      # use attendedSeqlenSummedOverLayers; drop explicit model.layers factor
  decode.ts       # use attendedSeqlenSummedOverLayers; drop explicit model.layers factor
src/data/
  models.ts       # correct Gemma 3 12B and 27B entries to hybrid attention
test/engine/
  sliding.test.ts # rename kvBytesPerToken describe block, update fixture math;
                  # add attendedSeqlenSummedOverLayers tests; delete effectiveAttentionLength tests
  prefill.test.ts # add hybrid attention term test
  decode.test.ts  # add hybrid attention term test
  memory.test.ts  # add hybrid KV cache test
  calc.test.ts    # add Gemma 3 27B integration test
```

All current 57 tests stay green throughout (the helper refactor is behavior-preserving for uniform-attention models, which is what every test fixture uses today).

---

## Task 1: Schema (add 'hybrid' variant)

Type-only change. Existing models declare `{ type: 'full' }` / `'sliding'` / `'mla'`, all still valid when the union grows.

**Files:**
- Modify: `calc/src/engine/types.ts`

- [ ] **Step 1: Add the hybrid variant to AttentionConfig**

Open `calc/src/engine/types.ts`. Locate the existing `AttentionConfig` type (lines 31-34):

```ts
export type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; kvLoraRank: number; qkRopeHeadDim: number }
```

Replace with:

```ts
export type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; kvLoraRank: number; qkRopeHeadDim: number }
  | { type: 'hybrid'; slidingWindow: number; numSlidingLayers: number; numGlobalLayers: number }
```

- [ ] **Step 2: Verify compile and tests still pass**

Run:

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. All 57 existing tests PASS (no model uses the new variant, so behavior is unchanged).

- [ ] **Step 3: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/types.ts
git commit -m "feat(calc): add 'hybrid' variant to AttentionConfig"
```

No Co-Authored-By footer.

---

## Task 2: Helper — attendedSeqlenSummedOverLayers (TDD)

Adds the new layer-aggregating helper alongside the existing helpers. Branches on attention type with all four cases. Does NOT yet replace any caller — that's Tasks 3 / 4.

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/sliding.test.ts`

- [ ] **Step 1: Append failing tests to `test/engine/sliding.test.ts`**

Update the import line at the top (currently line 2):

```ts
import {
  effectiveAttentionLength,
  activeParams,
  kvBytesPerToken,
  attentionDim,
  attendedSeqlenSummedOverLayers
} from '../../src/engine/memory'
```

Append the following `describe` block at the bottom of the file (after the closing of the `attentionDim` block):

```ts
describe('attendedSeqlenSummedOverLayers', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('full: layers × seqlen', () => {
    expect(attendedSeqlenSummedOverLayers(base, 100)).toBe(400) // 4 × 100
    expect(attendedSeqlenSummedOverLayers(base, 0)).toBe(0)
  })

  it('sliding: layers × min(seqlen, window)', () => {
    const m: ModelArch = { ...base, attention: { type: 'sliding', window: 50 } }
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(200) // 4 × 50
    expect(attendedSeqlenSummedOverLayers(m, 30)).toBe(120)  // 4 × 30
    expect(attendedSeqlenSummedOverLayers(m, 50)).toBe(200)  // 4 × 50
  })

  it('mla: layers × seqlen (dimensional reduction is in attentionDim, not seqlen)', () => {
    const m: ModelArch = { ...base, attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8 } }
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(400) // 4 × 100
  })

  it('hybrid: numSliding × min(seqlen, window) + numGlobal × seqlen', () => {
    const m: ModelArch = {
      ...base,
      layers: 6,
      attention: {
        type: 'hybrid', slidingWindow: 50,
        numSlidingLayers: 5, numGlobalLayers: 1
      }
    }
    // seqlen > window: 5 × min(100, 50) + 1 × 100 = 250 + 100 = 350
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(350)
    // seqlen < window: 5 × 30 + 1 × 30 = 180 (sliding cap inactive)
    expect(attendedSeqlenSummedOverLayers(m, 30)).toBe(180)
    // seqlen == window: 5 × 50 + 1 × 50 = 300
    expect(attendedSeqlenSummedOverLayers(m, 50)).toBe(300)
  })

  it('hybrid: throws when numSlidingLayers + numGlobalLayers ≠ model.layers', () => {
    const m: ModelArch = {
      ...base,
      layers: 6,
      attention: {
        type: 'hybrid', slidingWindow: 50,
        numSlidingLayers: 4, numGlobalLayers: 1  // 4+1=5 ≠ 6
      }
    }
    expect(() => attendedSeqlenSummedOverLayers(m, 100)).toThrow(/sum to model\.layers/)
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- test/engine/sliding.test.ts
```

Expected: 5 new tests FAIL with import / reference error for `attendedSeqlenSummedOverLayers`. All existing tests still pass.

- [ ] **Step 3: Add the helper to memory.ts**

Open `calc/src/engine/memory.ts`. After the existing `attentionDim` export (currently ends at line 29), add a new export:

```ts
export function attendedSeqlenSummedOverLayers(model: ModelArch, seqlen: number): number {
  const att = model.attention
  if (att.type === 'hybrid') {
    if (att.numSlidingLayers + att.numGlobalLayers !== model.layers) {
      throw new Error(
        `hybrid layer counts must sum to model.layers: ` +
        `${att.numSlidingLayers} + ${att.numGlobalLayers} ≠ ${model.layers}`
      )
    }
    return att.numSlidingLayers * Math.min(seqlen, att.slidingWindow)
         + att.numGlobalLayers * seqlen
  }
  const perLayer = att.type === 'sliding' ? Math.min(seqlen, att.window) : seqlen
  return model.layers * perLayer
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: all 62 tests PASS (57 existing + 5 new). Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/sliding.test.ts
git commit -m "feat(calc): add attendedSeqlenSummedOverLayers helper for hybrid attention"
```

No Co-Authored-By footer.

---

## Task 3: Refactor — kvBytesPerToken → kvBytesPerTokenPerLayer + wire computeMemory

Pulls the `layers ×` factor out of `kvBytesPerToken` (renaming it `kvBytesPerTokenPerLayer`) so the layer factor lives in exactly one place — inside `attendedSeqlenSummedOverLayers`. `computeMemory` becomes `kvBytesPerTokenPerLayer × attendedSeqlenSummedOverLayers`, which is byte-identical to today for uniform-attention models.

Math check (testModel: layers=2, kvHeads=1, headDim=2, fp16, seqlen=15):
- Old: `kvBytesPerToken × effSeq = (2 × 2 × 1 × 2 × 2) × 15 = 16 × 15 = 240` ✓
- New: `kvBytesPerTokenPerLayer × attendedSeq = (2 × 1 × 2 × 2) × (2 × 15) = 8 × 30 = 240` ✓

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/sliding.test.ts`

- [ ] **Step 1: Update `kvBytesPerToken` describe block in `test/engine/sliding.test.ts`**

Update the import (currently includes `kvBytesPerToken`) to use the new name:

```ts
import {
  effectiveAttentionLength,
  activeParams,
  kvBytesPerTokenPerLayer,
  attentionDim,
  attendedSeqlenSummedOverLayers
} from '../../src/engine/memory'
```

Replace the entire `describe('kvBytesPerToken', ...)` block (currently lines 53-84) with:

```ts
describe('kvBytesPerTokenPerLayer', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('GQA / full attention: 2 × kv_heads × head_dim × bytes (no layers factor)', () => {
    // 2 × 2 × 8 × 2 (fp16) = 64
    expect(kvBytesPerTokenPerLayer(base, 'fp16')).toBe(64)
  })

  it('sliding window uses same GQA formula', () => {
    const sliding: ModelArch = {
      ...base,
      attention: { type: 'sliding', window: 50 }
    }
    expect(kvBytesPerTokenPerLayer(sliding, 'fp16')).toBe(64)
  })

  it('MLA: (kv_lora + rope) × bytes (no factor of 2, no layers factor)', () => {
    const mla: ModelArch = {
      ...base,
      attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8 }
    }
    // (32 + 8) × 2 = 80
    expect(kvBytesPerTokenPerLayer(mla, 'fp16')).toBe(80)
  })
})
```

- [ ] **Step 2: Run tests to confirm helper-rename failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- test/engine/sliding.test.ts
```

Expected: the renamed `kvBytesPerTokenPerLayer` import fails to resolve, 3 helper tests FAIL.

- [ ] **Step 3: Rename helper + drop layers factor in `memory.ts`**

In `calc/src/engine/memory.ts`, replace the existing `kvBytesPerToken` export (currently lines 17-23):

```ts
export function kvBytesPerToken(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla') {
    return model.layers * (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  return 2 * model.layers * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}
```

With:

```ts
export function kvBytesPerTokenPerLayer(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}
```

- [ ] **Step 4: Update `computeMemory` to use both new helpers**

In `calc/src/engine/memory.ts`, locate the KV section inside `computeMemory` (currently lines 43-45):

```ts
const kvPerTokenPerRequest = kvBytesPerToken(model, quant.kv)
const effSeqlen = effectiveAttentionLength(seqlen, model.attention)
const kvCachePerRequest = kvPerTokenPerRequest * effSeqlen
```

Replace with:

```ts
const kvPerLayerPerToken = kvBytesPerTokenPerLayer(model, quant.kv)
const attendedSeqlen = attendedSeqlenSummedOverLayers(model, seqlen)
const kvCachePerRequest = kvPerLayerPerToken * attendedSeqlen
```

- [ ] **Step 5: Run tests to verify everything still passes**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: all 62 tests PASS. Type-check clean. `memory.test.ts` assertions (240 / 480 / 128 / 720 / 1440 for the four KV-cache cases) are unchanged because the new formulation is mathematically identical for uniform-attention models.

- [ ] **Step 6: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/sliding.test.ts
git commit -m "refactor(calc): rename kvBytesPerToken → kvBytesPerTokenPerLayer; wire computeMemory through layer-summed helper"
```

No Co-Authored-By footer.

---

## Task 4: Refactor prefill / decode + remove effectiveAttentionLength

`prefill.ts` and `decode.ts` currently spell out `model.layers × effectiveAttentionLength(...)`. Replace both with `attendedSeqlenSummedOverLayers(...)`. Once both callers are migrated, `effectiveAttentionLength` has zero callers — delete it and its tests.

Math check (testModel: layers=2, prompt=10, full attention):
- Old: `2 × model.layers × p × effP × attentionDim = 2 × 2 × 10 × 10 × 4 = 1600` (attention term in prefill flops)
- New: `2 × p × attendedSeqlenSummedOverLayers(model, p) × attentionDim = 2 × 10 × (2 × 10) × 4 = 1600` ✓

**Files:**
- Modify: `calc/src/engine/prefill.ts`
- Modify: `calc/src/engine/decode.ts`
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/sliding.test.ts`

- [ ] **Step 1: Update prefill.ts**

In `calc/src/engine/prefill.ts`:

Change the import line (currently line 3):

```ts
import { attendedSeqlenSummedOverLayers, activeParams, attentionDim } from './memory'
```

Replace the body of `computePrefill` flops computation (currently lines 13-16):

```ts
const effP = effectiveAttentionLength(p, model.attention)
const flops =
  2 * activeParams(model) * p +
  2 * model.layers * p * effP * attentionDim(model)
```

With:

```ts
const flops =
  2 * activeParams(model) * p +
  2 * p * attendedSeqlenSummedOverLayers(model, p) * attentionDim(model)
```

- [ ] **Step 2: Update decode.ts**

In `calc/src/engine/decode.ts`:

Change the import line (currently line 3):

```ts
import { attendedSeqlenSummedOverLayers, activeParams, attentionDim } from './memory'
```

Replace the body of `computeDecode` flops computation (currently lines 14-17):

```ts
const effAvg = effectiveAttentionLength(avgSeqlen, model.attention)
const flopsPerStep =
  (2 * activeParams(model) + 2 * model.layers * effAvg * attentionDim(model)) *
  workload.concurrency
```

With:

```ts
const flopsPerStep =
  (2 * activeParams(model) + 2 * attendedSeqlenSummedOverLayers(model, avgSeqlen) * attentionDim(model)) *
  workload.concurrency
```

- [ ] **Step 3: Remove effectiveAttentionLength from memory.ts**

In `calc/src/engine/memory.ts`, delete the export (currently lines 6-9):

```ts
export function effectiveAttentionLength(rawSeqlen: number, attention: AttentionConfig): number {
  if (attention.type === 'sliding') return Math.min(rawSeqlen, attention.window)
  return rawSeqlen
}
```

Also remove `AttentionConfig` from the type-import line at the top of the file if no other use remains — check by searching the file:

```bash
cd /Users/yao/workspace/llm-perf/calc
grep -n AttentionConfig src/engine/memory.ts
```

If no other references appear, drop `AttentionConfig` from the `import type` line.

- [ ] **Step 4: Remove the effectiveAttentionLength describe block from sliding.test.ts**

Delete the `describe('effectiveAttentionLength', ...)` block (currently lines 5-22). Also remove `effectiveAttentionLength` from the import at the top of the file:

```ts
import {
  activeParams,
  kvBytesPerTokenPerLayer,
  attentionDim,
  attendedSeqlenSummedOverLayers
} from '../../src/engine/memory'
```

- [ ] **Step 5: Run tests and type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: 58 tests PASS (62 from after Task 2, minus the 4 `effectiveAttentionLength` tests just deleted). Type-check clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/prefill.ts calc/src/engine/decode.ts calc/src/engine/memory.ts calc/test/engine/sliding.test.ts
git commit -m "refactor(calc): use attendedSeqlenSummedOverLayers in prefill/decode; remove effectiveAttentionLength"
```

No Co-Authored-By footer.

---

## Task 5: Wire hybrid attention into prefill / decode / memory (TDD)

Add three unit tests (one each for memory, prefill, decode) using a synthetic hybrid fixture. Because `attendedSeqlenSummedOverLayers` already handles the hybrid case (added in Task 2) and all three call sites already route through it (updated in Tasks 3 and 4), these tests SHOULD pass without any code changes — they are *regression locks* that confirm the wiring is correct.

If any of them fails on first run, that's a wiring bug to fix before continuing.

**Files:**
- Modify: `calc/test/engine/memory.test.ts`
- Modify: `calc/test/engine/prefill.test.ts`
- Modify: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Append the hybrid KV-cache test to `memory.test.ts`**

Append inside the `describe('computeMemory', ...)` block (just before its closing `})`):

```ts
it('kvCachePerRequest uses hybrid formula: numSliding × min(seq,W) + numGlobal × seq', () => {
  // testModel: layers=2, kvHeads=1, headDim=2, fp16 KV; prompt+output=15.
  // Hybrid with slidingWindow=5, numSlidingLayers=1, numGlobalLayers=1:
  //   per-layer KV bytes = 2 × 1 × 2 × 2 = 8
  //   attendedSeqlen = 1 × min(15, 5) + 1 × 15 = 5 + 15 = 20
  //   kvCachePerRequest = 8 × 20 = 160
  const hybridModel = {
    ...testInput.model,
    attention: {
      type: 'hybrid' as const,
      slidingWindow: 5,
      numSlidingLayers: 1,
      numGlobalLayers: 1
    }
  }
  const input = { ...testInput, model: hybridModel }
  const m = computeMemory(input)
  expect(m.kvCachePerRequest).toBe(160)
  // × concurrency 2 = 320 bytes total
  expect(m.kvCacheTotal).toBe(320)
})
```

- [ ] **Step 2: Append the hybrid prefill-flops test to `prefill.test.ts`**

Append inside the `describe('computePrefill', ...)` block (just before its closing `})`):

```ts
it('attention term uses hybrid formula in prefill flops', () => {
  // testModel: layers=2, hiddenDim=4, paramCount=1000, prompt=10.
  // numHeads=2, headDim=2 → attentionDim=4.
  // Hybrid with slidingWindow=5, numSlidingLayers=1, numGlobalLayers=1:
  //   attendedSeq(10) = 1 × min(10, 5) + 1 × 10 = 15
  //   MLP: 2 × 1000 × 10 = 20000
  //   Attention: 2 × 10 × 15 × 4 = 1200
  //   Total = 21200
  const hybridModel = {
    ...testInput.model,
    attention: {
      type: 'hybrid' as const,
      slidingWindow: 5,
      numSlidingLayers: 1,
      numGlobalLayers: 1
    }
  }
  const input = { ...testInput, model: hybridModel }
  const hybridMemory = computeMemory(input)
  const p = computePrefill(input, opPoint, hybridMemory)
  expect(p.flops).toBe(21200)
})
```

- [ ] **Step 3: Append the hybrid decode-flops test to `decode.test.ts`**

Append inside the `describe('computeDecode', ...)` block (just before its closing `})`):

```ts
it('attention term uses hybrid formula in decode flopsPerStep', () => {
  // testModel: layers=2, hiddenDim=4, paramCount=1000, concurrency=2.
  // numHeads=2, headDim=2 → attentionDim=4.
  // avgSeqlen = 10 + 5/2 = 12.5.
  // Hybrid with slidingWindow=5, numSlidingLayers=1, numGlobalLayers=1:
  //   attendedSeq(12.5) = 1 × min(12.5, 5) + 1 × 12.5 = 17.5
  //   flopsPerStep = (2×1000 + 2×17.5×4) × 2 = (2000 + 140) × 2 = 4280
  const hybridModel = {
    ...testInput.model,
    attention: {
      type: 'hybrid' as const,
      slidingWindow: 5,
      numSlidingLayers: 1,
      numGlobalLayers: 1
    }
  }
  const input = { ...testInput, model: hybridModel }
  const hybridMemory = computeMemory(input)
  const d = computeDecode(input, opPoint, hybridMemory)
  expect(d.flopsPerStep).toBe(4280)
})
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: all 61 tests PASS (58 from Task 4 + 3 new). Type-check clean.

If any of the three new tests fails, the wiring in Tasks 3 or 4 is incorrect — re-read the diff for that file and fix before committing.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/test/engine/memory.test.ts calc/test/engine/prefill.test.ts calc/test/engine/decode.test.ts
git commit -m "test(calc): hybrid attention wiring in memory, prefill, decode"
```

No Co-Authored-By footer.

---

## Task 6: Update Gemma 3 entries + integration test

Switch Gemma 3 12B and 27B from `{ type: 'full' }` to the correct hybrid configuration, then add an integration test that exercises the full calculation pipeline on Gemma 3 27B at long prompt.

**Files:**
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/engine/calc.test.ts`

- [ ] **Step 1: Update Gemma 3 27B entry**

In `calc/src/data/models.ts`, locate the gemma-3-27b entry (currently lines 73-80):

```ts
{
  id: 'gemma-3-27b', name: 'Gemma 3 27B', family: 'gemma-3',
  layers: 62, hiddenDim: 5376, intermediateDim: 21504,
  numHeads: 32, numKvHeads: 16, headDim: 128, vocabSize: 262144,
  paramCount: 27_009_000_000,
  attention: { type: 'full' },
  architecture: { type: 'dense' }
},
```

Replace the `attention` line with:

```ts
  attention: { type: 'hybrid', slidingWindow: 1024, numSlidingLayers: 52, numGlobalLayers: 10 },
```

Values verified against `unsloth/gemma-3-27b-it/config.json` (the Google original is gated): `sliding_window: 1024`, `sliding_window_pattern: 6`, `num_hidden_layers: 62`. With pattern=6 every 6th layer is global → 10 global, 52 sliding.

- [ ] **Step 2: Update Gemma 3 12B entry**

In `calc/src/data/models.ts`, locate the gemma-3-12b entry (currently lines 65-72):

```ts
{
  id: 'gemma-3-12b', name: 'Gemma 3 12B', family: 'gemma-3',
  layers: 48, hiddenDim: 3840, intermediateDim: 15360,
  numHeads: 16, numKvHeads: 8, headDim: 256, vocabSize: 262144,
  paramCount: 12_187_000_000,
  attention: { type: 'full' },
  architecture: { type: 'dense' }
},
```

Replace the `attention` line with:

```ts
  attention: { type: 'hybrid', slidingWindow: 1024, numSlidingLayers: 40, numGlobalLayers: 8 },
```

Verified against `unsloth/gemma-3-12b-it/config.json`: same window/pattern as 27B. With `num_hidden_layers: 48` and pattern=6: 8 global, 40 sliding.

- [ ] **Step 3: Append the integration test to `calc.test.ts`**

Append a new top-level describe block at the bottom of `calc/test/engine/calc.test.ts`:

```ts
describe('calculate — hybrid attention integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const gemma27b = MODELS.find(m => m.id === 'gemma-3-27b')!

  it('Gemma 3 27B at 8k prompt: KV cache uses hybrid formula (~3.8× smaller than full attention)', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: gemma27b,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Per-layer KV bytes per token = 2 × kvHeads × headDim × bytes(fp16)
    //                              = 2 × 16 × 128 × 2 = 8192
    // attendedSeq = 52 × min(8192, 1024) + 10 × 8192
    //             = 52 × 1024 + 10 × 8192
    //             = 53248 + 81920 = 135168
    // kvCachePerRequest = 8192 × 135168 = 1_107_296_256
    expect(r.memory.kvCachePerRequest).toBe(8192 * 135168)

    // Sanity vs the would-have-been full-attention value:
    //   2 × layers × kvHeads × headDim × bytes × seqlen
    // = 2 × 62 × 16 × 128 × 2 × 8192 = 8192 × (62 × 8192) = 8192 × 507_904
    const fullEquivalent = 8192 * 62 * 8192
    const ratio = fullEquivalent / r.memory.kvCachePerRequest
    expect(ratio).toBeGreaterThan(3.5)        // 3.76 actually
    expect(ratio).toBeLessThan(6.3)           // asymptote layers/numGlobal = 62/10 = 6.2
  })
})
```

- [ ] **Step 4: Run tests and type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: 62 tests PASS (61 from Task 5 + 1 integration). Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/data/models.ts calc/test/engine/calc.test.ts
git commit -m "feat(calc): correct Gemma 3 12B/27B to hybrid attention; add integration test"
```

No Co-Authored-By footer.

---

## Done

After Task 6:
- Branch `feat/calc-hybrid-attention` has 6 implementation commits on top of the spec commit
- 62 / 62 tests pass
- `npm run check` clean
- Gemma 3 12B and 27B now use `{ type: 'hybrid', ... }` and show realistic KV cache figures
- `effectiveAttentionLength` and the old `kvBytesPerToken` name are gone from the codebase
- Controller will offer to open the PR

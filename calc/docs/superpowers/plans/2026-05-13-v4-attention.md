# DeepSeek V4 Attention + MTP + fp4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `'csa-hca-hybrid'` attention variant, `numNextnLayers` required field on `ModelArch`, and `'fp4'` Dtype. Add DeepSeek V4-Pro and V4-Flash as canonical users.

**Architecture:** Pure schema + math additions. The new attention variant branches the three existing math helpers (`kvBytesPerTokenPerLayer`, `attentionDim`, `attendedSeqlenSummedOverLayers`). MTP is a model-level field that adjusts `decode.timePerTokenS` and `decode.aggregateTokensPerS` by `(1 + numNextnLayers)` — no impact on per-pass FLOPs/bytes. fp4 is a one-line Dtype addition for Blackwell-class GPUs.

**Tech Stack:** TypeScript, Svelte 5, Vitest. No new deps.

**Spec:** `calc/docs/superpowers/specs/2026-05-13-v4-attention-design.md`

---

## File Structure

```
src/engine/
  types.ts        # add 'csa-hca-hybrid' variant; 'fp4' to Dtype; numNextnLayers field on ModelArch
  dtypes.ts       # register fp4 in DTYPE_BYTES (0.5 bytes)
  memory.ts       # 'csa-hca-hybrid' branch in three helpers
  decode.ts       # MTP wire-up: divide timePerTokenS by (1 + numNextnLayers); scale aggregateTokensPerS
src/data/
  models.ts       # retrofit 22 entries with numNextnLayers: 0; add V4-Pro and V4-Flash entries
test/
  fixtures.ts     # testModel gets numNextnLayers: 0
test/engine/
  sliding.test.ts # 'base: ModelArch' fixtures get numNextnLayers: 0;
                  # add 'csa-hca-hybrid' branch tests for 3 helpers + invariant
  dtypes.test.ts  # new file: bytesOf('fp4') === 0.5 (1 trivial test)
  memory.test.ts  # add csa-hca-hybrid regression test
  prefill.test.ts # add csa-hca-hybrid regression test
  decode.test.ts  # add csa-hca-hybrid regression test + MTP-doubles-throughput test
  calc.test.ts    # add V4-Pro 1M integration + V4-Flash 128k integration + V4 MTP integration
```

---

## Task 1: Schema additions (types.ts + dtypes.ts)

Adds the three new schema atoms in one commit. After this task, every MoE/MLA entry will fail to compile because `numNextnLayers` is required.

**Files:**
- Modify: `calc/src/engine/types.ts`
- Modify: `calc/src/engine/dtypes.ts`
- Create: `calc/test/engine/dtypes.test.ts`

- [ ] **Step 1: Add `'fp4'` to the Dtype union in `types.ts`**

In `calc/src/engine/types.ts`, locate the `Dtype` type (currently first line of the file):

```ts
export type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4'
```

Replace with:

```ts
export type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'fp4' | 'int8' | 'int4'
```

- [ ] **Step 2: Add `'csa-hca-hybrid'` variant to AttentionConfig**

In `calc/src/engine/types.ts`, locate the `AttentionConfig` union. Append a new arm at the end:

```ts
  | { type: 'csa-hca-hybrid';
      // Layer counts (must sum to model.layers)
      numSlidingLayers: number;
      numCsaLayers: number;
      numHcaLayers: number;
      // Sliding-window size (applies to dedicated sliding layers AND
      // to the per-layer side-branch on CSA/HCA layers)
      slidingWindow: number;
      // CSA params
      csaCompressionM: number;
      csaTopK: number;
      csaIndexerHeads: number;
      csaIndexerHeadDim: number;
      // HCA params
      hcaCompressionM: number
    }
```

- [ ] **Step 3: Add `numNextnLayers` to `ModelArch`**

In `calc/src/engine/types.ts`, locate the `ModelArch` interface. Add a new line after `architecture: ArchitectureConfig`:

```ts
  numNextnLayers: number  // Multi-Token Prediction depth; 0 for non-MTP models
```

- [ ] **Step 4: Register `'fp4'` in `dtypes.ts`**

In `calc/src/engine/dtypes.ts`, locate the `DTYPE_BYTES` map and update:

```ts
const DTYPE_BYTES: Record<Dtype, number> = {
  fp32: 4, fp16: 2, bf16: 2, fp8: 1, fp4: 0.5, int8: 1, int4: 0.5
}
```

- [ ] **Step 5: Create `test/engine/dtypes.test.ts` with the fp4 unit test**

Create file `calc/test/engine/dtypes.test.ts` with content:

```ts
import { describe, it, expect } from 'vitest'
import { bytesOf } from '../../src/engine/dtypes'

describe('bytesOf', () => {
  it('returns 0.5 for fp4 (Blackwell native fp4 tensor cores)', () => {
    expect(bytesOf('fp4')).toBe(0.5)
  })

  it('returns 0.5 for int4 (same width as fp4 but distinct Dtype)', () => {
    expect(bytesOf('int4')).toBe(0.5)
  })

  it('returns 1 for fp8 and int8', () => {
    expect(bytesOf('fp8')).toBe(1)
    expect(bytesOf('int8')).toBe(1)
  })
})
```

- [ ] **Step 6: Run check; expect type errors for missing numNextnLayers**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
```

Expected: type-check FAILS with errors flagging every `ModelArch` literal that's missing `numNextnLayers`. The list is the 22 entries in `src/data/models.ts` plus a handful of test fixtures.

- [ ] **Step 7: Commit (with check still failing — Task 2 fixes the retrofit)**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/types.ts calc/src/engine/dtypes.ts calc/test/engine/dtypes.test.ts
git commit -m "feat(calc): schema atoms for V4 — 'csa-hca-hybrid', 'fp4', numNextnLayers"
```

Type-check intentionally fails post-commit. Task 2 retrofits and restores green build. The commit boundary keeps the schema-vs-retrofit changes isolated for git review.

No Co-Authored-By footer.

---

## Task 2: Retrofit `numNextnLayers: 0` across data + test fixtures

Add the new required field to every `ModelArch` literal in the codebase. Math doesn't read `numNextnLayers` when it's 0, so all existing assertions remain valid.

**Files:**
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/fixtures.ts`
- Modify: `calc/test/engine/sliding.test.ts`

- [ ] **Step 1: Run check to enumerate all missing-field sites**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check 2>&1 | grep "numNextnLayers" | head -40
```

Expected: list of `ModelArch` literals missing `numNextnLayers`. Most will be in `src/data/models.ts`, plus `test/fixtures.ts` and `test/engine/sliding.test.ts`.

- [ ] **Step 2: Add `numNextnLayers: 0` to every entry in `src/data/models.ts`**

Each of the 22 model entries gets an additional field. For each entry, add `numNextnLayers: 0,` on a new line, after `paramCount: ...,` and before the `attention:` line. The pattern:

```ts
{
  id: '...', name: '...', family: '...',
  layers: ..., hiddenDim: ..., intermediateDim: ...,
  numHeads: ..., numKvHeads: ..., headDim: ..., vocabSize: ...,
  paramCount: ...,
  numNextnLayers: 0,                  // <-- ADD THIS LINE
  attention: { ... },
  architecture: { ... }
}
```

Apply to all 22 existing entries. The implementer can use `sed` or repeated manual edits — TypeScript will confirm completeness when check passes.

- [ ] **Step 3: Add `numNextnLayers: 0` to `testModel` in `test/fixtures.ts`**

In `calc/test/fixtures.ts`, locate the `testModel` declaration. Add the field after `paramCount: 1000,`:

```ts
export const testModel: ModelArch = {
  id: 'test-model',
  name: 'Test Model',
  family: 'test',
  layers: 2,
  hiddenDim: 4,
  intermediateDim: 8,
  numHeads: 2,
  numKvHeads: 1,
  headDim: 2,
  vocabSize: 100,
  paramCount: 1000,
  numNextnLayers: 0,
  attention: { type: 'full' },
  architecture: { type: 'dense' }
}
```

- [ ] **Step 4: Add `numNextnLayers: 0` to every `ModelArch` declaration in `test/engine/sliding.test.ts`**

`sliding.test.ts` has several `const base: ModelArch = { ... }` declarations (one per describe block: `activeParams`, `kvBytesPerTokenPerLayer`, `attentionDim`, `attendedSeqlenSummedOverLayers`, plus the helpers for `linear-mla-hybrid`). Each gets `numNextnLayers: 0` added after `paramCount: 1000,`.

There's also a custom-layers fixture inside the `linear-mla-hybrid` describe block (`base.layers = 4`) — same treatment.

Specifically: search the file for `const base: ModelArch = {` and `const m: ModelArch = {` patterns. Every such literal gets `numNextnLayers: 0` added.

- [ ] **Step 5: Run check + tests**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. All 88+ existing tests PASS byte-for-byte (math doesn't read the new field when it's 0).

- [ ] **Step 6: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/data/models.ts calc/test/fixtures.ts calc/test/engine/sliding.test.ts
git commit -m "feat(calc): retrofit numNextnLayers: 0 across data + test fixtures"
```

No Co-Authored-By footer.

---

## Task 3: Helpers — `csa-hca-hybrid` branch in 3 existing helpers (TDD)

Three helpers each get one new branch. KV-storage vs attention-compute formulas differ for CSA layers (storage = `seqlen/m_csa`, compute = `topK`). Plus invariant check on layer counts.

**Files:**
- Modify: `calc/test/engine/sliding.test.ts`
- Modify: `calc/src/engine/memory.ts`

- [ ] **Step 1: Append failing tests to `test/engine/sliding.test.ts`**

Append the following at the bottom of the file (as a new top-level describe block):

```ts
describe('csa-hca-hybrid branches in existing helpers', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 3, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 1, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    numNextnLayers: 0,
    attention: {
      type: 'csa-hca-hybrid',
      numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 1,
      slidingWindow: 2,
      csaCompressionM: 2, csaTopK: 3,
      csaIndexerHeads: 2, csaIndexerHeadDim: 2,
      hcaCompressionM: 4
    },
    architecture: { type: 'dense' }
  }

  it('kvBytesPerTokenPerLayer: 2 × numKvHeads × headDim × bytes (MQA-style)', () => {
    // 2 × 1 × 8 × 2 (fp16) = 32
    expect(kvBytesPerTokenPerLayer(base, 'fp16')).toBe(32)
  })

  it('attentionDim: numHeads × headDim (full Q-head MQA)', () => {
    // 8 × 8 = 64
    expect(attentionDim(base)).toBe(64)
  })

  it('attendedSeqlenSummedOverLayers (forKv=true): storage formula', () => {
    // seqlen=20:
    // sliding contrib: 1 × min(20, 2) = 2
    // CSA contrib:     1 × (20/2 + 2) = 12
    // HCA contrib:     1 × (20/4 + 2) = 7
    // total = 21
    expect(attendedSeqlenSummedOverLayers(base, 20, true)).toBe(21)
  })

  it('attendedSeqlenSummedOverLayers (forKv=false default): compute formula', () => {
    // seqlen=20:
    // sliding contrib: 1 × min(20, 2) = 2
    // CSA contrib:     1 × (csaTopK=3 + 2) = 5    [topK used instead of seqlen/m_csa]
    // HCA contrib:     1 × (20/4 + 2) = 7
    // total = 14
    expect(attendedSeqlenSummedOverLayers(base, 20)).toBe(14)
  })

  it('attendedSeqlenSummedOverLayers throws when layer counts do not sum to model.layers', () => {
    const m: ModelArch = {
      ...base,
      attention: {
        type: 'csa-hca-hybrid',
        numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 2,  // 1+1+2=4 ≠ 3
        slidingWindow: 2,
        csaCompressionM: 2, csaTopK: 3,
        csaIndexerHeads: 2, csaIndexerHeadDim: 2,
        hcaCompressionM: 4
      }
    }
    expect(() => attendedSeqlenSummedOverLayers(m, 20)).toThrow(/sum to model\.layers/)
  })
})
```

- [ ] **Step 2: Run tests; expect 5 new failures**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- test/engine/sliding.test.ts
```

Expected: 5 new tests FAIL (helpers fall through wrong paths for the unknown variant).

- [ ] **Step 3: Add the `csa-hca-hybrid` branches in `memory.ts`**

In `calc/src/engine/memory.ts`:

Update `kvBytesPerTokenPerLayer` — add the new branch (placement: after the existing `mla | mla-dsa | linear-mla-hybrid` branches, before the GQA fallthrough):

```ts
export function kvBytesPerTokenPerLayer(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  if (att.type === 'linear-mla-hybrid') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  if (att.type === 'csa-hca-hybrid') {
    return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
  }
  return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}
```

Update `attentionDim`:

```ts
export function attentionDim(model: ModelArch): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') return att.kvLoraRank + att.qkRopeHeadDim
  if (att.type === 'linear-mla-hybrid') return att.kvLoraRank + att.qkRopeHeadDim
  if (att.type === 'csa-hca-hybrid') return model.numHeads * model.headDim
  return model.numHeads * model.headDim
}
```

Update `attendedSeqlenSummedOverLayers` — add the new branch BEFORE the trailing sliding/full fallthrough:

```ts
export function attendedSeqlenSummedOverLayers(model: ModelArch, seqlen: number, forKv = false): number {
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
  if (att.type === 'linear-mla-hybrid') {
    if (att.numLinearLayers + att.numFullLayers !== model.layers) {
      throw new Error(
        `linear-mla-hybrid layer counts must sum to model.layers: ` +
        `${att.numLinearLayers} + ${att.numFullLayers} ≠ ${model.layers}`
      )
    }
    return att.numFullLayers * seqlen
  }
  if (att.type === 'csa-hca-hybrid') {
    if (att.numSlidingLayers + att.numCsaLayers + att.numHcaLayers !== model.layers) {
      throw new Error(
        `csa-hca-hybrid layer counts must sum to model.layers: ` +
        `${att.numSlidingLayers} + ${att.numCsaLayers} + ${att.numHcaLayers} ≠ ${model.layers}`
      )
    }
    const csaCount = forKv ? (seqlen / att.csaCompressionM) : att.csaTopK
    return att.numSlidingLayers * Math.min(seqlen, att.slidingWindow)
         + att.numCsaLayers * (csaCount + att.slidingWindow)
         + att.numHcaLayers * (seqlen / att.hcaCompressionM + att.slidingWindow)
  }
  if (att.type === 'mla-dsa') return model.layers * (forKv ? seqlen : Math.min(seqlen, att.topK))
  const perLayer = att.type === 'sliding' ? Math.min(seqlen, att.window) : seqlen
  return model.layers * perLayer
}
```

- [ ] **Step 4: Run tests + type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: 5 new helper tests PASS. All existing tests still PASS. Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/sliding.test.ts
git commit -m "feat(calc): csa-hca-hybrid branches in kvBytesPerTokenPerLayer, attentionDim, attendedSeqlenSummedOverLayers"
```

No Co-Authored-By footer.

---

## Task 4: MTP wire-up in `computeDecode` + MTP unit test

Multiply `aggregateTokensPerS` by `(1 + numNextnLayers)` and divide `timePerTokenS` by the same factor. Per-pass `flopsPerStep` and `bytesPerStep` are unchanged — they describe work per forward pass.

**Files:**
- Modify: `calc/src/engine/decode.ts`
- Modify: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Append failing MTP unit test to `decode.test.ts`**

Append inside `describe('computeDecode', ...)` (before the closing `})`):

```ts
  it('MTP doubles aggregateTokensPerS and halves timePerTokenS for numNextnLayers=1', () => {
    const mtpModel = { ...testInput.model, numNextnLayers: 1 }
    const input = { ...testInput, model: mtpModel }
    const mtpMemory = computeMemory(input)
    const dMtp = computeDecode(input, opPoint, mtpMemory)
    const dBase = computeDecode(testInput, opPoint, memory)
    // Per-pass FLOPs and bytes are unchanged
    expect(dMtp.flopsPerStep).toBe(dBase.flopsPerStep)
    expect(dMtp.bytesPerStep).toBe(dBase.bytesPerStep)
    // Effective per-token time halves; aggregate throughput doubles
    expect(dMtp.timePerTokenS).toBeCloseTo(dBase.timePerTokenS / 2, 12)
    expect(dMtp.aggregateTokensPerS).toBeCloseTo(dBase.aggregateTokensPerS * 2, 6)
  })
```

- [ ] **Step 2: Run tests; expect MTP test failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- test/engine/decode.test.ts
```

Expected: the MTP test FAILS — current code doesn't divide by `(1 + numNextnLayers)`.

- [ ] **Step 3: Wire MTP into `computeDecode`**

In `calc/src/engine/decode.ts`, replace the return block:

```ts
  return {
    flopsPerStep,
    bytesPerStep,
    timePerTokenS: timeS,
    regime,
    aggregateTokensPerS: workload.concurrency / timeS
  }
```

With:

```ts
  const mtpFactor = 1 + model.numNextnLayers
  return {
    flopsPerStep,
    bytesPerStep,
    timePerTokenS: timeS / mtpFactor,
    regime,
    aggregateTokensPerS: workload.concurrency * mtpFactor / timeS
  }
```

- [ ] **Step 4: Run tests + type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: MTP test PASSES. All other tests still PASS byte-for-byte (factor=1 when numNextnLayers=0). Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/decode.ts calc/test/engine/decode.test.ts
git commit -m "feat(calc): MTP wire-up in computeDecode (numNextnLayers scales effective throughput)"
```

No Co-Authored-By footer.

---

## Task 5: Wiring regression tests in memory / prefill / decode

Three regression-lock tests using a synthetic `'csa-hca-hybrid'` fixture (layers=3, one of each type). They should pass without further code changes — confirming Task 3's wire-up is correct end-to-end.

**Files:**
- Modify: `calc/test/engine/memory.test.ts`
- Modify: `calc/test/engine/prefill.test.ts`
- Modify: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Append memory test**

Append inside `describe('computeMemory', ...)`:

```ts
  it('kvCachePerRequest for csa-hca-hybrid sums sliding + CSA + HCA contributions', () => {
    // testModel base: prompt+output=15, fp16, concurrency=2.
    // csa-hca-hybrid with layers=3 (1 sliding + 1 CSA + 1 HCA):
    //   slidingWindow=2, csaCompressionM=2, csaTopK=3, hcaCompressionM=4
    // attendedSeqlen(forKv=true) =
    //   1 × min(15, 2) + 1 × (15/2 + 2) + 1 × (15/4 + 2) = 2 + 9.5 + 5.75 = 17.25
    // kvBytesPerTokenPerLayer = 2 × 1 × 2 × 2 (fp16) = 8
    // kvCachePerRequest = 8 × 17.25 = 138
    const hybridModel: ModelArch = {
      ...testInput.model,
      layers: 3,
      attention: {
        type: 'csa-hca-hybrid',
        numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 1,
        slidingWindow: 2,
        csaCompressionM: 2, csaTopK: 3,
        csaIndexerHeads: 2, csaIndexerHeadDim: 2,
        hcaCompressionM: 4
      }
    }
    const input = { ...testInput, model: hybridModel }
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(138)
    // × concurrency 2 = 276
    expect(m.kvCacheTotal).toBe(276)
  })
```

Note: the test file's `testInput` and `computeMemory` imports may need `ModelArch` brought in. Check the existing imports at top of `memory.test.ts` and add `import type { ModelArch } from '../../src/engine/types'` if not already present.

- [ ] **Step 2: Append prefill test**

Append inside `describe('computePrefill', ...)`:

```ts
  it('flops for csa-hca-hybrid uses topK for CSA layer compute', () => {
    // testModel base: paramCount=1000, prompt=10.
    // csa-hca-hybrid (layers=3, same params as memory test):
    //   attendedSeqlen(forKv=false, seqlen=10) =
    //     1 × min(10, 2) + 1 × (csaTopK=3 + 2) + 1 × (10/4 + 2)
    //     = 2 + 5 + 4.5 = 11.5
    //   attentionDim = numHeads × headDim = 2 × 2 = 4
    //   MLP: 2 × 1000 × 10 = 20000
    //   Attention: 2 × prompt × attendedSeq × attentionDim = 2 × 10 × 11.5 × 4 = 920
    //   Total: 20920
    const hybridModel: ModelArch = {
      ...testInput.model,
      layers: 3,
      attention: {
        type: 'csa-hca-hybrid',
        numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 1,
        slidingWindow: 2,
        csaCompressionM: 2, csaTopK: 3,
        csaIndexerHeads: 2, csaIndexerHeadDim: 2,
        hcaCompressionM: 4
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, hybridMemory)
    expect(p.flops).toBe(20920)
  })
```

Add `import type { ModelArch } from '../../src/engine/types'` if not already present.

- [ ] **Step 3: Append decode test**

Append inside `describe('computeDecode', ...)`:

```ts
  it('flopsPerStep and bytesPerStep for csa-hca-hybrid include all three layer types', () => {
    // testModel base: paramCount=1000, concurrency=2, prompt+output=15.
    // avgSeqlen = 12.5.
    // csa-hca-hybrid (layers=3):
    //   attendedSeqlen(forKv=false, 12.5) =
    //     1 × min(12.5, 2) + 1 × (csaTopK=3 + 2) + 1 × (12.5/4 + 2)
    //     = 2 + 5 + 5.125 = 12.125
    //   attentionDim = 4
    //   flopsPerStep = (2 × 1000 + 2 × 12.125 × 4) × 2 = (2000 + 97) × 2 = 4194
    //   memory.kvCachePerRequest (from prompt+output=15) = 138
    //   bytesPerStep = 1000 × 2 + 138 × 2 = 2276
    const hybridModel: ModelArch = {
      ...testInput.model,
      layers: 3,
      attention: {
        type: 'csa-hca-hybrid',
        numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 1,
        slidingWindow: 2,
        csaCompressionM: 2, csaTopK: 3,
        csaIndexerHeads: 2, csaIndexerHeadDim: 2,
        hcaCompressionM: 4
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, hybridMemory)
    expect(d.flopsPerStep).toBe(4194)
    expect(d.bytesPerStep).toBe(2276)
  })
```

Add `import type { ModelArch } from '../../src/engine/types'` if not already present.

- [ ] **Step 4: Run tests + type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: all 3 new tests PASS. Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/test/engine/memory.test.ts calc/test/engine/prefill.test.ts calc/test/engine/decode.test.ts
git commit -m "test(calc): csa-hca-hybrid wiring in memory, prefill, decode"
```

No Co-Authored-By footer.

---

## Task 6: V4-Flash + V4-Pro data entries + integration tests

Final task. Adds V4-Pro and V4-Flash entries to the seed and integration tests that exercise both end-to-end.

V4-Flash config (verified against `deepseek-ai/DeepSeek-V4-Flash/config.json`):
- 43 layers, hidden 4096, `moe_intermediate_size: 2048`
- 64 query heads, 1 KV head (MQA), head_dim 512, vocab 129280
- 256 routed + 1 shared experts, 6 active per token
- compress_ratios array: 2 leading 0s (sliding) + 40 alternating 4/128 starting with 4 → 21 CSA + 20 HCA + 1 terminal
- MTP depth 1; csaTopK=512; csaIndexerHeads=64; csaIndexerHeadDim=128

V4-Pro config (verified against `deepseek-ai/DeepSeek-V4-Pro/config.json`):
- 61 layers, hidden 7168, `moe_intermediate_size: 3072`
- 128 query heads, 1 KV head (MQA), head_dim 512, vocab 129280
- 384 routed + 1 shared experts, 6 active per token
- compress_ratios array: 2 leading 128s (HCA) + 59 alternating 4/128 starting with 4 → 30 CSA + 31 HCA + 1 terminal
- MTP depth 1; csaTopK=1024; csaIndexerHeads=64; csaIndexerHeadDim=128

**Files:**
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/engine/calc.test.ts`

- [ ] **Step 1: Add V4-Flash entry**

In `calc/src/data/models.ts`, locate the `// === DeepSeek ===` family block. After the existing `deepseek-v3.2` entry, append:

```ts
  {
    id: 'deepseek-v4-flash', name: 'DeepSeek-V4-Flash', family: 'deepseek',
    layers: 43, hiddenDim: 4096, intermediateDim: 2048, vocabSize: 129280,
    numHeads: 64, numKvHeads: 1, headDim: 512,
    paramCount: 284_000_000_000,
    numNextnLayers: 1,
    attention: {
      type: 'csa-hca-hybrid',
      numSlidingLayers: 2, numCsaLayers: 21, numHcaLayers: 20,
      slidingWindow: 128,
      csaCompressionM: 4, csaTopK: 512,
      csaIndexerHeads: 64, csaIndexerHeadDim: 128,
      hcaCompressionM: 128
    },
    architecture: {
      type: 'moe',
      numExperts: 256, numExpertsActive: 6,
      numSharedExperts: 1,
      activeParamCount: 13_000_000_000
    }
  },
```

- [ ] **Step 2: Add V4-Pro entry**

After the just-added `deepseek-v4-flash` entry, append:

```ts
  {
    id: 'deepseek-v4-pro', name: 'DeepSeek-V4-Pro', family: 'deepseek',
    layers: 61, hiddenDim: 7168, intermediateDim: 3072, vocabSize: 129280,
    numHeads: 128, numKvHeads: 1, headDim: 512,
    paramCount: 1_600_000_000_000,
    numNextnLayers: 1,
    attention: {
      type: 'csa-hca-hybrid',
      numSlidingLayers: 0, numCsaLayers: 30, numHcaLayers: 31,
      slidingWindow: 128,
      csaCompressionM: 4, csaTopK: 1024,
      csaIndexerHeads: 64, csaIndexerHeadDim: 128,
      hcaCompressionM: 128
    },
    architecture: {
      type: 'moe',
      numExperts: 384, numExpertsActive: 6,
      numSharedExperts: 1,
      activeParamCount: 49_000_000_000
    }
  },
```

- [ ] **Step 3: Append integration tests to `calc.test.ts`**

Append a new top-level describe block at the bottom:

```ts
describe('calculate — DeepSeek V4 integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const v32 = MODELS.find(m => m.id === 'deepseek-v3.2')!
  const v4Flash = MODELS.find(m => m.id === 'deepseek-v4-flash')!
  const v4Pro = MODELS.find(m => m.id === 'deepseek-v4-pro')!

  it('V4-Pro at 1M context: KV cache sums CSA + HCA per-layer-type contributions', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: v4Pro,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 1048576, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Per-compressed-entry bytes = 2 × 1 × 512 × 2 = 2048
    // attendedSeqlen (kv) =
    //   0 sliding + 30 × (1048576/4 + 128) + 31 × (1048576/128 + 128)
    //   = 30 × 262272 + 31 × 8320
    //   = 7868160 + 257920
    //   = 8126080
    // kvCachePerRequest = 2048 × 8126080 = 16_642_211_840 bytes ≈ 16.64 GB
    const expected = 2048 * (30 * (1048576 / 4 + 128) + 31 * (1048576 / 128 + 128))
    expect(r.memory.kvCachePerRequest).toBe(expected)
  })

  it('V4-Pro at 1M: KV cache is ~4.4× smaller than V3.2 at fp16 apples-to-apples', () => {
    const baseInput: Omit<CalcInput, 'model'> = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 1048576, outputTokens: 0, concurrency: 1 }
    }
    const r4 = calculate({ ...baseInput, model: v4Pro })
    const r32 = calculate({ ...baseInput, model: v32 })
    const ratio = r32.memory.kvCachePerRequest / r4.memory.kvCachePerRequest
    // V4-Pro fp16 / V3.2 fp16: actual ≈ 4.43×
    // Paper's "10×" claim assumes V4 uses fp8 KV — that's a deployment choice, not modeled here.
    expect(ratio).toBeGreaterThan(4)
    expect(ratio).toBeLessThan(5)
  })

  it('V4-Pro decode throughput is 2× the without-MTP equivalent (numNextnLayers=1)', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: v4Pro,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 }
    }
    const rMtp = calculate(input)
    const noMtpInput = { ...input, model: { ...v4Pro, numNextnLayers: 0 } }
    const rNoMtp = calculate(noMtpInput)
    expect(rMtp.perf['peak'].decode.aggregateTokensPerS).toBeCloseTo(
      rNoMtp.perf['peak'].decode.aggregateTokensPerS * 2, 6
    )
    expect(rMtp.perf['peak'].decode.timePerTokenS).toBeCloseTo(
      rNoMtp.perf['peak'].decode.timePerTokenS / 2, 12
    )
    // Per-pass FLOPs and bytes unchanged
    expect(rMtp.perf['peak'].decode.flopsPerStep).toBe(rNoMtp.perf['peak'].decode.flopsPerStep)
    expect(rMtp.perf['peak'].decode.bytesPerStep).toBe(rNoMtp.perf['peak'].decode.bytesPerStep)
  })

  it('V4-Flash at 128k: KV cache uses 2 sliding + 21 CSA + 20 HCA', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: v4Flash,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 131072, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // 2048 × (2 × 128 + 21 × (131072/4 + 128) + 20 × (131072/128 + 128))
    //      = 2048 × (256 + 21 × 32896 + 20 × 1152)
    //      = 2048 × (256 + 690816 + 23040)
    //      = 2048 × 714112
    //      = 1_462_501_376  ≈ 1.46 GB
    const expected = 2048 * (
      2 * 128 +
      21 * (131072 / 4 + 128) +
      20 * (131072 / 128 + 128)
    )
    expect(r.memory.kvCachePerRequest).toBe(expected)
  })

  it('V4-Pro 1.6T weights at fp16 do not fit single H100 SXM-80', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: v4Pro,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // 1.6T × 2 bytes = 3.2 TB
    expect(r.memory.weights / 1e12).toBeCloseTo(3.2, 1)
    expect(r.memory.fits).toBe(false)
    // decode bytes ~ activeParams × 2 = 98 GB (49B active × fp16); memory-bound
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})
```

- [ ] **Step 4: Run tests + type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: tests PASS. Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/data/models.ts calc/test/engine/calc.test.ts
git commit -m "feat(calc): add DeepSeek V4-Flash and V4-Pro (csa-hca-hybrid + MTP)"
```

No Co-Authored-By footer.

---

## Done

After Task 6:
- Branch `feat/calc-v4-attention` has 6 implementation commits on top of the spec
- `'csa-hca-hybrid'` variant, `numNextnLayers` field, `'fp4'` Dtype all in place
- V4-Pro and V4-Flash in the seed
- All tests pass; type-check clean
- Controller will offer to open the PR

Notable items the spec footnotes as **not modeled**:
- CSA lightning indexer FLOPs/state (production-quantized)
- Q-LoRA compression (folded into activeParams)
- Output projection groups (folded into activeParams)
- Hash MoE routing for first 3 layers (small overhead)
- mHC residual variant (no roofline impact)
- MTP verification overhead (we give upper bound; production V4 sees ~75-85% acceptance)
- fp4 TFLOPS on Blackwell GPUs (operating-point follow-up)

# Linear Attention (KDA-style) + MLA Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `'linear-mla-hybrid'` as a new variant on `AttentionConfig`. Capture (a) constant-in-seqlen state size for linear-attention layers and (b) full softmax-attention work over MLA layers. Per-token linear compute modeled via a new helper. Add Kimi-Linear-48B-A3B as the canonical user.

**Architecture:** New discriminated-union variant with 8 fields (4 MLA dims + 2 layer counts + 2 KDA-geometry fields). Three existing math helpers gain a `'linear-mla-hybrid'` branch — KV bytes and attention dim delegate to the inner MLA, attended seqlen counts only the full layers. Two new helpers `linearAttentionStateBytes` and `linearAttentionFlopsPerToken` capture the linear-attention contribution. `computeMemory`, `computePrefill`, and `computeDecode` each gain one additive term.

**Tech Stack:** TypeScript, Svelte 5, Vitest. No new deps.

**Spec:** `calc/docs/superpowers/specs/2026-05-12-linear-attention-design.md`

**Depends on:** `feat/calc-mla-dims` branch (PR #98) — Kimi-Linear's MLA fields use the post-#98 schema (`qkNopeHeadDim`, `vHeadDim`).

---

## File Structure

```
src/engine/
  types.ts        # add 'linear-mla-hybrid' variant to AttentionConfig
  memory.ts       # add 'linear-mla-hybrid' branch to 3 existing helpers;
                  # add 2 new helpers (linearAttentionStateBytes, linearAttentionFlopsPerToken);
                  # wire computeMemory to add KDA state bytes
  prefill.ts      # add per-token linear-attention FLOPs term
  decode.ts       # add per-token linear-attention FLOPs + per-step state write-back
src/data/
  models.ts       # add Kimi-Linear entry
test/engine/
  sliding.test.ts # add helper tests for new branches + 2 new helpers + invariant
  memory.test.ts  # add memory test with synthetic linear-mla-hybrid fixture
  prefill.test.ts # add prefill test with synthetic fixture
  decode.test.ts  # add decode test with synthetic fixture
  calc.test.ts    # add Kimi-Linear integration test
```

---

## Task 1: Schema (add 'linear-mla-hybrid' variant)

Type-only change. Existing variants unaffected.

**Files:**
- Modify: `calc/src/engine/types.ts`

- [ ] **Step 1: Add the variant to AttentionConfig**

In `calc/src/engine/types.ts`, locate the `AttentionConfig` union. Append the new variant as the last arm:

```ts
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

- [ ] **Step 2: Run check + tests**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. All 79 existing tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/types.ts
git commit -m "feat(calc): add 'linear-mla-hybrid' variant to AttentionConfig"
```

No Co-Authored-By footer.

---

## Task 2: Helpers — 3 new branches + 2 new helpers (TDD)

Existing helpers each gain one branch for `'linear-mla-hybrid'`. Two new helpers (`linearAttentionStateBytes`, `linearAttentionFlopsPerToken`) capture the linear-attention bytes and FLOPs contributions.

**Files:**
- Modify: `calc/test/engine/sliding.test.ts`
- Modify: `calc/src/engine/memory.ts`

- [ ] **Step 1: Append failing tests to `test/engine/sliding.test.ts`**

Update the import at the top to include the two new helpers:

```ts
import {
  activeParams,
  kvBytesPerTokenPerLayer,
  attentionDim,
  attendedSeqlenSummedOverLayers,
  linearAttentionStateBytes,
  linearAttentionFlopsPerToken
} from '../../src/engine/memory'
```

Append the following blocks at the bottom of the file (each as a top-level `describe`):

```ts
describe('linear-mla-hybrid branches in existing helpers', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: {
      type: 'linear-mla-hybrid',
      kvLoraRank: 8, qkRopeHeadDim: 2,
      qkNopeHeadDim: 2, vHeadDim: 2,
      numLinearLayers: 3, numFullLayers: 1,
      numLinearHeads: 2, linearHeadDim: 4
    },
    architecture: { type: 'dense' }
  }

  it('kvBytesPerTokenPerLayer: returns the per-full-MLA-layer-per-token bytes', () => {
    // (kvLoraRank + qkRopeHeadDim) × 2 (fp16) = (8 + 2) × 2 = 20
    expect(kvBytesPerTokenPerLayer(base, 'fp16')).toBe(20)
  })

  it('attentionDim: returns the MLA absorbed-form attention dim', () => {
    // kvLoraRank + qkRopeHeadDim = 10
    expect(attentionDim(base)).toBe(10)
  })

  it('attendedSeqlenSummedOverLayers: returns numFullLayers × seqlen', () => {
    // 1 × 100 = 100  (only full layers do softmax-attention work)
    expect(attendedSeqlenSummedOverLayers(base, 100)).toBe(100)
    // 1 × 30 = 30
    expect(attendedSeqlenSummedOverLayers(base, 30)).toBe(30)
  })

  it('attendedSeqlenSummedOverLayers throws when layer counts do not sum to model.layers', () => {
    const m: ModelArch = {
      ...base,
      attention: {
        type: 'linear-mla-hybrid',
        kvLoraRank: 8, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2,
        numLinearLayers: 2, numFullLayers: 1,  // 2+1=3 ≠ 4
        numLinearHeads: 2, linearHeadDim: 4
      }
    }
    expect(() => attendedSeqlenSummedOverLayers(m, 100)).toThrow(/sum to model\.layers/)
  })
})

describe('linearAttentionStateBytes', () => {
  it('returns 0 for non-linear attention types', () => {
    const m: ModelArch = {
      id: 't', name: 'Test', family: 'test',
      layers: 4, hiddenDim: 16, intermediateDim: 64,
      numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
      paramCount: 1000,
      attention: { type: 'full' },
      architecture: { type: 'dense' }
    }
    expect(linearAttentionStateBytes(m, 'fp16')).toBe(0)
  })

  it('returns numLinearLayers × numLinearHeads × linearHeadDim² × bytes for linear-mla-hybrid', () => {
    const m: ModelArch = {
      id: 't', name: 'Test', family: 'test',
      layers: 4, hiddenDim: 16, intermediateDim: 64,
      numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
      paramCount: 1000,
      attention: {
        type: 'linear-mla-hybrid',
        kvLoraRank: 8, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2,
        numLinearLayers: 3, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 4
      },
      architecture: { type: 'dense' }
    }
    // 3 × 2 × 4² × 2 (fp16) = 96
    expect(linearAttentionStateBytes(m, 'fp16')).toBe(96)
  })
})

describe('linearAttentionFlopsPerToken', () => {
  it('returns 0 for non-linear attention types', () => {
    const m: ModelArch = {
      id: 't', name: 'Test', family: 'test',
      layers: 4, hiddenDim: 16, intermediateDim: 64,
      numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
      paramCount: 1000,
      attention: { type: 'full' },
      architecture: { type: 'dense' }
    }
    expect(linearAttentionFlopsPerToken(m)).toBe(0)
  })

  it('returns 2 × numLinearLayers × numLinearHeads × linearHeadDim² for linear-mla-hybrid', () => {
    const m: ModelArch = {
      id: 't', name: 'Test', family: 'test',
      layers: 4, hiddenDim: 16, intermediateDim: 64,
      numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
      paramCount: 1000,
      attention: {
        type: 'linear-mla-hybrid',
        kvLoraRank: 8, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2,
        numLinearLayers: 3, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 4
      },
      architecture: { type: 'dense' }
    }
    // 2 × 3 × 2 × 4² = 192
    expect(linearAttentionFlopsPerToken(m)).toBe(192)
  })
})
```

- [ ] **Step 2: Run tests; expect failures**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- test/engine/sliding.test.ts
```

Expected: the new tests FAIL because the helpers don't yet recognize `'linear-mla-hybrid'` and the two new helpers don't exist.

- [ ] **Step 3: Update helpers and add new ones in `memory.ts`**

In `calc/src/engine/memory.ts`:

Update `kvBytesPerTokenPerLayer` — add the linear-mla-hybrid branch alongside the existing MLA / MLA-DSA check:

```ts
export function kvBytesPerTokenPerLayer(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  if (att.type === 'linear-mla-hybrid') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
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
  return model.numHeads * model.headDim
}
```

Update `attendedSeqlenSummedOverLayers` — add the linear-mla-hybrid branch BEFORE the existing fallthrough:

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
  if (att.type === 'mla-dsa') return model.layers * (forKv ? seqlen : Math.min(seqlen, att.topK))
  const perLayer = att.type === 'sliding' ? Math.min(seqlen, att.window) : seqlen
  return model.layers * perLayer
}
```

Add the two new helpers after `attendedSeqlenSummedOverLayers`:

```ts
// Constant per-request state bytes from linear-attention layers. Zero for non-linear models.
export function linearAttentionStateBytes(model: ModelArch, kvDtype: Dtype): number {
  if (model.attention.type !== 'linear-mla-hybrid') return 0
  const a = model.attention
  return a.numLinearLayers * a.numLinearHeads * a.linearHeadDim * a.linearHeadDim * bytesOf(kvDtype)
}

// FLOPs per token from linear-attention layers (constant in seqlen). Zero for non-linear models.
export function linearAttentionFlopsPerToken(model: ModelArch): number {
  if (model.attention.type !== 'linear-mla-hybrid') return 0
  const a = model.attention
  return 2 * a.numLinearLayers * a.numLinearHeads * a.linearHeadDim * a.linearHeadDim
}
```

- [ ] **Step 4: Run tests + type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: all 88 tests PASS (79 existing + 9 new helper tests). Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/sliding.test.ts
git commit -m "feat(calc): linear-mla-hybrid helper branches + new linearAttention helpers"
```

No Co-Authored-By footer.

---

## Task 3: Wire helpers into memory / prefill / decode

`computeMemory` gains the KDA-state-bytes additive term. `computePrefill` and `computeDecode` gain the linear-attention FLOPs term. `computeDecode` additionally adds the state write-back bytes term.

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/src/engine/prefill.ts`
- Modify: `calc/src/engine/decode.ts`

- [ ] **Step 1: Wire `computeMemory.kvCachePerRequest`**

In `calc/src/engine/memory.ts`, locate the KV section inside `computeMemory` (currently lines 56-58 after PR #98):

```ts
  const kvPerLayerPerToken = kvBytesPerTokenPerLayer(model, quant.kv)
  const attendedSeqlen = attendedSeqlenSummedOverLayers(model, seqlen, true)
  const kvCachePerRequest = kvPerLayerPerToken * attendedSeqlen
```

Replace with:

```ts
  const kvPerLayerPerToken = kvBytesPerTokenPerLayer(model, quant.kv)
  const attendedSeqlen = attendedSeqlenSummedOverLayers(model, seqlen, true)
  const kvCachePerRequest =
    kvPerLayerPerToken * attendedSeqlen
    + linearAttentionStateBytes(model, quant.kv)
```

- [ ] **Step 2: Wire `computePrefill.flops`**

In `calc/src/engine/prefill.ts`, update the import:

```ts
import { attendedSeqlenSummedOverLayers, activeParams, attentionDim, linearAttentionFlopsPerToken } from './memory'
```

Locate the flops computation:

```ts
  const flops =
    2 * activeParams(model) * p +
    2 * p * attendedSeqlenSummedOverLayers(model, p) * attentionDim(model)
```

Replace with:

```ts
  const flops =
    2 * activeParams(model) * p +
    2 * p * attendedSeqlenSummedOverLayers(model, p) * attentionDim(model) +
    p * linearAttentionFlopsPerToken(model)
```

- [ ] **Step 3: Wire `computeDecode.flopsPerStep` and `computeDecode.bytesPerStep`**

In `calc/src/engine/decode.ts`, update the import:

```ts
import {
  attendedSeqlenSummedOverLayers,
  activeParams,
  attentionDim,
  linearAttentionFlopsPerToken,
  linearAttentionStateBytes
} from './memory'
```

Locate the flopsPerStep computation:

```ts
  const flopsPerStep =
    (2 * activeParams(model) + 2 * attendedSeqlenSummedOverLayers(model, avgSeqlen) * attentionDim(model)) *
    workload.concurrency
```

Replace with:

```ts
  const flopsPerStep =
    (2 * activeParams(model)
     + 2 * attendedSeqlenSummedOverLayers(model, avgSeqlen) * attentionDim(model)
     + linearAttentionFlopsPerToken(model)) *
    workload.concurrency
```

Locate the bytesPerStep computation:

```ts
  const bytesPerStep =
    activeParams(model) * bytesOf(quant.weights) +
    memory.kvCachePerRequest * workload.concurrency
```

Replace with:

```ts
  const bytesPerStep =
    activeParams(model) * bytesOf(quant.weights) +
    memory.kvCachePerRequest * workload.concurrency +
    linearAttentionStateBytes(model, quant.kv) * workload.concurrency  // KDA state write-back
```

- [ ] **Step 4: Run tests + type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: 88 tests PASS byte-for-byte (the new helpers return 0 for all existing fixtures, so no test value changes). Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/src/engine/prefill.ts calc/src/engine/decode.ts
git commit -m "feat(calc): wire linearAttention helpers into computeMemory, computePrefill, computeDecode"
```

No Co-Authored-By footer.

---

## Task 4: Wiring regression tests in memory / prefill / decode

Three regression-lock tests using a synthetic `'linear-mla-hybrid'` fixture, one per call site. They should pass without further code changes — confirming the wire-up in Task 3 is correct.

**Files:**
- Modify: `calc/test/engine/memory.test.ts`
- Modify: `calc/test/engine/prefill.test.ts`
- Modify: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Append memory test**

Append inside `describe('computeMemory', ...)` (just before its closing `})`):

```ts
  it('kvCachePerRequest for linear-mla-hybrid = MLA kv + KDA state', () => {
    // testModel: layers=2, fp16; prompt+output=15.
    // linear-mla-hybrid with numLinear=1, numFull=1; MLA kvLoraRank=5, rope=1;
    // KDA: numLinearHeads=2, linearHeadDim=2.
    //   per-full-layer-per-token KV bytes = (5 + 1) × 2 = 12
    //   attendedSeqlen for kv (numFull × seq) = 1 × 15 = 15
    //   KDA state bytes = 1 × 2 × 2² × 2 = 16
    //   kvCachePerRequest = 12 × 15 + 16 = 196
    const hybridModel = {
      ...testInput.model,
      attention: {
        type: 'linear-mla-hybrid' as const,
        kvLoraRank: 5, qkRopeHeadDim: 1,
        qkNopeHeadDim: 1, vHeadDim: 1,
        numLinearLayers: 1, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 2
      }
    }
    const input = { ...testInput, model: hybridModel }
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(196)
    // × concurrency 2 = 392 bytes total
    expect(m.kvCacheTotal).toBe(392)
  })
```

- [ ] **Step 2: Append prefill test**

Append inside `describe('computePrefill', ...)`:

```ts
  it('flops for linear-mla-hybrid includes KDA per-token term', () => {
    // testModel: layers=2, paramCount=1000, prompt=10.
    // linear-mla-hybrid as above:
    //   attentionDim = 5 + 1 = 6
    //   attendedSeqlen(10) = 1 × 10 = 10  (only the 1 full layer)
    //   MLP: 2 × 1000 × 10 = 20000
    //   Softmax attention: 2 × 10 × 10 × 6 = 1200
    //   KDA per-token FLOPs = 2 × 1 × 2 × 2² = 16; × 10 prompt = 160
    //   Total = 21360
    const hybridModel = {
      ...testInput.model,
      attention: {
        type: 'linear-mla-hybrid' as const,
        kvLoraRank: 5, qkRopeHeadDim: 1,
        qkNopeHeadDim: 1, vHeadDim: 1,
        numLinearLayers: 1, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 2
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, hybridMemory)
    expect(p.flops).toBe(21360)
  })
```

- [ ] **Step 3: Append decode test**

Append inside `describe('computeDecode', ...)`:

```ts
  it('flopsPerStep and bytesPerStep for linear-mla-hybrid include KDA terms', () => {
    // testModel: layers=2, paramCount=1000, concurrency=2.
    // avgSeqlen = 10 + 5/2 = 12.5.
    // linear-mla-hybrid:
    //   attentionDim = 6
    //   attendedSeqlen(12.5) = 1 × 12.5 = 12.5  (only the 1 full layer)
    //   KDA per-token FLOPs = 16
    //   flopsPerStep = (2 × 1000 + 2 × 12.5 × 6 + 16) × 2 = (2000 + 150 + 16) × 2 = 4332
    //   KDA state bytes = 16
    //   kvCachePerRequest = 12 × 12.5_? -- but bytesPerStep uses MEMORY.kvCachePerRequest from prompt+output=15:
    //     12 × 15 + 16 = 196
    //   bytesPerStep = 1000 × 2 + 196 × 2 + 16 × 2 = 2000 + 392 + 32 = 2424
    const hybridModel = {
      ...testInput.model,
      attention: {
        type: 'linear-mla-hybrid' as const,
        kvLoraRank: 5, qkRopeHeadDim: 1,
        qkNopeHeadDim: 1, vHeadDim: 1,
        numLinearLayers: 1, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 2
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, hybridMemory)
    expect(d.flopsPerStep).toBe(4332)
    expect(d.bytesPerStep).toBe(2424)
  })
```

- [ ] **Step 4: Run tests + type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: 91 tests PASS (88 + 3 new). Type-check clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/test/engine/memory.test.ts calc/test/engine/prefill.test.ts calc/test/engine/decode.test.ts
git commit -m "test(calc): linear-mla-hybrid wiring in memory, prefill, decode"
```

No Co-Authored-By footer.

---

## Task 5: Kimi-Linear data entry + integration test

Adds Kimi-Linear-48B-A3B-Instruct as canonical user. Integration test compares Kimi-Linear's KV cache to the hypothetical all-MLA equivalent.

Kimi-Linear config (verified against `moonshotai/Kimi-Linear-48B-A3B-Instruct/config.json`):
- 27 layers, hidden 2304, intermediate 9216
- 32 attention heads, 32 KV heads, vocab 163840
- Full layers (7 of 27): MLA `kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128`
- KDA layers (20 of 27): `num_heads=32, head_dim=128`
- MoE: 256 routed + 1 shared, 8 active

paramCount / activeParamCount: 48B / 3B per the model card name.

**Files:**
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/engine/calc.test.ts`

- [ ] **Step 1: Add the Kimi-Linear entry**

In `calc/src/data/models.ts`, locate the `// === Moonshot / Kimi ===` family block. After the existing `kimi-k2` entry, append:

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
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 3_000_000_000
    }
  },
```

- [ ] **Step 2: Append the Kimi-Linear integration test**

In `calc/test/engine/calc.test.ts`, append a new top-level describe block at the bottom:

```ts
describe('calculate — Kimi-Linear (linear + MLA hybrid) integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const kl = MODELS.find(m => m.id === 'kimi-linear')!

  it('Kimi-Linear at 128k prompt: KV cache uses 7 MLA layers + 20 KDA state', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: kl,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 131072, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // MLA per-layer-per-token = (512+64) × 2 = 1152 bytes
    // attendedSeq (kv) = 7 × 131072 = 917_504
    // MLA KV cache = 1152 × 917_504 = 1_056_964_608 ≈ 1.057 GB
    // KDA state = 20 × 32 × 128² × 2 = 20_971_520 ≈ 20 MB
    // Total kvCachePerRequest = 1_077_936_128 ≈ 1.08 GB
    const mlaKv = 7 * (512 + 64) * 2 * 131072
    const kdaState = 20 * 32 * 128 * 128 * 2
    expect(r.memory.kvCachePerRequest).toBe(mlaKv + kdaState)
  })

  it('Kimi-Linear KV cache at 128k is ~3.78× smaller than hypothetical all-MLA equivalent', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: kl,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 131072, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Hypothetical all-MLA: 27 × 576 × 2 × 131072 = 4_076_863_488
    const allMlaEquivalent = 27 * 576 * 2 * 131072
    const ratio = allMlaEquivalent / r.memory.kvCachePerRequest
    expect(ratio).toBeGreaterThan(3.5)        // actual ≈ 3.78
    expect(ratio).toBeLessThan(3.9)           // asymptote 27/7 ≈ 3.86
  })

  it('Kimi-Linear decode at batch=1 is memory-bound on weight reads (3B active)', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: kl,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // 48B × 2 = 96 GB → exceeds H100 SXM-80 capacity (80 GB)
    expect(r.memory.fits).toBe(false)
    // decode bytes/step ≈ 3B × 2 = 6 GB (active params) + small KV/state
    const activeBytes = 3_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 1e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})
```

- [ ] **Step 3: Run tests + type-check**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test
npm run check
```

Expected: 94 tests PASS (91 + 3 new). Type-check clean.

- [ ] **Step 4: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/data/models.ts calc/test/engine/calc.test.ts
git commit -m "feat(calc): add Kimi-Linear-48B-A3B (linear+MLA hybrid)"
```

No Co-Authored-By footer.

---

## Done

After Task 5:
- Branch `feat/calc-linear-attention` has 5 implementation commits on top of the spec commit (stacked on `feat/calc-mla-dims`)
- 94 / 94 tests pass
- `npm run check` clean
- `'linear-mla-hybrid'` variant exists; Kimi-Linear populates it
- Controller will offer to open the PR (target base may be `feat/calc-mla-dims` while #98 is in review, or `main` once #98 merges)

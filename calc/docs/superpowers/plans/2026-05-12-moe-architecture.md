# MoE Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for uniform Mixture-of-Experts models (Mixtral 8x7B as the canonical user) so prefill/decode FLOPs and decode bytes scale with active params while weight-storage memory still scales with total params.

**Architecture:** Discriminated-union `ArchitectureConfig` field on `ModelArch`, mirroring the `attention` axis from the sliding-window feature. A single helper `activeParams(model)` consumed by `prefill.ts` (MLP FLOPs) and `decode.ts` (MLP FLOPs + weight bytes). `memory.ts` deliberately keeps using `paramCount` for storage. Two axes (`attention`, `architecture`) compose orthogonally — Mixtral exercises both.

**Tech Stack:** TypeScript, Svelte 5, Vitest. No new deps.

**Spec:** `calc/docs/superpowers/specs/2026-05-12-moe-architecture-design.md`

---

## File Structure

```
src/engine/
  types.ts         # add ArchitectureConfig type, add `architecture` to ModelArch
  memory.ts        # add activeParams helper next to effectiveAttentionLength
  prefill.ts       # use activeParams for MLP FLOPs term
  decode.ts        # use activeParams for MLP FLOPs and per-step weight bytes
src/data/
  models.ts        # retrofit 13 entries with architecture:dense; add Mixtral 8x7B
test/
  fixtures.ts      # tag testModel with architecture:dense
  engine/
    sliding.test.ts    # extend with activeParams cases (helper file already exists)
    prefill.test.ts    # MoE FLOPs test
    decode.test.ts     # MoE FLOPs + bytes tests
    calc.test.ts       # Mixtral 8x7B integration test
```

---

## Task 1: Schema + retrofit (single commit)

Adding the required `architecture` field to `ModelArch` breaks compilation for every existing model. Bundle the type addition with the retrofit so each commit leaves the repo in a buildable state. Mirrors Task 1 of the sliding-window plan.

**Files:**
- Modify: `calc/src/engine/types.ts`
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/fixtures.ts`

- [ ] **Step 1: Add ArchitectureConfig type and field to types.ts**

Open `calc/src/engine/types.ts`. Just above the `ModelArch` interface, after the existing `AttentionConfig` type, insert:

```ts
export type ArchitectureConfig =
  | { type: 'dense' }
  | { type: 'moe'; numExperts: number; numExpertsActive: number; activeParamCount: number }
```

Then add `architecture: ArchitectureConfig` as a new required field on `ModelArch` (after `attention`):

```ts
export interface ModelArch {
  id: string
  name: string
  family: string
  layers: number
  hiddenDim: number
  intermediateDim: number
  numHeads: number
  numKvHeads: number
  headDim: number
  vocabSize: number
  paramCount: number
  attention: AttentionConfig
  architecture: ArchitectureConfig
}
```

- [ ] **Step 2: Retrofit every entry in models.ts with architecture: { type: 'dense' }**

There are 13 entries (qwen3-1.7b, qwen3-4b, qwen3-8b, qwen3-14b, qwen3-32b, llama-3.3-70b, llama-3.1-405b, gemma-3-12b, gemma-3-27b, mistral-7b-v0.1, mistral-small-3.1-24b, mistral-large-2, phi-4). Add `architecture: { type: 'dense' }` as the last field of EACH. Example for the first:

```ts
{
  id: 'qwen3-1.7b', name: 'Qwen3 1.7B', family: 'qwen3',
  layers: 28, hiddenDim: 2048, intermediateDim: 6144,
  numHeads: 16, numKvHeads: 8, headDim: 128, vocabSize: 151936,
  paramCount: 1_720_000_000,
  attention: { type: 'full' },
  architecture: { type: 'dense' }       // ← add this line
},
```

Repeat for all 13 entries.

- [ ] **Step 3: Update test fixture's testModel**

Open `calc/test/fixtures.ts`. Add `architecture: { type: 'dense' }` to the `testModel` const as the last field:

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
  attention: { type: 'full' },
  architecture: { type: 'dense' }
}
```

- [ ] **Step 4: Verify compile and tests still pass**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: no TS errors, all 41 existing tests PASS (behavior unchanged because new field exists but no engine code reads it yet).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/types.ts calc/src/data/models.ts calc/test/fixtures.ts
git commit -m "feat(calc): add ArchitectureConfig schema and retrofit existing models as dense"
```

No Co-Authored-By footer.

---

## Task 2: activeParams helper (TDD)

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/sliding.test.ts`

(We extend the existing helper-tests file rather than create a new one — it already lives alongside `effectiveAttentionLength`.)

- [ ] **Step 1: Append failing tests to `test/engine/sliding.test.ts`**

Add a new describe block at the bottom of `calc/test/engine/sliding.test.ts`:

```ts
import { activeParams } from '../../src/engine/memory'
import type { ModelArch } from '../../src/engine/types'

describe('activeParams', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 2, hiddenDim: 4, intermediateDim: 8,
    numHeads: 2, numKvHeads: 1, headDim: 2, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('returns paramCount for dense models', () => {
    expect(activeParams(base)).toBe(1000)
  })

  it('returns activeParamCount for MoE models', () => {
    const moe: ModelArch = {
      ...base,
      paramCount: 8000,
      architecture: {
        type: 'moe',
        numExperts: 8,
        numExpertsActive: 2,
        activeParamCount: 2000
      }
    }
    expect(activeParams(moe)).toBe(2000)
  })
})
```

(The import statement is appended to the top of the file — alongside the existing `effectiveAttentionLength` import. Make sure there's only one combined import line:
`import { effectiveAttentionLength, activeParams } from '../../src/engine/memory'`.)

- [ ] **Step 2: Verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/sliding.test.ts
```

Expected: 2 new tests FAIL with `activeParams is not a function`. The 4 existing helper tests continue to PASS.

- [ ] **Step 3: Implement the helper in memory.ts**

Open `calc/src/engine/memory.ts`. The current types import is:
`import type { AttentionConfig, CalcInput, GpuVariant, MemoryResult } from './types'`

Replace with:
`import type { AttentionConfig, CalcInput, GpuVariant, MemoryResult, ModelArch } from './types'`

Then near `effectiveAttentionLength` (just after it), add:

```ts
export function activeParams(model: ModelArch): number {
  return model.architecture.type === 'moe'
    ? model.architecture.activeParamCount
    : model.paramCount
}
```

- [ ] **Step 4: Verify pass**

```bash
npx vitest run test/engine/sliding.test.ts
```

Expected: 6 tests PASS in `sliding.test.ts` (4 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/sliding.test.ts
git commit -m "feat(calc): add activeParams helper for MoE compute scaling"
```

---

## Task 3: Prefill MLP FLOPs use activeParams

**Files:**
- Modify: `calc/src/engine/prefill.ts`
- Modify: `calc/test/engine/prefill.test.ts`

- [ ] **Step 1: Append failing test to `test/engine/prefill.test.ts`**

Inside the existing `describe('computePrefill', ...)` block, add this `it` block BEFORE the closing `})`:

```ts
  it('FLOPs MLP term uses activeParams for MoE', () => {
    // testModel: paramCount=1000, hiddenDim=4, layers=2, prompt=10.
    // For MoE with activeParamCount=250:
    //   MLP: 2 × 250 × 10 = 5000
    //   Attention: 2 × 2 × 10 × 10 × 4 = 1600 (full attention, unchanged)
    //   Total = 6600
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, moeMemory)
    expect(p.flops).toBe(6600)
  })
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/prefill.test.ts
```

Expected: new test FAILS (flops reports 21600 — still using `paramCount` 1000); 5 existing prefill tests still PASS.

- [ ] **Step 3: Update `computePrefill` in `src/engine/prefill.ts`**

Extend the existing helper import to include `activeParams`:

```ts
import { effectiveAttentionLength, activeParams } from './memory'
```

Then find the existing flops calculation:

```ts
const effP = effectiveAttentionLength(p, model.attention)
const flops =
  2 * model.paramCount * p +
  2 * model.layers * p * effP * model.hiddenDim
```

Replace `model.paramCount` (the FIRST occurrence in the formula — the MLP term) with `activeParams(model)`:

```ts
const effP = effectiveAttentionLength(p, model.attention)
const flops =
  2 * activeParams(model) * p +
  2 * model.layers * p * effP * model.hiddenDim
```

The attention term (second line) is unchanged.

- [ ] **Step 4: Verify pass**

```bash
npx vitest run test/engine/prefill.test.ts
```

Expected: 6 prefill tests PASS (5 old + 1 new). Full suite: `npm test` → 43 total passing.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/prefill.ts calc/test/engine/prefill.test.ts
git commit -m "feat(calc): scale prefill MLP FLOPs by activeParams for MoE"
```

---

## Task 4: Decode MLP FLOPs use activeParams

**Files:**
- Modify: `calc/src/engine/decode.ts`
- Modify: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Append failing test to `test/engine/decode.test.ts`**

Inside the existing `describe('computeDecode', ...)` block, add:

```ts
  it('flopsPerStep MLP term uses activeParams for MoE', () => {
    // testModel: paramCount=1000, hiddenDim=4, layers=2, concurrency=2.
    // avgSeqlen = 10 + 5/2 = 12.5.
    // For MoE with activeParamCount=250:
    //   (2 × 250 + 2 × 2 × 12.5 × 4) × 2 = (500 + 200) × 2 = 1400
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, moeMemory)
    expect(d.flopsPerStep).toBe(1400)
  })
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/decode.test.ts
```

Expected: new test FAILS (flopsPerStep reports 4400 — still using `paramCount` 1000); 5 existing decode tests still PASS.

- [ ] **Step 3: Update `computeDecode` in `src/engine/decode.ts`**

Extend the existing helper import:

```ts
import { effectiveAttentionLength, activeParams } from './memory'
```

Find the existing flopsPerStep calculation:

```ts
const effAvg = effectiveAttentionLength(avgSeqlen, model.attention)
const flopsPerStep =
  (2 * model.paramCount + 2 * model.layers * effAvg * model.hiddenDim) *
  workload.concurrency
```

Replace `model.paramCount` with `activeParams(model)`:

```ts
const effAvg = effectiveAttentionLength(avgSeqlen, model.attention)
const flopsPerStep =
  (2 * activeParams(model) + 2 * model.layers * effAvg * model.hiddenDim) *
  workload.concurrency
```

(Note: `bytesPerStep` is the next task, leave it unchanged for now.)

- [ ] **Step 4: Verify pass**

```bash
npx vitest run test/engine/decode.test.ts
```

Expected: 6 decode tests PASS. Full suite: 44 total.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/decode.ts calc/test/engine/decode.test.ts
git commit -m "feat(calc): scale decode MLP FLOPs by activeParams for MoE"
```

---

## Task 5: Decode bytes per step use activeParams (weight read)

**Files:**
- Modify: `calc/src/engine/decode.ts`
- Modify: `calc/test/engine/decode.test.ts`

This is the second decode change — the per-step weight bandwidth. Separated into its own task / commit so the TDD cycle stays focused.

- [ ] **Step 1: Append failing test to `test/engine/decode.test.ts`**

Inside `describe('computeDecode', ...)`, append:

```ts
  it('bytesPerStep weight term uses activeParams for MoE', () => {
    // testModel: paramCount=1000, fp16 weights → 2 bytes/param.
    // For MoE with activeParamCount=250:
    //   weight bytes per step = 250 × 2 = 500
    //   kv per request = 240 (existing fixture), × concurrency 2 = 480
    //   total bytesPerStep = 500 + 480 = 980
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, moeMemory)
    expect(d.bytesPerStep).toBe(980)
  })
```

- [ ] **Step 2: Verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/decode.test.ts
```

Expected: new test FAILS (bytesPerStep reports 2480 — using `memory.weights` = 1000 × 2 = 2000 instead of activeParams × 2 = 500). 6 existing decode tests still PASS.

- [ ] **Step 3: Update `computeDecode` bytes calculation in `src/engine/decode.ts`**

The current decode bytesPerStep computation looks like:

```ts
const bytesPerStep = memory.weights + memory.kvCachePerRequest * workload.concurrency
```

Replace with a direct calculation using `activeParams` and the dtype lookup:

```ts
import { bytesOf } from './dtypes'
// ...
const bytesPerStep =
  activeParams(model) * bytesOf(quant.weights) +
  memory.kvCachePerRequest * workload.concurrency
```

The `bytesOf` import: add it to the existing imports at the top of `decode.ts` if not already there. After the change the top of `decode.ts` has:

```ts
import { effectiveAttentionLength, activeParams } from './memory'
import { bytesOf } from './dtypes'
import { roofline } from './roofline'
```

(Adjust based on the actual existing import order — the point is `bytesOf` becomes a new import.)

- [ ] **Step 4: Verify pass**

```bash
npx vitest run test/engine/decode.test.ts
```

Expected: 7 decode tests PASS. Full suite: 45 total.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/decode.ts calc/test/engine/decode.test.ts
git commit -m "feat(calc): scale decode bytesPerStep weight read by activeParams for MoE"
```

---

## Task 6: Add Mixtral 8x7B (verified) and integration test

**Files:**
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/engine/calc.test.ts`

- [ ] **Step 1: Verify Mixtral 8x7B v0.1 config via WebFetch**

Use WebFetch on `https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json` with prompt:

> Return the raw JSON contents. Specifically: num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim if present, vocab_size, sliding_window, num_local_experts, num_experts_per_tok.

Expected values (subject to verification):
- `num_hidden_layers`: 32
- `hidden_size`: 4096
- `intermediate_size`: 14336
- `num_attention_heads`: 32
- `num_key_value_heads`: 8
- `head_dim`: 128 (or `hidden_size / num_attention_heads`)
- `vocab_size`: 32000
- `sliding_window`: 4096
- `num_local_experts`: 8
- `num_experts_per_tok`: 2

If any value differs, **use the value from the config**. The `paramCount` (46_702_792_704) and `activeParamCount` (12_879_204_352) come from the Mixtral model card / safetensors metadata, not the config — accept as-is.

- [ ] **Step 2: Add Mixtral 8x7B entry to models.ts**

Insert the new entry into the Mistral section (after `mistral-7b-v0.1`, before `mistral-small-3.1-24b`):

```ts
{
  id: 'mixtral-8x7b', name: 'Mixtral 8x7B v0.1', family: 'mistral',
  layers: 32, hiddenDim: 4096, intermediateDim: 14336,
  numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
  paramCount: 46_702_792_704,
  attention: { type: 'sliding', window: 4096 },
  architecture: {
    type: 'moe',
    numExperts: 8,
    numExpertsActive: 2,
    activeParamCount: 12_879_204_352
  }
},
```

Adjust any field if Step 1 surfaced a discrepancy.

- [ ] **Step 3: Add integration test**

Open `calc/test/engine/calc.test.ts`. At the bottom of the file (after the last existing `describe` block), append:

```ts
describe('calculate — MoE integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const mixtral = MODELS.find(m => m.id === 'mixtral-8x7b')!

  it('Mixtral 8x7B on H100 SXM-80: weights use total params, decode bytes use active', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: mixtral,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)

    // memory.weights = paramCount (46.7B) × 2 bytes (fp16) ≈ 93.4 GB.
    // 93.4 GB > 80 GB capacity → memory.fits === false.
    expect(r.memory.weights / 1e9).toBeCloseTo(93.4, 0)
    expect(r.memory.fits).toBe(false)

    // decode.bytesPerStep = activeParamCount × 2 bytes + kvCachePerRequest × 1.
    // activeParamCount = 12.879B → 25.76 GB; KV is small at batch=1.
    // Expect decode.bytesPerStep ≈ 25.76 GB, well below paramCount-based 93.4 GB.
    const expectedActiveBytes = 12_879_204_352 * 2  // fp16
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(expectedActiveBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(expectedActiveBytes + 2e9)

    // Decode is memory-bound at batch=1 (active weight reads dominate).
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})
```

- [ ] **Step 4: Run all checks**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. Total test count = 46 (45 from earlier tasks + 1 integration).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/data/models.ts calc/test/engine/calc.test.ts
git commit -m "feat(calc): add Mixtral 8x7B (MoE 8/2) with integration test"
```

---

## Self-Review Notes

Spec coverage:

- **Schema (Section "Schema" of spec)** — Task 1 adds `ArchitectureConfig`, retrofits 13 models + fixture
- **Helper (Section "Math", `activeParams`)** — Task 2
- **Prefill MLP FLOPs (Section "Math" subsection 1)** — Task 3
- **Decode MLP FLOPs (Section "Math" subsection 2)** — Task 4
- **Decode bytes per step (Section "Math" subsection 3)** — Task 5
- **Memory storage UNCHANGED (Section "Math" — memory.weights stays at paramCount)** — verified by Task 6 integration assertion (`memory.weights / 1e9 ≈ 93.4`)
- **Prefill bytes UNCHANGED** — implicit; no task touches prefill.bytes
- **Data update (Section "Data")** — Task 6
- **Testing (Section "Testing")** — distributed:
  - Helper: Task 2
  - Prefill: Task 3
  - Decode FLOPs: Task 4
  - Decode bytes: Task 5
  - Memory regression: Task 6 integration test (asserts weights = paramCount-based)
  - Integration with Mixtral 8x7B: Task 6
- **Regression of 41 existing tests** — continuous (`npm test` after each task)
- **Evolution path** — addressed by the discriminated-union schema ready for `'moe-shared'` and other future variants

Type / API consistency check:

- `ArchitectureConfig` and `ModelArch.architecture` defined in Task 1, consumed unchanged in Tasks 2–6
- `activeParams(model: ModelArch): number` defined in Task 2, called identically in Tasks 3, 4, 5 (`activeParams(model)`)
- `effectiveAttentionLength` from sliding-window feature continues to be imported and used alongside; both helpers live in `memory.ts`
- Existing function signatures (`computeMemory`, `computePrefill`, `computeDecode`) unchanged — internal-only edits

No placeholders or vague steps. Every code block contains the exact text to write.

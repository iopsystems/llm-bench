# Sliding Window Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for uniform sliding-window attention models (Mistral 7B v0.1 as the canonical user) so the roofline math correctly bounds KV cache size, prefill attention FLOPs, and decode attention FLOPs to the window when applicable.

**Architecture:** Discriminated-union `AttentionConfig` field on `ModelArch`; a single helper `effectiveAttentionLength` consumed by `memory.ts` / `prefill.ts` / `decode.ts`; full-attention models retrofitted explicitly (`{ type: 'full' }`) so TS prevents silent drift as future attention types are added.

**Tech Stack:** TypeScript, Svelte 5, Vitest. No new deps.

**Spec:** `calc/docs/superpowers/specs/2026-05-12-sliding-window-attention-design.md`

---

## File Structure

```
src/engine/
  types.ts        # add AttentionConfig type, add `attention` to ModelArch
  memory.ts       # add effectiveAttentionLength helper; use for KV cache
  prefill.ts      # use helper for attention FLOPs
  decode.ts       # use helper for attention FLOPs
src/data/
  models.ts       # retrofit 12 existing entries; add Mistral 7B v0.1
test/
  fixtures.ts     # tag testGpu's testModel with attention
  engine/
    memory.test.ts    # extend with sliding-window cases + helper test
    prefill.test.ts   # extend with sliding-window case
    decode.test.ts    # extend with sliding-window case
    calc.test.ts      # extend with Mistral-7B integration test
```

---

## Task 1: Schema + retrofit (single commit)

Adding the required `attention` field to `ModelArch` breaks compilation for every existing model. Bundle the type addition with the retrofit so each commit leaves the repo in a buildable state.

**Files:**
- Modify: `calc/src/engine/types.ts`
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/fixtures.ts`

- [ ] **Step 1: Add the AttentionConfig type and field to types.ts**

Open `calc/src/engine/types.ts`. Find the `ModelArch` interface (around line 25). Just above it, insert:

```ts
export type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
```

Then inside the `ModelArch` interface, add `attention: AttentionConfig` as the last field, e.g.:

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
}
```

- [ ] **Step 2: Retrofit every entry in models.ts with attention: { type: 'full' }**

Open `calc/src/data/models.ts`. There are 12 entries. Add `attention: { type: 'full' }` as the last field of each. Example diff for the first entry:

```ts
{
  id: 'qwen3-1.7b', name: 'Qwen3 1.7B', family: 'qwen3',
  layers: 28, hiddenDim: 2048, intermediateDim: 6144,
  numHeads: 16, numKvHeads: 8, headDim: 128, vocabSize: 151936,
  paramCount: 1_720_000_000,
  attention: { type: 'full' }      // ← add this line
},
```

Apply the same one-line addition to all 12 entries. Use the `replace_all` Edit option carefully — entries differ. Safest: edit each entry by its unique `id` line.

- [ ] **Step 3: Update test fixture's testModel**

Open `calc/test/fixtures.ts`. The `testModel` const lacks `attention`. Add it:

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
  attention: { type: 'full' }
}
```

- [ ] **Step 4: Verify compile and tests still pass**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: no TS errors, all 33 existing tests pass (behavior unchanged because new field exists but no engine code reads it yet).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/types.ts calc/src/data/models.ts calc/test/fixtures.ts
git commit -m "feat(calc): add AttentionConfig schema and retrofit existing models as full"
```

No Co-Authored-By footer.

---

## Task 2: effectiveAttentionLength helper (TDD)

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Create: `calc/test/engine/sliding.test.ts`

- [ ] **Step 1: Write failing test**

```bash
cd /Users/yao/workspace/llm-perf/calc
cat > test/engine/sliding.test.ts <<'EOF'
import { describe, it, expect } from 'vitest'
import { effectiveAttentionLength } from '../../src/engine/memory'

describe('effectiveAttentionLength', () => {
  it('returns rawSeqlen for full attention', () => {
    expect(effectiveAttentionLength(100, { type: 'full' })).toBe(100)
    expect(effectiveAttentionLength(0, { type: 'full' })).toBe(0)
  })

  it('caps at window for sliding attention when raw exceeds window', () => {
    expect(effectiveAttentionLength(100, { type: 'sliding', window: 50 })).toBe(50)
  })

  it('returns raw when sliding window is larger than raw', () => {
    expect(effectiveAttentionLength(30, { type: 'sliding', window: 50 })).toBe(30)
  })

  it('returns raw when equal to window', () => {
    expect(effectiveAttentionLength(50, { type: 'sliding', window: 50 })).toBe(50)
  })
})
EOF
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npx vitest run test/engine/sliding.test.ts
```

Expected: FAIL with `effectiveAttentionLength` not exported from `memory.ts`.

- [ ] **Step 3: Implement the helper in memory.ts**

Open `calc/src/engine/memory.ts`. Near the top of the file, after the existing imports and before any function declarations, add:

```ts
import type { AttentionConfig } from './types'

export function effectiveAttentionLength(rawSeqlen: number, attention: AttentionConfig): number {
  if (attention.type === 'sliding') return Math.min(rawSeqlen, attention.window)
  return rawSeqlen
}
```

If `AttentionConfig` is already covered by the existing `import type { CalcInput, GpuVariant, MemoryResult } from './types'` line, extend that import rather than adding a new one. Concretely:

```ts
import type { AttentionConfig, CalcInput, GpuVariant, MemoryResult } from './types'
```

- [ ] **Step 4: Run test to verify it passes**

```bash
npx vitest run test/engine/sliding.test.ts
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/sliding.test.ts
git commit -m "feat(calc): add effectiveAttentionLength helper"
```

---

## Task 3: Memory math (KV cache bounded by window)

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/memory.test.ts`

- [ ] **Step 1: Add failing test for sliding-window KV cache**

Open `calc/test/engine/memory.test.ts`. Inside the existing `describe('computeMemory', ...)` block, append this test before the closing `})`:

```ts
  it('kvCachePerRequest caps at window for sliding attention', () => {
    // testModel uses full attention; build a sliding variant with window=8
    // (prompt+output=15, so should cap at 8 tokens instead of 15)
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 8 }
    }
    const input = { ...testInput, model: slidingModel }
    const m = computeMemory(input)
    // 16 bytes per token × 8 (window) = 128 bytes per request
    expect(m.kvCachePerRequest).toBe(128)
    // × concurrency 2 = 256 bytes
    expect(m.kvCacheTotal).toBe(256)
  })
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/memory.test.ts
```

Expected: the new test FAILS (kvCachePerRequest reports 240, not 128); the 8 existing tests still PASS.

- [ ] **Step 3: Update computeMemory to use the helper**

Open `calc/src/engine/memory.ts`. Locate the existing line:

```ts
const kvCachePerRequest = kvPerTokenPerRequest * seqlen
```

Replace it with:

```ts
const effSeqlen = effectiveAttentionLength(seqlen, model.attention)
const kvCachePerRequest = kvPerTokenPerRequest * effSeqlen
```

(The `seqlen` variable defined just above remains, used by the helper.)

- [ ] **Step 4: Run test to verify it passes**

```bash
npx vitest run test/engine/memory.test.ts
```

Expected: 9 tests PASS (8 old + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/memory.ts calc/test/engine/memory.test.ts
git commit -m "feat(calc): bound KV cache to sliding window"
```

---

## Task 4: Prefill math (attention term bounded by window)

**Files:**
- Modify: `calc/src/engine/prefill.ts`
- Modify: `calc/test/engine/prefill.test.ts`

- [ ] **Step 1: Add failing test for sliding-window prefill attention**

Open `calc/test/engine/prefill.test.ts`. Inside the existing `describe('computePrefill', ...)` block, append:

```ts
  it('attention term caps at window for sliding attention', () => {
    // testModel: layers=2, hiddenDim=4. Prompt=10. With window=5:
    // attention term = 2 × 2 × 10 × min(10, 5) × 4 = 800 (vs full's 1600)
    // MLP term = 2 × 1000 × 10 = 20000 (unchanged)
    // total = 20800
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 5 }
    }
    const input = { ...testInput, model: slidingModel }
    const slidingMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, slidingMemory)
    expect(p.flops).toBe(20800)
  })
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/prefill.test.ts
```

Expected: the new test FAILS (flops reports 21600, not 20800); 4 existing tests still PASS.

- [ ] **Step 3: Update computePrefill to use the helper**

Open `calc/src/engine/prefill.ts`. Add the import at the top (extending existing `roofline` import):

```ts
import { effectiveAttentionLength } from './memory'
import { roofline } from './roofline'
```

Then locate the existing flops calculation:

```ts
const flops =
  2 * model.paramCount * p +
  2 * model.layers * p * p * model.hiddenDim
```

Replace with:

```ts
const effP = effectiveAttentionLength(p, model.attention)
const flops =
  2 * model.paramCount * p +
  2 * model.layers * p * effP * model.hiddenDim
```

- [ ] **Step 4: Run test to verify it passes**

```bash
npx vitest run test/engine/prefill.test.ts
```

Expected: 5 tests PASS (4 old + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/prefill.ts calc/test/engine/prefill.test.ts
git commit -m "feat(calc): bound prefill attention FLOPs to sliding window"
```

---

## Task 5: Decode math (attention term bounded by window)

**Files:**
- Modify: `calc/src/engine/decode.ts`
- Modify: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Add failing test for sliding-window decode attention**

Open `calc/test/engine/decode.test.ts`. Inside the existing `describe('computeDecode', ...)` block, append:

```ts
  it('attention term caps at window for sliding attention', () => {
    // testModel: layers=2, hiddenDim=4, concurrency=2.
    // avgSeqlen = 10 + 5/2 = 12.5. With window=8, effSeqlen = 8.
    // flopsPerStep = (2 × 1000 + 2 × 2 × 8 × 4) × 2 = (2000 + 128) × 2 = 4256
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 8 }
    }
    const input = { ...testInput, model: slidingModel }
    const slidingMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, slidingMemory)
    expect(d.flopsPerStep).toBe(4256)
  })
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/yao/workspace/llm-perf/calc
npx vitest run test/engine/decode.test.ts
```

Expected: the new test FAILS (flopsPerStep reports 4400, not 4256); 4 existing tests still PASS.

- [ ] **Step 3: Update computeDecode to use the helper**

Open `calc/src/engine/decode.ts`. Add the import at the top (extending existing imports):

```ts
import { effectiveAttentionLength } from './memory'
import { roofline } from './roofline'
```

Then locate the existing flopsPerStep calculation:

```ts
const flopsPerStep =
  (2 * model.paramCount + 2 * model.layers * avgSeqlen * model.hiddenDim) *
  workload.concurrency
```

Replace with:

```ts
const effAvg = effectiveAttentionLength(avgSeqlen, model.attention)
const flopsPerStep =
  (2 * model.paramCount + 2 * model.layers * effAvg * model.hiddenDim) *
  workload.concurrency
```

- [ ] **Step 4: Run test to verify it passes**

```bash
npx vitest run test/engine/decode.test.ts
```

Expected: 5 tests PASS (4 old + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/decode.ts calc/test/engine/decode.test.ts
git commit -m "feat(calc): bound decode attention FLOPs to sliding window"
```

---

## Task 6: Add Mistral 7B v0.1 (verified) and integration test

**Files:**
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/engine/calc.test.ts`

- [ ] **Step 1: Verify Mistral 7B v0.1 config via WebFetch**

Use WebFetch on `https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json` with prompt:

> Return the raw JSON contents. Specifically: num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim if present, vocab_size, sliding_window.

Expected values (subject to the verification — these match Mistral 7B v0.1's actual config at time of writing):
- num_hidden_layers: 32
- hidden_size: 4096
- intermediate_size: 14336
- num_attention_heads: 32
- num_key_value_heads: 8
- head_dim: 128 (or hidden_size / num_attention_heads = 128)
- vocab_size: 32000
- sliding_window: 4096

If any value differs from the above, **use the value from the config**, not the value in this plan. The paramCount comes from the model card: `7_241_732_096`.

- [ ] **Step 2: Add Mistral 7B v0.1 entry to models.ts**

Open `calc/src/data/models.ts`. Find the Mistral block (between Gemma-3 and Phi). Add a new entry **before** the existing `mistral-small-3.1-24b` entry:

```ts
{
  id: 'mistral-7b-v0.1', name: 'Mistral 7B v0.1', family: 'mistral',
  layers: 32, hiddenDim: 4096, intermediateDim: 14336,
  numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
  paramCount: 7_241_732_096,
  attention: { type: 'sliding', window: 4096 }
},
```

(Adjust any field if Step 1 surfaced a discrepancy.)

- [ ] **Step 3: Add integration test**

Open `calc/test/engine/calc.test.ts`. At the bottom of the file (after the existing `'calculate — real data integration'` describe block), append:

```ts
describe('calculate — sliding window integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const mistral = MODELS.find(m => m.id === 'mistral-7b-v0.1')!

  it('Mistral 7B at 32k prompt: KV cache bounded by 4k window, not 32k', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: mistral,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // kv per token = 2 × 32 × 8 × 128 × 2 = 131072 bytes
    // bounded at window 4096 → 131072 × 4096 = 536870912 bytes
    expect(r.memory.kvCachePerRequest).toBe(131072 * 4096)
    // Sanity: if it were full attention this would be 131072 × 32768 (8× more)
  })
})
```

- [ ] **Step 4: Run all tests**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. Total test count = 33 (original) + 4 (Task 2 helper) + 3 (Tasks 3, 4, 5) + 1 (this task) = 41. All PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/data/models.ts calc/test/engine/calc.test.ts
git commit -m "feat(calc): add Mistral 7B v0.1 (sliding window 4096) with integration test"
```

---

## Self-Review Notes

Spec coverage:

- **Schema (Section 2 of spec)** — Task 1 adds `AttentionConfig`, retrofits 12 models + fixture
- **Math: KV cache (Section 3.1)** — Task 3
- **Math: Prefill (Section 3.2)** — Task 4
- **Math: Decode (Section 3.3)** — Task 5
- **Data update (Section 4)** — Task 6
- **Testing (Section 5)** — distributed across Tasks 2–6:
  - Helper unit test: Task 2
  - Memory test: Task 3
  - Prefill test: Task 4
  - Decode test: Task 5
  - Regression of 27 existing engine fixture tests: continuous (`npm test` after each task)
  - Integration test with Mistral 7B at long prompt: Task 6
- **UI (no required changes)** — no task needed
- **Evolution path** — addressed by the schema being a discriminated union ready for `'mla'` and `'hybrid'` variants

Type / API consistency check:

- `AttentionConfig` and `ModelArch.attention` defined in Task 1, consumed unchanged in Tasks 2–6
- `effectiveAttentionLength(rawSeqlen: number, attention: AttentionConfig): number` defined in Task 2, called identically in Tasks 3, 4, 5 (`effectiveAttentionLength(seqlen, model.attention)`)
- Existing function signatures (`computeMemory`, `computePrefill`, `computeDecode`) unchanged — internal-only edits

No placeholders or vague steps. Every code block contains the exact text to write.

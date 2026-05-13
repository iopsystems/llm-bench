# Shared Experts (MoE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `numSharedExperts: number` as a required field on the `'moe'` architecture variant; retrofit existing MoE entries; add DeepSeek V3 as the canonical user. The per-token math is unchanged — this is a schema-honesty / data-integrity change that unblocks data PRs for shared-expert MoE families.

**Architecture:** Single required field added to the existing `'moe'` variant on `ArchitectureConfig`. TypeScript refuses to compile any MoE entry without the field, forcing the retrofit. `activeParams(model)` and every downstream roofline calculation are untouched — `numSharedExperts` is metadata.

**Tech Stack:** TypeScript, Svelte 5, Vitest. No new deps.

**Spec:** `calc/docs/superpowers/specs/2026-05-12-shared-experts-design.md`

---

## File Structure

```
src/engine/
  types.ts        # add numSharedExperts: number to ArchitectureConfig 'moe' variant
src/data/
  models.ts       # retrofit Mixtral 8x7B (numSharedExperts: 0) + DeepSeek V2 (2);
                  # remove obsolete shared-experts comment above V2;
                  # add DeepSeek V3 entry
test/engine/
  sliding.test.ts # retrofit the activeParams MoE fixture
  prefill.test.ts # retrofit the MoE fixture
  decode.test.ts  # retrofit the two MoE fixtures
  calc.test.ts    # add DeepSeek V3 integration test
```

No new source files. All math, helpers, and existing tests are unchanged.

---

## Task 1: Schema + retrofit

Add the field, fix every compile error TS surfaces. Math is untouched; every existing assertion produces the same number byte-for-byte.

**Files:**
- Modify: `calc/src/engine/types.ts`
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/engine/sliding.test.ts`
- Modify: `calc/test/engine/prefill.test.ts`
- Modify: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Add the field to `ArchitectureConfig`**

In `calc/src/engine/types.ts`, locate the `ArchitectureConfig` definition (currently lines 36-38):

```ts
export type ArchitectureConfig =
  | { type: 'dense' }
  | { type: 'moe'; numExperts: number; numExpertsActive: number; activeParamCount: number }
```

Replace with:

```ts
export type ArchitectureConfig =
  | { type: 'dense' }
  | { type: 'moe';
      numExperts: number;          // routed-only
      numExpertsActive: number;    // top-K routed per token
      numSharedExperts: number;    // always-active expert count (separate from routed pool)
      activeParamCount: number;    // aggregate routed-active + shared (from model card)
    }
```

- [ ] **Step 2: Run check; observe compile errors flagging every MoE entry**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
```

Expected: type-check FAILS. The compiler should list every `architecture: { type: 'moe', ... }` literal that's missing `numSharedExperts` — two in `src/data/models.ts` plus four in test files.

- [ ] **Step 3: Retrofit Mixtral 8x7B entry**

In `calc/src/data/models.ts`, locate the Mixtral entry (currently lines 90-102). Update the `architecture` block:

```ts
    architecture: {
      type: 'moe',
      numExperts: 8,
      numExpertsActive: 2,
      numSharedExperts: 0,
      activeParamCount: 12_879_204_352
    }
```

- [ ] **Step 4: Retrofit DeepSeek V2 entry and remove obsolete comment**

In `calc/src/data/models.ts`, the DeepSeek V2 entry currently has an explanatory comment block above it (currently lines 120-124):

```ts
  // === DeepSeek ===
  // DeepSeek-V2 has 2 shared experts always active in addition to 6 routed
  // experts per token. The current schema doesn't have a numSharedExperts
  // field (deferred to a later feature) — the activeParamCount value below
  // is from the model card and already includes the shared-expert
  // contribution, so compute math comes out correctly.
```

Replace the entire block — keep just the family header — with:

```ts
  // === DeepSeek ===
```

Then update the V2 entry's `architecture` block (currently lines 131-136):

```ts
    architecture: {
      type: 'moe',
      numExperts: 160,
      numExpertsActive: 6,
      numSharedExperts: 2,
      activeParamCount: 21_000_000_000
    }
```

- [ ] **Step 5: Retrofit the test fixtures**

Four occurrences across three test files. Each is a synthetic MoE fixture used to assert the math handles MoE correctly. None of them are testing shared-expert behavior, so adding `numSharedExperts: 0` is the right value (preserves all existing assertions).

In `calc/test/engine/sliding.test.ts` (the `activeParams` describe block, currently around lines 25-32), update the moe fixture:

```ts
    const moe: ModelArch = {
      ...base,
      paramCount: 8000,
      architecture: {
        type: 'moe',
        numExperts: 8,
        numExpertsActive: 2,
        numSharedExperts: 0,
        activeParamCount: 2000
      }
    }
```

In `calc/test/engine/prefill.test.ts` (the MoE prefill test, currently around lines 57-65), update:

```ts
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        numSharedExperts: 0,
        activeParamCount: 250
      }
    }
```

In `calc/test/engine/decode.test.ts` (both MoE tests, currently around lines 58-66 and 78-86), update both fixtures:

```ts
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        numSharedExperts: 0,
        activeParamCount: 250
      }
    }
```

- [ ] **Step 6: Run check + tests**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. All 62 tests PASS byte-for-byte (no math change).

- [ ] **Step 7: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/engine/types.ts calc/src/data/models.ts calc/test/engine/sliding.test.ts calc/test/engine/prefill.test.ts calc/test/engine/decode.test.ts
git commit -m "feat(calc): add numSharedExperts field to MoE variant; retrofit existing entries"
```

No Co-Authored-By footer.

---

## Task 2: DeepSeek V3 data entry + integration test

Adds DeepSeek V3 as the canonical user for the new schema, plus an integration test that exercises `calculate()` end-to-end with V3's geometry.

V3 config values (verified against `deepseek-ai/DeepSeek-V3/config.json` on HuggingFace — public, not gated):
- `num_hidden_layers: 61`, `hidden_size: 7168`, `intermediate_size: 18432`
- `num_attention_heads: 128`, `num_key_value_heads: 128`, `vocab_size: 129280`
- `n_routed_experts: 256`, `n_shared_experts: 1`, `num_experts_per_tok: 8`
- `kv_lora_rank: 512`, `qk_rope_head_dim: 64`, `qk_nope_head_dim: 128`, `v_head_dim: 128`
- `headDim = qk_nope_head_dim + qk_rope_head_dim = 192` (matches V2's convention)

Model card values: `paramCount: 671_000_000_000`, `activeParamCount: 37_000_000_000`.

**Files:**
- Modify: `calc/src/data/models.ts`
- Modify: `calc/test/engine/calc.test.ts`

- [ ] **Step 1: Add the DeepSeek V3 entry**

In `calc/src/data/models.ts`, locate the DeepSeek V2 entry (after Task 1's retrofit). Add the V3 entry immediately after V2, before the next family header. Insert:

```ts
  {
    id: 'deepseek-v3', name: 'DeepSeek-V3', family: 'deepseek',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
    paramCount: 671_000_000_000,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 37_000_000_000
    }
  },
```

- [ ] **Step 2: Append the V3 integration test**

In `calc/test/engine/calc.test.ts`, append a new top-level describe block at the bottom of the file:

```ts
describe('calculate — DeepSeek V3 (MLA + shared-expert MoE) integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const dsv3 = MODELS.find(m => m.id === 'deepseek-v3')!

  it('DeepSeek V3 at 32k prompt: MLA KV cache matches V3 geometry', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: dsv3,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // MLA KV per-layer-per-token bytes = (kv_lora + rope) × bytes(fp16) = 576 × 2 = 1152
    // attendedSeq = layers × seq = 61 × 32768 = 1_998_848
    // kvCachePerRequest = 1152 × 1_998_848 = 2_302_672_896
    expect(r.memory.kvCachePerRequest).toBe(61 * (512 + 64) * 2 * 32768)

    // Sanity: GQA-equivalent (same kvHeads=128, headDim=192) would be ~85× larger.
    // 2 × 61 × 128 × 192 × 2 × 32768 vs 61 × 576 × 2 × 32768
    //   = (2 × 128 × 192) / 576 = 49152 / 576 = 85.33
    const gqaEquivalent = 2 * 61 * 128 * 192 * 2 * 32768
    expect(gqaEquivalent / r.memory.kvCachePerRequest).toBeGreaterThan(80)
  })

  it('DeepSeek V3 decode bytes/step use activeParams (37B), not paramCount (671B)', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
      model: dsv3,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // decode.bytesPerStep = activeParams × bytes(fp16) + kvCachePerRequest × concurrency
    //                    = 37e9 × 2 + small KV
    // Lower bound: 37e9 × 2 = 74 GB (small KV is negligible at batch=1, prompt=2048)
    const activeBytes = 37_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 5e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})
```

- [ ] **Step 3: Run check + tests**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check
npm test
```

Expected: type-check clean. 64 tests PASS (62 + 2 new V3 integration cases).

- [ ] **Step 4: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/src/data/models.ts calc/test/engine/calc.test.ts
git commit -m "feat(calc): add DeepSeek V3 (MLA + 256 routed + 1 shared experts)"
```

No Co-Authored-By footer.

---

## Done

After Task 2:
- Branch `feat/calc-shared-experts` has 2 implementation commits on top of the spec commit
- 64 / 64 tests pass
- `npm run check` clean
- `numSharedExperts` field exists on the MoE variant; Mixtral 8x7B (0), DeepSeek V2 (2), and DeepSeek V3 (1) populate it correctly
- The obsolete shared-experts comment above DeepSeek V2 is gone
- Controller will offer to open the PR

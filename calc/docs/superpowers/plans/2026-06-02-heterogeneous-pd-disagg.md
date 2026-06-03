# Heterogeneous PD-disagg — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the simulator's disagg block use different hardware and parallelism for prefill vs decode clusters, with two-phase memory checks and op-point pairing by name.

**Architecture:** Three layers of change. (1) **Engine**: split memory into per-side profiles (`prefillSide` / `decodeSide`), add `pairOpPoints()` helper, extend `CalcInput` with optional decode-side fields, refactor `calculate()` to loop over paired op-points and compute prefill/decode on their respective sides. (2) **Stores**: 5 new writables for decode-side hw + parallelism + the `heterogeneous` toggle, derived `decodeMultiDevice`, updated `simInputDisagg`. (3) **UI**: `DisaggInputPanel` grows the toggle + decode-side hw + parallelism. `Simulator.svelte` uses the two-sided OOM gate and renders op-pair labels with slashes when ids differ.

**Tech Stack:** TypeScript + Svelte 5; Vitest (node env, no DOM testing libs); npm from `calc/`; git from repo root `/Users/yao/workspace/llm-perf`. Branch `feat/heterogeneous-pd-disagg` (spec already committed at `4efd069`).

**Spec:** [`calc/docs/superpowers/specs/2026-06-02-heterogeneous-pd-disagg-design.md`](../specs/2026-06-02-heterogeneous-pd-disagg-design.md)

---

### Task 1: Two-phase memory model

**Why:** Per-side memory profiles unlock the heterogeneous use case (small-HBM decode + large-HBM prefill). Today's combined `memory.total` over-estimates the decode side (includes prefill activations the decode cluster never sees).

**Files:**
- Modify: `/Users/yao/workspace/llm-perf/calc/src/engine/types.ts` (`MemoryResult` interface around lines 427-444)
- Modify: `/Users/yao/workspace/llm-perf/calc/src/engine/memory.ts` (`computeMemory` function lines 126-186)
- Test: `/Users/yao/workspace/llm-perf/calc/test/engine/memory.test.ts` (file may not yet exist — check)

- [ ] **Step 1: Check whether `memory.test.ts` exists; create scaffold if not**

```bash
ls /Users/yao/workspace/llm-perf/calc/test/engine/memory.test.ts 2>/dev/null || echo "MISSING"
```

If `MISSING`, create the file with:

```ts
import { describe, it, expect } from 'vitest'
import { computeMemory } from '../../src/engine/memory'
import { testInput } from '../fixtures'

describe('computeMemory', () => {
  // tests added below
})
```

- [ ] **Step 2: Write the failing tests for per-side profiles**

Append inside `describe('computeMemory', ...)`:

```ts
  it('exposes decodeActivationsPeak much smaller than activationsPeak (prefill activations)', () => {
    const m = computeMemory(testInput)
    expect(m.decodeActivationsPeak).toBeLessThan(m.activationsPeak)
    // decode activations are O(1 × hidden), prefill are O(promptTokens × hidden) — gap >> 10x in test config
    expect(m.activationsPeak / m.decodeActivationsPeak).toBeGreaterThan(2)
  })

  it('prefillSide.total equals weights + kvCacheTotal + activationsPeak (= today\'s total)', () => {
    const m = computeMemory(testInput)
    expect(m.prefillSide.total).toBe(m.weights + m.kvCacheTotal + m.activationsPeak)
    expect(m.prefillSide.total).toBe(m.total)
  })

  it('decodeSide.total uses decodeActivationsPeak instead of prefill activations', () => {
    const m = computeMemory(testInput)
    expect(m.decodeSide.total).toBe(m.weights + m.kvCacheTotal + m.decodeActivationsPeak)
    expect(m.decodeSide.total).toBeLessThan(m.prefillSide.total)
  })

  it('hbmCapacityGB per side defaults to prefill variant when decodeAccelerator absent', () => {
    const m = computeMemory(testInput)
    expect(m.prefillSide.hbmCapacityGB).toBe(testInput.accelerator.variants[0].hbmCapacityGB)
    expect(m.decodeSide.hbmCapacityGB).toBe(testInput.accelerator.variants[0].hbmCapacityGB)
  })

  it('per-side fits flags computed against their respective HBM capacities', () => {
    const m = computeMemory(testInput)
    const cap = testInput.accelerator.variants[0].hbmCapacityGB * 1024 * 1024 * 1024
    expect(m.prefillSide.fits).toBe(m.prefillSide.total <= cap)
    expect(m.decodeSide.fits).toBe(m.decodeSide.total <= cap)
  })

  it('backward-compat: total/fits/headroom/hbmCapacityGB mirror prefillSide', () => {
    const m = computeMemory(testInput)
    expect(m.total).toBe(m.prefillSide.total)
    expect(m.fits).toBe(m.prefillSide.fits)
    expect(m.headroom).toBe(m.prefillSide.headroom)
    expect(m.hbmCapacityGB).toBe(m.prefillSide.hbmCapacityGB)
  })
```

- [ ] **Step 3: Run tests to verify failure**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- memory 2>&1 | tail -15
```
Expected: FAIL — `m.decodeActivationsPeak`, `m.prefillSide`, `m.decodeSide` are undefined.

- [ ] **Step 4: Extend `MemoryResult` in `types.ts`**

In `calc/src/engine/types.ts`, find `MemoryResult` interface (around line 427). Insert a new `MemorySide` interface above it and replace `MemoryResult` with:

```ts
export interface MemorySide {
  weights: number
  activations: number         // prefillActivationsPeak or decodeActivationsPeak
  kvCache: number             // = kvCacheTotal on both sides (prefill builds it; decode holds it)
  total: number               // sum of the above
  hbmCapacityGB: number       // capacity of this side's accelerator variant
  headroom: number
  fits: boolean
  perRank?: {
    weights: number
    kvCachePerRequest: number
    activations: number
    total: number
    headroom: number
    fits: boolean
  }
}

export interface MemoryResult {
  weights: number
  kvCachePerRequest: number
  kvCacheTotal: number
  activationsPeak: number              // = prefill activations (existing; scales with prompt)
  decodeActivationsPeak: number        // NEW: decode-side activations (scales with 1×hidden)
  prefillSide: MemorySide
  decodeSide: MemorySide
  // Backward-compat fields (= prefillSide values). Existing callers keep working.
  total: number
  hbmCapacityGB: number
  headroom: number
  fits: boolean
  perRank?: {
    weights: number
    kvCachePerRequest: number
    activationsPeak: number
    total: number
    headroom: number
    fits: boolean
  }
}
```

- [ ] **Step 5: Refactor `computeMemory` to populate per-side profiles**

In `calc/src/engine/memory.ts`, replace the `computeMemory` function (lines 126-186) with:

```ts
export function computeMemory(input: CalcInput): MemoryResult {
  const { model, quant, workload } = input
  const prefillVariant = findVariant(input)
  const seqlen = workload.promptTokens + workload.outputTokens

  const weights = model.paramCount * bytesOf(quant.weights)
  const kvPerLayerPerToken = kvBytesPerTokenPerLayer(model, quant.kv)
  const attendedSeqlen = attendedSeqlenSummedOverLayers(model, seqlen, true)
  const kvCachePerRequest =
    kvPerLayerPerToken * attendedSeqlen
    + linearAttentionStateBytes(model, quant.kv)
    + deltaStateBytes(model, quant.kv)
  const kvCacheTotal = kvCachePerRequest * workload.concurrency

  // Prefill activations: one big batched pass, scales with promptTokens × hidden.
  const activationsPeak =
    workload.concurrency * workload.promptTokens *
    (model.hiddenDim + model.intermediateDim) * bytesOf(quant.activations) * 2

  // Decode activations: single-token forward pass per layer; orders of magnitude smaller.
  const decodeActivationsPeak =
    workload.concurrency * 1 *
    (model.hiddenDim + model.intermediateDim) * bytesOf(quant.activations) * 2

  // Resolve decode-side variant; falls back to prefill when asymmetric fields absent.
  const decodeAccelerator = input.decodeAccelerator ?? input.accelerator
  const decodeVariantId = input.decodeAcceleratorVariantId ?? input.acceleratorVariantId
  const decodeVariant =
    decodeAccelerator.variants.find(v => v.id === decodeVariantId) ?? prefillVariant

  const prefillSide = buildSide(
    weights, kvCacheTotal, activationsPeak,
    prefillVariant.hbmCapacityGB,
    input.multiDevice, model, workload, kvCachePerRequest, activationsPeak
  )
  const decodeSide = buildSide(
    weights, kvCacheTotal, decodeActivationsPeak,
    decodeVariant.hbmCapacityGB,
    input.decodeMultiDevice ?? input.multiDevice, model, workload, kvCachePerRequest, decodeActivationsPeak
  )

  return {
    weights,
    kvCachePerRequest,
    kvCacheTotal,
    activationsPeak,
    decodeActivationsPeak,
    prefillSide,
    decodeSide,
    // Backward-compat: mirror prefillSide
    total: prefillSide.total,
    hbmCapacityGB: prefillSide.hbmCapacityGB,
    headroom: prefillSide.headroom,
    fits: prefillSide.fits,
    ...(prefillSide.perRank && {
      perRank: {
        weights: prefillSide.perRank.weights,
        kvCachePerRequest: prefillSide.perRank.kvCachePerRequest,
        activationsPeak: prefillSide.perRank.activations,
        total: prefillSide.perRank.total,
        headroom: prefillSide.perRank.headroom,
        fits: prefillSide.perRank.fits,
      }
    })
  }
}

function buildSide(
  weights: number,
  kvCacheTotal: number,
  activations: number,
  hbmCapacityGB: number,
  multiDevice: MultiDeviceConfig | undefined,
  model: ModelArch,
  workload: Workload,
  kvCachePerRequest: number,
  sideActivations: number,
): MemorySide {
  const total = weights + kvCacheTotal + activations
  const hbmCapacityBytes = hbmCapacityGB * BYTES_PER_GB
  const headroom = hbmCapacityBytes - total
  const fits = headroom >= 0

  let perRank: MemorySide['perRank'] = undefined
  if (multiDevice) {
    const divisors = perRankMemoryDivisors(
      multiDevice.parallelism,
      multiDevice.parallelismDegrees,
      model
    )
    const rankWeights = weights / divisors.weights
    const perReplicaConcurrency = workload.concurrency / divisors.replicas
    const rankKvPerRequest = kvCachePerRequest / divisors.kv
    const rankKvTotal = rankKvPerRequest * perReplicaConcurrency
    const rankActivations = sideActivations / divisors.activations
    const rankTotal = rankWeights + rankKvTotal + rankActivations
    const rankHeadroom = hbmCapacityBytes - rankTotal
    perRank = {
      weights: rankWeights,
      kvCachePerRequest: rankKvPerRequest,
      activations: rankActivations,
      total: rankTotal,
      headroom: rankHeadroom,
      fits: rankHeadroom >= 0,
    }
  }

  return {
    weights,
    activations,
    kvCache: kvCacheTotal,
    total,
    hbmCapacityGB,
    headroom,
    fits,
    ...(perRank && { perRank }),
  }
}
```

Also update the imports at the top of `memory.ts` to include `MultiDeviceConfig`, `ModelArch`, `Workload`, `MemorySide`:

```ts
import type {
  CalcInput, AcceleratorVariant, MemoryResult, MemorySide,
  MultiDeviceConfig, ModelArch, Workload, Dtype
} from './types'
```

(Check the existing import line and merge/extend as needed.)

- [ ] **Step 6: Verify tests pass + no regressions**

```bash
npm test 2>&1 | grep -E "(Tests |FAIL)" | tail -3
npm run check 2>&1 | tail -2
```
Expected: all green; 0 type errors.

- [ ] **Step 7: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/engine/types.ts calc/src/engine/memory.ts calc/test/engine/memory.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): two-phase memory model (prefillSide/decodeSide)

Splits activations into prefill (= promptTokens × hidden, today's value)
and decode (= 1 × hidden, orders of magnitude smaller). Returns per-side
profiles with their own HBM capacity, headroom, fits, and perRank slices.
Backward-compat: total/fits/headroom/hbmCapacityGB still mirror prefill
side (= today's behavior). Sets up the asymmetric-disagg use case where
a small-HBM decode cluster can fit even when prefill side wouldn't."
```

DO NOT add Co-Authored-By footer (project convention).

---

### Task 2: `opPoints.ts` — pair-by-name helper

**Why:** When prefill and decode hardware each expose multiple operating points (e.g. peak + achievable), match by id so users compare "peak with peak, achievable with achievable" rather than the cartesian product's crossed combos.

**Files:**
- Create: `/Users/yao/workspace/llm-perf/calc/src/engine/opPoints.ts`
- Create: `/Users/yao/workspace/llm-perf/calc/test/engine/opPoints.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `calc/test/engine/opPoints.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { pairOpPoints } from '../../src/engine/opPoints'
import type { AcceleratorVariant } from '../../src/engine/types'

const variant = (opIds: string[]): AcceleratorVariant => ({
  id: 'v', label: 'V', hbmCapacityGB: 80,
  operatingPoints: opIds.map(id => ({
    id, label: id, tflops: { fp16: 1 }, hbmBandwidthGBs: 1
  }))
})

describe('pairOpPoints', () => {
  it('pairs matched ids: peak/peak, achievable/achievable', () => {
    const pairs = pairOpPoints(variant(['peak', 'achievable']), variant(['peak', 'achievable']))
    expect(pairs).toHaveLength(2)
    expect(pairs[0]).toMatchObject({ id: 'peak' })
    expect(pairs[0].prefillOp.id).toBe('peak')
    expect(pairs[0].decodeOp.id).toBe('peak')
    expect(pairs[1]).toMatchObject({ id: 'achievable' })
  })

  it('symmetric (same variant on both sides) collapses to that variant\'s op list', () => {
    const v = variant(['peak'])
    const pairs = pairOpPoints(v, v)
    expect(pairs).toHaveLength(1)
    expect(pairs[0].id).toBe('peak')
    expect(pairs[0].prefillOp).toBe(pairs[0].decodeOp)
  })

  it('falls back to decode side\'s first op when prefill name has no match', () => {
    // Prefill has 'peak' + 'achievable'; decode only has 'peak'.
    // The 'achievable' prefill op pairs with decode\'s only op (peak).
    const pairs = pairOpPoints(variant(['peak', 'achievable']), variant(['peak']))
    expect(pairs).toHaveLength(2)
    expect(pairs[0]).toMatchObject({ id: 'peak' })
    expect(pairs[1].prefillOp.id).toBe('achievable')
    expect(pairs[1].decodeOp.id).toBe('peak')   // fallback to decode\'s first op
    expect(pairs[1].id).toBe('achievable/peak')  // composite id signals the cross-fallback
  })

  it('single op-point on prefill, multiple on decode: still iterates over prefill list', () => {
    const pairs = pairOpPoints(variant(['peak']), variant(['peak', 'achievable']))
    expect(pairs).toHaveLength(1)
    expect(pairs[0].prefillOp.id).toBe('peak')
    expect(pairs[0].decodeOp.id).toBe('peak')   // matched-by-name
    expect(pairs[0].id).toBe('peak')
  })
})
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- opPoints 2>&1 | tail -10
```
Expected: FAIL — module doesn\'t exist.

- [ ] **Step 3: Implement `opPoints.ts`**

Create `calc/src/engine/opPoints.ts`:

```ts
// Pair operating points across prefill and decode variants for heterogeneous
// PD-disagg. Matches by id ("peak" with "peak", "achievable" with "achievable").
// If a prefill op has no matching id on the decode side, falls back to the
// decode side\'s first op and synthesizes a composite pair id like
// "prefillId/decodeId" so the UI can disambiguate.

import type { AcceleratorVariant, AcceleratorOperatingPoint } from './types'

export interface OpPointPair {
  prefillOp: AcceleratorOperatingPoint
  decodeOp: AcceleratorOperatingPoint
  id: string
}

export function pairOpPoints(
  prefill: AcceleratorVariant,
  decode: AcceleratorVariant,
): OpPointPair[] {
  return prefill.operatingPoints.map(prefillOp => {
    const matched = decode.operatingPoints.find(o => o.id === prefillOp.id)
    const decodeOp = matched ?? decode.operatingPoints[0]
    const id = matched ? prefillOp.id : `${prefillOp.id}/${decodeOp.id}`
    return { prefillOp, decodeOp, id }
  })
}
```

- [ ] **Step 4: Run tests + typecheck**

```bash
npm test -- opPoints 2>&1 | tail -10
npm run check 2>&1 | tail -2
```
Expected: 4 cases green; 0 type errors.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/engine/opPoints.ts calc/test/engine/opPoints.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): pairOpPoints helper — match-by-name with fallback for heterogeneous P/D"
```

---

### Task 3: `CalcInput` decode fields + `MultiDeviceConfig` override args

**Why:** The engine needs to know the decode-side accelerator/variant/multiDevice when heterogeneous is on. `computePrefill`/`computeDecode` get an optional override arg so `calc.ts` can call them with either side\'s `multiDevice`.

**Files:**
- Modify: `/Users/yao/workspace/llm-perf/calc/src/engine/types.ts` (`CalcInput` interface around lines 410-425)
- Modify: `/Users/yao/workspace/llm-perf/calc/src/engine/prefill.ts`
- Modify: `/Users/yao/workspace/llm-perf/calc/src/engine/decode.ts`

No new tests in this task — the fields are just schema additions; their use is tested in Task 4 (calc.ts).

- [ ] **Step 1: Add decode-side fields to `CalcInput`**

In `calc/src/engine/types.ts`, find `CalcInput` (around line 410) and add three optional fields after `multiDevice?:`:

```ts
export interface CalcInput {
  accelerator: AcceleratorSpec
  acceleratorVariantId: string
  model: ModelArch
  quant: Quantization
  workload: Workload
  multiDevice?: MultiDeviceConfig

  // Decode cluster, used in heterogeneous PD-disagg. Absent ⇒ engine reuses
  // prefill side for both phases (= v1 symmetric).
  decodeAccelerator?: AcceleratorSpec
  decodeAcceleratorVariantId?: string
  decodeMultiDevice?: MultiDeviceConfig

  // PD-disagg: prefill ships KV to decode over this fabric (InterconnectSpec.id).
  // Undefined = integrated serving (no transfer cost).
  disaggKvTransferFabricId?: string
  // When disagg is active, whether prefill emits the first decoded token locally
  // while KV transfer streams in parallel. Defaults true.
  disaggFirstTokenOnPrefill?: boolean
}
```

- [ ] **Step 2: Add `multiDevice` override arg to `computePrefill`**

In `calc/src/engine/prefill.ts`, replace the function signature + multi-device resolution:

```ts
export function computePrefill(
  input: CalcInput,
  opPoint: AcceleratorOperatingPoint,
  memory: MemoryResult,
  multiDeviceOverride?: MultiDeviceConfig,
): PerfTier['prefill'] {
  const { model, quant, workload } = input
  const p = workload.promptTokens
  const multiDevice = multiDeviceOverride ?? input.multiDevice

  const flops =
    2 * activeParams(model) * p +
    2 * p * attendedSeqlenSummedOverLayers(model, p) * attentionDim(model) +
    p * linearAttentionFlopsPerToken(model) +
    p * deltaAttentionFlopsPerToken(model)
  const bytes = memory.weights + memory.activationsPeak

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  let commsBytes: number | undefined = undefined
  let interconnectBwGBs: number | undefined = undefined
  if (multiDevice) {
    const B = workload.promptTokens * workload.concurrency
    commsBytes = commsBytesPerStep(
      multiDevice.parallelism,
      multiDevice.parallelismDegrees,
      model,
      B,
      quant.activations
    )
    const ic = INTERCONNECTS.find(i => i.id === multiDevice.system.interconnectId)
    if (ic) interconnectBwGBs = ic.perDirectionGBs ?? ic.perGpuBandwidthGBs / 2
  }

  const { timeS, regime } = roofline({
    flops, bytes, tflops, bwGBs: opPoint.hbmBandwidthGBs,
    commsBytes, interconnectBwGBs
  })
  return { flops, bytes, timeS, regime }
}
```

Add `MultiDeviceConfig` to the import at the top:

```ts
import type {
  CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier, MultiDeviceConfig
} from './types'
```

(Merge with existing import if it already exists.)

- [ ] **Step 3: Add `multiDevice` override arg to `computeDecode`**

In `calc/src/engine/decode.ts`, do the analogous change. Signature becomes:

```ts
export function computeDecode(
  input: CalcInput,
  opPoint: AcceleratorOperatingPoint,
  memory: MemoryResult,
  multiDeviceOverride?: MultiDeviceConfig,
): PerfTier['decode'] {
  const { model, quant, workload } = input
  const multiDevice = multiDeviceOverride ?? input.decodeMultiDevice ?? input.multiDevice
  const avgSeqlen = workload.promptTokens + workload.outputTokens / 2

  // ... rest of body unchanged except all `input.multiDevice` references become `multiDevice`
}
```

Concretely, find the `if (input.multiDevice) { ... }` block and change to `if (multiDevice) { ... }`, and the inner `input.multiDevice.parallelism` → `multiDevice.parallelism`, `input.multiDevice.parallelismDegrees` → `multiDevice.parallelismDegrees`, `input.multiDevice!.system.interconnectId` → `multiDevice.system.interconnectId`.

Note the default resolution chain: `multiDeviceOverride ?? input.decodeMultiDevice ?? input.multiDevice`. Decode functions default to the decode side (heterogeneous-aware); prefill functions default to the prefill side (heterogeneous-agnostic).

Add `MultiDeviceConfig` to the import at the top of decode.ts:

```ts
import type {
  CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier, MultiDeviceConfig
} from './types'
```

- [ ] **Step 4: Run full suite + typecheck**

```bash
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
```
Expected: all green; 0 type errors. (No new tests yet — Task 4 exercises this in `calc.ts`.)

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/engine/types.ts calc/src/engine/prefill.ts calc/src/engine/decode.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): CalcInput decode fields + multiDevice override on prefill/decode"
```

---

### Task 4: `calc.ts` — paired op-points loop + heterogeneous TTFT

**Why:** This is the engine\'s integration step. The loop iterates over paired op-points; each pair computes prefill on prefill hw and decode on decode hw. For the `firstTokenOnPrefill=true` case, an extra `computeDecode` call uses the PREFILL multiDevice so the TTFT formula reflects the prefill cluster generating token #1.

**Files:**
- Modify: `/Users/yao/workspace/llm-perf/calc/src/engine/calc.ts`
- Test: `/Users/yao/workspace/llm-perf/calc/test/engine/calc.test.ts` (add tests)

- [ ] **Step 1: Write the failing tests**

In `calc/test/engine/calc.test.ts`, append inside the top-level `describe('calculate', ...)` block:

```ts
  it('heterogeneous P/D: decode perf uses decode-side accelerator', () => {
    // Build a fixture with decodeAccelerator different from prefill side.
    // Use ACCELERATORS to pick two real entries.
    const { ACCELERATORS } = require('../../src/data')
    const h100 = ACCELERATORS.find((a: any) => a.id === 'h100')!
    const h200 = ACCELERATORS.find((a: any) => a.id === 'h200')!
    const inp = {
      ...testInput,
      accelerator: h100,
      acceleratorVariantId: h100.variants[0].id,
      decodeAccelerator: h200,
      decodeAcceleratorVariantId: h200.variants[0].id,
    }
    const result = calculate(inp)
    // h200 has higher HBM bandwidth → decode tpot should be lower than the symmetric h100 case.
    const symmetric = calculate({ ...inp, decodeAccelerator: undefined, decodeAcceleratorVariantId: undefined })
    const op = Object.keys(result.perf)[0]
    expect(result.perf[op].decode.timePerTokenS).toBeLessThan(symmetric.perf[op].decode.timePerTokenS)
  })

  it('heterogeneous P/D with firstTokenOnPrefill=true: TTFT uses prefill-cluster decode-step time', () => {
    const { ACCELERATORS } = require('../../src/data')
    const h100 = ACCELERATORS.find((a: any) => a.id === 'h100')!
    const h200 = ACCELERATORS.find((a: any) => a.id === 'h200')!
    const inp = {
      ...testInput,
      accelerator: h100,
      acceleratorVariantId: h100.variants[0].id,
      decodeAccelerator: h200,
      decodeAcceleratorVariantId: h200.variants[0].id,
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: true,
    }
    const result = calculate(inp)
    const op = Object.keys(result.perf)[0]
    // TTFT = prefill.timeS + (decode step on prefill cluster's hw, NOT decode cluster's).
    // The decode-step-on-prefill is computed from prefill's TFLOPS/HBM, not h200's.
    const ttft = result.perf[op].ttftS
    const prefillTime = result.perf[op].prefill.timeS
    // Decode-step on prefill cluster is some positive delta; assert TTFT > prefillTime by a small margin.
    expect(ttft).toBeGreaterThan(prefillTime)
    // And TTFT must NOT equal prefill + decode-cluster-tpot (which would be the h200 number).
    const decodeClusterTpot = result.perf[op].decode.timePerTokenS
    expect(Math.abs(ttft - (prefillTime + decodeClusterTpot))).toBeGreaterThan(1e-9)
  })

  it('symmetric (no decode fields) still works — backward compat', () => {
    const result = calculate(testInput)
    expect(Object.keys(result.perf).length).toBeGreaterThan(0)
  })
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- calc.test 2>&1 | tail -15
```
Expected: FAIL — heterogeneous tests fail because calc.ts still ignores decode-side fields.

- [ ] **Step 3: Refactor `calc.ts`**

In `calc/src/engine/calc.ts`, replace the entire `calculate` function with:

```ts
import type { CalcInput, CalcResult, PerfTier } from './types'
import { bytesOf } from './dtypes'
import { computeMemory } from './memory'
import { computePrefill } from './prefill'
import { computeDecode } from './decode'
import { DerivationBuilder } from './derivation'
import { INTERCONNECTS } from '../data/interconnects'
import { pairOpPoints } from './opPoints'

export function calculate(input: CalcInput): CalcResult {
  // Resolve both sides. Decode side falls back to prefill when fields absent.
  const prefillVariant = input.accelerator.variants.find(v => v.id === input.acceleratorVariantId)
  if (!prefillVariant) {
    throw new Error(`Variant ${input.acceleratorVariantId} not in ${input.accelerator.id}`)
  }
  const decodeAccelerator = input.decodeAccelerator ?? input.accelerator
  const decodeVariantId = input.decodeAcceleratorVariantId ?? input.acceleratorVariantId
  const decodeVariant = decodeAccelerator.variants.find(v => v.id === decodeVariantId)
  if (!decodeVariant) {
    throw new Error(`Variant ${decodeVariantId} not in ${decodeAccelerator.id}`)
  }

  // Validate activations dtype against both sides\' operating points.
  for (const op of prefillVariant.operatingPoints) {
    if (op.tflops[input.quant.activations] === undefined) {
      const supported = Object.keys(op.tflops).join(', ')
      throw new Error(
        `${input.accelerator.name} ${prefillVariant.label} has no ${input.quant.activations} ` +
        `compute throughput. Try: ${supported}.`
      )
    }
  }
  for (const op of decodeVariant.operatingPoints) {
    if (op.tflops[input.quant.activations] === undefined) {
      const supported = Object.keys(op.tflops).join(', ')
      throw new Error(
        `${decodeAccelerator.name} ${decodeVariant.label} has no ${input.quant.activations} ` +
        `compute throughput. Try: ${supported}.`
      )
    }
  }

  const memory = computeMemory(input)
  const d = new DerivationBuilder()

  d.add('weights', 'paramCount × bytes(weight_dtype)', memory.weights, 'bytes')
  d.add(
    'kv per token per request',
    '2 × layers × kv_heads × head_dim × bytes(kv_dtype)',
    memory.kvCachePerRequest / (input.workload.promptTokens + input.workload.outputTokens),
    'bytes'
  )
  d.add('kv per request', 'kv_per_token × (prompt + output)', memory.kvCachePerRequest, 'bytes')
  d.add('kv total', 'kv_per_request × concurrency', memory.kvCacheTotal, 'bytes')
  d.add(
    'activations peak (prefill, coarse)',
    'concurrency × prompt × (hidden + intermediate) × bytes(act_dtype) × 2',
    memory.activationsPeak, 'bytes'
  )
  d.add(
    'activations peak (decode, coarse)',
    'concurrency × 1 × (hidden + intermediate) × bytes(act_dtype) × 2',
    memory.decodeActivationsPeak, 'bytes'
  )
  d.add('prefill side total', 'weights + kv_total + prefill_activations', memory.prefillSide.total, 'bytes')
  d.add('decode side total',  'weights + kv_total + decode_activations',  memory.decodeSide.total,  'bytes')

  // Disagg KV transfer (unchanged).
  let kvTransferS = 0
  if (input.disaggKvTransferFabricId) {
    const fab = INTERCONNECTS.find(i => i.id === input.disaggKvTransferFabricId)
    if (fab) {
      const bw = fab.perDirectionGBs ?? fab.perGpuBandwidthGBs / 2
      kvTransferS = memory.kvCachePerRequest / (bw * 1e9)
    }
  }
  const firstTokenOnPrefill = input.disaggFirstTokenOnPrefill ?? true

  const perf: Record<string, PerfTier> = {}
  for (const pair of pairOpPoints(prefillVariant, decodeVariant)) {
    const prefill = computePrefill(input, pair.prefillOp, memory)  // uses input.multiDevice (prefill side)
    const decode  = computeDecode(input, pair.decodeOp,  memory)   // uses input.decodeMultiDevice ?? input.multiDevice (decode side)

    // TTFT case-B: first decode step runs on the PREFILL cluster, so use
    // prefill hw\'s TFLOPS/HBM. Override multiDevice to prefill side.
    let firstStepOnPrefillS = decode.timePerTokenS  // fallback (covers symmetric case)
    if (kvTransferS > 0 && firstTokenOnPrefill) {
      const onPrefill = computeDecode(input, pair.prefillOp, memory, input.multiDevice)
      firstStepOnPrefillS = onPrefill.timePerTokenS
    }

    const ttftS = kvTransferS > 0 && firstTokenOnPrefill
      ? prefill.timeS + firstStepOnPrefillS
      : prefill.timeS + kvTransferS

    perf[pair.id] = {
      prefill, decode,
      ttftS,
      kvTransferS,
      inputTokenRate: input.workload.promptTokens / prefill.timeS,
      outputTokenRate: decode.aggregateTokensPerS,
      ...(pair.prefillOp.tflopsSources && { tflopsSources: pair.prefillOp.tflopsSources }),
      ...(pair.prefillOp.bandwidthSources && { bandwidthSources: pair.prefillOp.bandwidthSources }),
      ...(pair.prefillOp.asOf && { asOf: pair.prefillOp.asOf }),
      ...(pair.prefillOp.notes && { notes: pair.prefillOp.notes })
    }
    d.add(`prefill time @ ${pair.id}`, 'max(prefill_flops / tflops, prefill_bytes / bw)', prefill.timeS, 's')
    if (kvTransferS > 0) {
      d.add(
        `kv transfer time @ ${pair.id}`,
        firstTokenOnPrefill
          ? 'kv_cache_per_request / disagg_fabric_bw (overlapped with first decode)'
          : 'kv_cache_per_request / disagg_fabric_bw',
        kvTransferS, 's'
      )
    }
    d.add(`decode time per token @ ${pair.id}`, 'max(decode_flops / tflops, decode_bytes / bw)', decode.timePerTokenS, 's')
  }

  void bytesOf

  return { memory, perf, derivation: d.steps() }
}
```

(Two notable additions: `decodeVariant` resolution + per-side dtype validation; the inner loop now uses `pairOpPoints()`; the `firstStepOnPrefillS` extra `computeDecode` call for case-B TTFT.)

- [ ] **Step 4: Run tests + typecheck**

```bash
npm test -- calc.test 2>&1 | tail -15
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
```
Expected: all 3 new tests green + full suite green + 0 type errors.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/engine/calc.ts calc/test/engine/calc.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): heterogeneous P/D in calculate() — paired op-points + prefill-cluster TTFT"
```

---

### Task 5: Stores — decode-side writables + `heterogeneous` + `simInputDisagg`

**Why:** UI components read/write the decode-side configuration through Svelte stores. `simInputDisagg` derives the heterogeneous-aware `CalcInput` for the disagg block.

**Files:**
- Modify: `/Users/yao/workspace/llm-perf/calc/src/ui/stores.ts`
- Test: `/Users/yao/workspace/llm-perf/calc/test/ui/sim-disagg-stores.test.ts` (add tests; file exists from prior PR)

- [ ] **Step 1: Write the failing tests**

In `calc/test/ui/sim-disagg-stores.test.ts`, append at the bottom (before the closing of the file):

```ts
import {
  decodeAcceleratorId, decodeVariantId, decodeSystemId,
  decodeParallelismOverride, heterogeneous, decodeMultiDevice
} from '../../src/ui/stores'

describe('heterogeneous P/D — store wiring', () => {
  beforeEach(() => {
    heterogeneous.set(false)
    decodeAcceleratorId.set('')
    decodeVariantId.set('')
    decodeSystemId.set('')
    decodeParallelismOverride.set(null)
    workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 1 })
    disaggKvTransferFabricId.set('roce-400')
    disaggFirstTokenOnPrefill.set(true)
  })

  it('when heterogeneous=false: simInputDisagg has no decode-side fields', () => {
    const inp = get(simInputDisagg)!
    expect(inp.decodeAccelerator).toBeUndefined()
    expect(inp.decodeAcceleratorVariantId).toBeUndefined()
    expect(inp.decodeMultiDevice).toBeUndefined()
  })

  it('when heterogeneous=true with no decode-side selections: falls back to prefill on every field', () => {
    heterogeneous.set(true)
    const inp = get(simInputDisagg)!
    // Decode-side stores are empty → engine sees same accelerator as prefill.
    expect(inp.decodeAccelerator).toBe(inp.accelerator)
    expect(inp.decodeAcceleratorVariantId).toBe(inp.acceleratorVariantId)
  })

  it('when heterogeneous=true and decodeAcceleratorId set: decode side resolves to that accelerator', () => {
    heterogeneous.set(true)
    decodeAcceleratorId.set('h200')
    decodeVariantId.set('sxm-141')
    const inp = get(simInputDisagg)!
    expect(inp.decodeAccelerator?.id).toBe('h200')
    expect(inp.decodeAcceleratorVariantId).toBe('sxm-141')
  })
})
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- sim-disagg-stores 2>&1 | tail -15
```
Expected: FAIL — new exports don\'t exist.

- [ ] **Step 3: Add the new writables + derived in `stores.ts`**

In `calc/src/ui/stores.ts`, after the existing `disaggFirstTokenOnPrefill` writable (around line 30), add:

```ts
// Heterogeneous PD-disagg — separate hw + parallelism for the decode cluster.
// All decode-side stores have empty / null defaults; when `heterogeneous` is
// false they are ignored. When true, any empty/null field falls back to the
// prefill-side value (lets the user toggle into asymmetric mode and change
// knobs one at a time).
export const decodeAcceleratorId        = writable<string>('')
export const decodeVariantId            = writable<string>('')
export const decodeSystemId             = writable<string>('')
export const decodeParallelismOverride  = writable<ParallelismConfig | null>(null)
export const heterogeneous              = writable<boolean>(false)
```

After the existing `multiDevice` derived store (around line 72), add a sibling for the decode side:

```ts
export const decodeMultiDevice: Readable<MultiDeviceConfig | undefined> = derived(
  [decodeSystemId, modelId, decodeParallelismOverride],
  ([$decodeSystemId, $modelId, $override]) => {
    if (!$decodeSystemId) return undefined
    const system = SYSTEMS.find(s => s.id === $decodeSystemId)
    const model = MODELS.find(m => m.id === $modelId)
    if (!system || !model) return undefined
    const pc = $override ?? defaultParallelism(system, model)
    return {
      system,
      parallelism: pc.parallelism,
      parallelismDegrees: pc.parallelismDegrees,
    }
  }
)
```

Then replace the existing `simInputDisagg` derived (around line 134) with the heterogeneous-aware version:

```ts
export const simInputDisagg: Readable<CalcInput | null> = derived(
  [input, heterogeneous, decodeAcceleratorId, decodeVariantId,
   decodeSystemId, decodeMultiDevice],
  ([$input, $het, $decodeAcceleratorId, $decodeVariantId,
    $decodeSystemId, $decodeMultiDevice]) => {
    if (!$input) return null
    const base: CalcInput = {
      ...$input,
      workload: { ...$input.workload, concurrency: 1 },
    }
    if (!$het) return base
    // Heterogeneous: spread decode-side overrides. Each field falls back to
    // the prefill side when the corresponding decode store is empty.
    let decodeAccelerator = $input.accelerator
    let decodeAcceleratorVariantId = $input.acceleratorVariantId
    if ($decodeSystemId && $decodeMultiDevice) {
      decodeAccelerator = ACCELERATORS.find(a => a.id === $decodeMultiDevice.system.accelerator.id) ?? $input.accelerator
      decodeAcceleratorVariantId = $decodeMultiDevice.system.accelerator.variantId
    } else if ($decodeAcceleratorId) {
      const found = ACCELERATORS.find(a => a.id === $decodeAcceleratorId)
      if (found) {
        decodeAccelerator = found
        decodeAcceleratorVariantId = $decodeVariantId || found.variants[0].id
      }
    }
    const decodeMD = $decodeMultiDevice ?? $input.multiDevice
    return {
      ...base,
      decodeAccelerator,
      decodeAcceleratorVariantId,
      ...(decodeMD && { decodeMultiDevice: decodeMD }),
    }
  }
)
```

- [ ] **Step 4: Run targeted tests + full suite + typecheck**

```bash
npm test -- sim-disagg-stores 2>&1 | tail -15
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
```
Expected: all targeted tests green; full suite green; 0 type errors.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/stores.ts calc/test/ui/sim-disagg-stores.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): heterogeneous P/D stores + decodeMultiDevice + simInputDisagg overrides"
```

---

### Task 6: `share.ts` — `het` + decode-side URL keys

**Why:** Sharable URLs need to round-trip heterogeneous state. New keys: `het=1`, `a2`, `v2`, `s2`, `p2`. All emitted only when `heterogeneous === true`.

**Files:**
- Modify: `/Users/yao/workspace/llm-perf/calc/src/ui/share.ts`
- Test: `/Users/yao/workspace/llm-perf/calc/test/ui/share-route.test.ts` (add a new describe block)

- [ ] **Step 1: Write the failing tests**

In `calc/test/ui/share-route.test.ts`, append at the bottom:

```ts
describe('heterogeneous P/D URL state', () => {
  const base = {
    acceleratorId: 'h100', variantId: 'sxm-80', systemId: '', modelId: 'llama-3.3-70b',
    quant: { weights: 'bf16', kv: 'fp16', activations: 'bf16' } as const,
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
    parallelismOverride: null,
    disaggKvTransferFabricId: 'roce-400',
    disaggFirstTokenOnPrefill: true,
  }

  it('omits all decode-side keys when heterogeneous=false', () => {
    const enc = encodeState({ ...base, heterogeneous: false,
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141',
      decodeSystemId: '', decodeParallelismOverride: null })
    expect(enc).not.toContain('het=')
    expect(enc).not.toContain('a2=')
    expect(enc).not.toContain('v2=')
  })

  it('emits het=1 + a2/v2 when heterogeneous=true with single-chip decode side', () => {
    const enc = encodeState({ ...base, heterogeneous: true,
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141',
      decodeSystemId: '', decodeParallelismOverride: null })
    expect(enc).toContain('het=1')
    expect(enc).toContain('a2=h200')
    expect(enc).toContain('v2=sxm-141')
    expect(enc).not.toContain('s2=')
  })

  it('emits s2 instead of a2/v2 when decode side is multi-device', () => {
    const enc = encodeState({ ...base, heterogeneous: true,
      decodeAcceleratorId: '', decodeVariantId: '',
      decodeSystemId: 'hgx-h200-8', decodeParallelismOverride: null })
    expect(enc).toContain('s2=hgx-h200-8')
    expect(enc).not.toMatch(/(^|&)a2=/)
    expect(enc).not.toMatch(/(^|&)v2=/)
  })

  it('round-trips: encode then decode preserves heterogeneous state', () => {
    const original = { ...base, heterogeneous: true,
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141',
      decodeSystemId: '', decodeParallelismOverride: null }
    const round = decodeState(encodeState(original))
    expect(round.heterogeneous).toBe(true)
    expect(round.decodeAcceleratorId).toBe('h200')
    expect(round.decodeVariantId).toBe('sxm-141')
  })

  it('URL without het keys decodes to non-heterogeneous (no decode-side state)', () => {
    const round = decodeState('a=h100&v=sxm-80&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=1')
    expect(round.heterogeneous).toBeUndefined()
    expect(round.decodeAcceleratorId).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- share-route 2>&1 | tail -15
```
Expected: FAIL — `ShareableState` doesn\'t have the new fields, encoder/decoder don\'t handle them.

- [ ] **Step 3: Extend `ShareableState` interface**

In `calc/src/ui/share.ts`, find the `ShareableState` interface (around line 39) and add the new fields:

```ts
export interface ShareableState {
  acceleratorId: string
  variantId: string
  systemId: string
  modelId: string
  quant: Quantization
  workload: Workload
  parallelismOverride: ParallelismConfig | null
  disaggKvTransferFabricId: string
  disaggFirstTokenOnPrefill: boolean

  // Heterogeneous PD-disagg — only encoded when heterogeneous is true.
  heterogeneous: boolean
  decodeAcceleratorId: string
  decodeVariantId: string
  decodeSystemId: string
  decodeParallelismOverride: ParallelismConfig | null
}
```

- [ ] **Step 4: Update `encodeState` to emit decode-side keys**

In `calc/src/ui/share.ts`, find the end of `encodeState` (just before the `return p.toString()` line, around line 78). Insert the heterogeneous-encoding block:

```ts
  if (state.heterogeneous) {
    p.set('het', '1')
    if (state.decodeSystemId) {
      p.set('s2', state.decodeSystemId)
    } else if (state.decodeAcceleratorId) {
      p.set('a2', state.decodeAcceleratorId)
      if (state.decodeVariantId) p.set('v2', state.decodeVariantId)
    }
    if (state.decodeParallelismOverride) {
      p.set('p2', encodeParallelism(state.decodeParallelismOverride))
    }
  }
  return p.toString()
```

(The `return p.toString()` was the last line; the new block goes right before it.)

- [ ] **Step 5: Update `decodeState` to parse decode-side keys**

In `calc/src/ui/share.ts`, find the end of `decodeState` (just before the `return out` line, around line 162). Insert the heterogeneous-decoding block:

```ts
  if (params.get('het') === '1') {
    out.heterogeneous = true
    const s2 = params.get('s2')
    if (s2 !== null) {
      const sys = SYSTEMS.find(x => x.id === s2)
      if (sys) {
        out.decodeSystemId = s2
        out.decodeAcceleratorId = sys.accelerator.id
        out.decodeVariantId = sys.accelerator.variantId
      }
    } else if (params.has('a2')) {
      const a2 = params.get('a2')!
      const accel = ACCELERATORS.find(x => x.id === a2)
      if (accel) {
        out.decodeSystemId = ''
        out.decodeAcceleratorId = a2
        const v2 = params.get('v2')
        out.decodeVariantId = v2 && accel.variants.find(x => x.id === v2)
          ? v2 : accel.variants[0].id
      }
    }
    if (params.has('p2')) {
      const pc = decodeParallelism(params.get('p2')!)
      out.decodeParallelismOverride = pc ?? null
    }
  }

  return out
```

- [ ] **Step 6: Update `readStoreState` and `applyToStores`**

In `calc/src/ui/share.ts`, find `readStoreState()` (around line 189) and extend the returned object to include the new fields by reading from the new stores. Add imports at the top if needed.

First, ensure the imports include the new stores:

```ts
import {
  acceleratorId, variantId, systemId, modelId,
  parallelismOverride, disaggKvTransferFabricId, disaggFirstTokenOnPrefill,
  quant, workload,
  heterogeneous, decodeAcceleratorId, decodeVariantId, decodeSystemId,
  decodeParallelismOverride,
} from './stores'
```

Then update `readStoreState`:

```ts
function readStoreState(): ShareableState {
  return {
    acceleratorId: get(acceleratorId),
    variantId: get(variantId),
    systemId: get(systemId),
    modelId: get(modelId),
    quant: get(quant),
    workload: get(workload),
    parallelismOverride: get(parallelismOverride),
    disaggKvTransferFabricId: get(disaggKvTransferFabricId),
    disaggFirstTokenOnPrefill: get(disaggFirstTokenOnPrefill),
    heterogeneous: get(heterogeneous),
    decodeAcceleratorId: get(decodeAcceleratorId),
    decodeVariantId: get(decodeVariantId),
    decodeSystemId: get(decodeSystemId),
    decodeParallelismOverride: get(decodeParallelismOverride),
  }
}
```

Then extend `applyToStores` (around line 203):

```ts
function applyToStores(partial: Partial<ShareableState>): void {
  // Existing writes (unchanged)…
  if (partial.acceleratorId !== undefined) acceleratorId.set(partial.acceleratorId)
  if (partial.variantId !== undefined) variantId.set(partial.variantId)
  if (partial.systemId !== undefined) systemId.set(partial.systemId)
  if (partial.modelId !== undefined) modelId.set(partial.modelId)
  // … quant / workload / parallelism / disagg block unchanged …

  // Heterogeneous fields.
  if (partial.heterogeneous !== undefined) heterogeneous.set(partial.heterogeneous)
  if (partial.decodeAcceleratorId !== undefined) decodeAcceleratorId.set(partial.decodeAcceleratorId)
  if (partial.decodeVariantId !== undefined) decodeVariantId.set(partial.decodeVariantId)
  if (partial.decodeSystemId !== undefined) decodeSystemId.set(partial.decodeSystemId)
  if (partial.decodeParallelismOverride !== undefined) decodeParallelismOverride.set(partial.decodeParallelismOverride)
}
```

(The existing body should remain; add the new writes at the bottom. Leave the existing quant/workload/parallelism/disagg writes alone.)

- [ ] **Step 7: Update `startUrlSync` to subscribe to the new stores**

In `calc/src/ui/share.ts`, find the `startUrlSync` function and its `unsubs` array (around lines 263-281). Append the new store subscriptions:

```ts
  const unsubs = [
    acceleratorId.subscribe(write),
    variantId.subscribe(write),
    systemId.subscribe(write),
    modelId.subscribe(write),
    parallelismOverride.subscribe(write),
    disaggKvTransferFabricId.subscribe(write),
    disaggFirstTokenOnPrefill.subscribe(write),
    quant.subscribe(write),
    workload.subscribe(write),
    heterogeneous.subscribe(write),
    decodeAcceleratorId.subscribe(write),
    decodeVariantId.subscribe(write),
    decodeSystemId.subscribe(write),
    decodeParallelismOverride.subscribe(write),
  ]
```

- [ ] **Step 8: Run tests + typecheck**

```bash
npm test -- share-route 2>&1 | tail -15
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
```
Expected: all 5 new tests green; full suite green; 0 type errors.

- [ ] **Step 9: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/share.ts calc/test/ui/share-route.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): heterogeneous P/D URL state (het, a2, v2, s2, p2)"
```

---

### Task 7: `DisaggInputPanel` — heterogeneous toggle + decode-side selectors

**Why:** User-facing UI for the v2 feature. Adds a checkbox to enable heterogeneous mode and a decode-side hardware combo dropdown + parallelism picker that appear when on. Pre-populated from the prefill side when the toggle is first flipped.

**Files:**
- Modify: `/Users/yao/workspace/llm-perf/calc/src/ui/DisaggInputPanel.svelte`

No new tests (presentational; the store wiring is tested in Task 5).

- [ ] **Step 1: Rewrite the component**

Replace the entire contents of `calc/src/ui/DisaggInputPanel.svelte` with:

```svelte
<script lang="ts">
  import {
    acceleratorId, variantId, systemId,
    disaggKvTransferFabricId, disaggFirstTokenOnPrefill,
    heterogeneous, decodeAcceleratorId, decodeVariantId, decodeSystemId,
  } from './stores'
  import { groupedDisaggFabrics, formatFabricLabel } from './disaggFabrics'
  import { ACCELERATORS } from '../data'
  import { SYSTEMS } from '../data/systems'
  import { orderSkus } from './catalogOrder'
  import ParallelismPicker from './ParallelismPicker.svelte'

  // V2: when $heterogeneous is on, the decode-side combo dropdown lets the
  // user pick a different accelerator/variant/system for the decode cluster.
  // Decode-side parallelism is shown when a multi-device decode system is
  // selected (via the existing ParallelismPicker, but bound to decode stores
  // — toggle prop below).
  $: groups = groupedDisaggFabrics($acceleratorId)
  $: skuGroups = orderSkus(ACCELERATORS, SYSTEMS)

  // Combo value for the decode-side picker (chip: | sys:).
  $: decodeComboValue = $decodeSystemId
    ? `sys:${$decodeSystemId}`
    : `chip:${$decodeAcceleratorId || $acceleratorId}`   // fallback to prefill\'s accelerator when not yet picked

  $: decodeAcceleratorObj = ACCELERATORS.find(a => a.id === ($decodeAcceleratorId || $acceleratorId))
  $: decodeVariants = decodeAcceleratorObj?.variants ?? []

  function onDecodeComboChange(e: Event) {
    const v = (e.target as HTMLSelectElement).value
    if (v.startsWith('sys:')) {
      decodeSystemId.set(v.slice(4))
    } else {
      decodeSystemId.set('')
      decodeAcceleratorId.set(v.slice(5))
    }
  }

  // Pre-populate decode-side stores from prefill on first toggle-on, so the
  // user transitions from symmetric to asymmetric by changing one knob.
  function onHetToggle(e: Event) {
    const on = (e.target as HTMLInputElement).checked
    heterogeneous.set(on)
    if (on && !$decodeAcceleratorId && !$decodeSystemId) {
      decodeAcceleratorId.set($acceleratorId)
      decodeVariantId.set($variantId)
      decodeSystemId.set($systemId)
    }
  }
</script>

<div class="disagg-inputs">
  <label>
    KV transfer fabric
    <select bind:value={$disaggKvTransferFabricId}>
      <option value="">— off (monolithic only) —</option>
      {#if groups.scaleUp.length > 0}
        <optgroup label="Intra-domain (scale-up)">
          {#each groups.scaleUp as f}
            <option value={f.id}>{formatFabricLabel(f)}</option>
          {/each}
        </optgroup>
      {/if}
      <optgroup label="Cross-rack (scale-out)">
        {#each groups.scaleOut as f}
          <option value={f.id}>{formatFabricLabel(f)}</option>
        {/each}
      </optgroup>
    </select>
  </label>
  {#if $disaggKvTransferFabricId}
    <label class="inline">
      <input type="checkbox" bind:checked={$disaggFirstTokenOnPrefill} />
      <span>1st token on prefill (hide transfer in TTFT)</span>
    </label>
    <label class="inline">
      <input type="checkbox" checked={$heterogeneous} on:change={onHetToggle} />
      <span>Use different hardware for decode cluster</span>
    </label>
  {/if}
</div>

{#if $heterogeneous && $disaggKvTransferFabricId}
  <div class="decode-cluster">
    <div class="section-label">Decode cluster</div>
    <div class="row">
      <label>
        Accelerator
        <select value={decodeComboValue} on:change={onDecodeComboChange}>
          {#each skuGroups as g}
            <optgroup label={g.publisher}>
              {#each g.entries as e}
                {#if e.kind === 'single'}
                  <option value={`chip:${e.id}`}>{e.name}</option>
                {:else}
                  <option value={`sys:${e.id}`}>{e.name} ({e.count}×)</option>
                {/if}
              {/each}
            </optgroup>
          {/each}
        </select>
      </label>
      {#if !$decodeSystemId}
        <label>
          Variant
          <select bind:value={$decodeVariantId}>
            {#each decodeVariants as v}
              <option value={v.id}>{v.label}</option>
            {/each}
          </select>
        </label>
      {/if}
      <ParallelismPicker side="decode" />
    </div>
  </div>
{/if}

<style>
  .disagg-inputs {
    display: flex; flex-direction: row; flex-wrap: wrap;
    gap: 0.75rem; align-items: flex-end;
    padding: 0.6rem 0.9rem;
    background: #fafafa;
    border: 1px solid #e0e0e0; border-radius: 0.3rem;
    margin-bottom: 0.5rem;
  }
  .decode-cluster {
    padding: 0.6rem 0.9rem;
    background: #fafafa;
    border: 1px solid #e0e0e0; border-radius: 0.3rem;
    margin-bottom: 0.75rem;
  }
  .section-label {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: #555; font-weight: 600; margin-bottom: 0.4rem;
  }
  .row { display: flex; flex-direction: row; flex-wrap: wrap; gap: 0.75rem; align-items: flex-end; }
  label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.9rem; }
  label.inline { flex-direction: row; align-items: center; gap: 0.4rem; font-size: 0.85rem; }
  label.inline input[type=checkbox] { width: auto; }
  select { font-size: 1rem; padding: 0.25rem; min-width: 200px; }
</style>
```

- [ ] **Step 2: Add the `side` prop to `ParallelismPicker.svelte`**

The above uses `<ParallelismPicker side="decode" />`. Today\'s `ParallelismPicker` reads from `parallelismOverride` directly. For the decode side, it should read `decodeParallelismOverride` and the `decodeMultiDevice` system.

Read the existing file to see the current shape:

```bash
cat /Users/yao/workspace/llm-perf/calc/src/ui/ParallelismPicker.svelte
```

Add a `side?: 'prefill' | 'decode'` prop (default `'prefill'`) at the top of its script block:

```ts
  export let side: 'prefill' | 'decode' = 'prefill'
```

Then conditionally select which stores to read/write based on `side`. The cleanest way: use derived "active" references at the top of the script. For example, if the current file reads `$systemId` and `$parallelismOverride`, add:

```ts
  import {
    systemId as prefillSystemId, parallelismOverride as prefillParallelism,
    decodeSystemId, decodeParallelismOverride,
    multiDevice as prefillMultiDevice, decodeMultiDevice,
  } from './stores'

  $: activeSystemId = side === 'decode' ? $decodeSystemId : $prefillSystemId
  $: activeMultiDevice = side === 'decode' ? $decodeMultiDevice : $prefillMultiDevice
  $: activeParallelismStore = side === 'decode' ? decodeParallelismOverride : prefillParallelism
```

Then in the body, replace `$systemId` → `activeSystemId`, `$multiDevice` → `activeMultiDevice`, and the parallelism override reads/writes go through `activeParallelismStore`. (The exact edit depends on what the current file looks like — adapt accordingly.)

If `ParallelismPicker` is too tangled to retrofit cleanly with the `side` prop, fall back to extracting the parallelism math into a small helper consumed by both the picker (existing) and a new sub-component `DecodeParallelismPicker.svelte`. Decide based on what you find when reading the file. Either approach is in scope.

- [ ] **Step 3: Run typecheck + full suite**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check 2>&1 | tail -3
npm test 2>&1 | grep -E "Tests " | tail -1
```
Expected: 0 type errors; tests unchanged.

- [ ] **Step 4: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/DisaggInputPanel.svelte calc/src/ui/ParallelismPicker.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): DisaggInputPanel — heterogeneous toggle + decode-side hw + parallelism"
```

---

### Task 8: `Simulator.svelte` — two-sided OOM + op-pair labels

**Why:** The OOM gate switches from monolithic combined check to the new per-side check on the disagg block. The op-name caption in multi-op-point cards renders "peak / achievable" when ids differ.

**Files:**
- Modify: `/Users/yao/workspace/llm-perf/calc/src/ui/Simulator.svelte`

No new tests (presentational; engine + store changes are tested upstream).

- [ ] **Step 1: Update the OOM gate**

In `calc/src/ui/Simulator.svelte`, find the existing memory gate (around lines 31-32):

```ts
  $: memory = $simResultMonolithic?.memory
  $: fits = memory ? (memory.perRank?.fits ?? memory.fits) : false
```

Replace with a structure that knows about the disagg block specifically:

```ts
  // Monolithic block: uses today's combined-total check (= prefill side).
  $: monolithicMemory = $simResultMonolithic?.memory
  $: monolithicFits = monolithicMemory
    ? (monolithicMemory.perRank?.fits ?? monolithicMemory.fits)
    : false

  // Disagg block: two-sided per-cluster check.
  $: disaggMemory = $simResultDisagg?.memory
  $: disaggPrefillFits = disaggMemory
    ? (disaggMemory.prefillSide.perRank?.fits ?? disaggMemory.prefillSide.fits)
    : true
  $: disaggDecodeFits = disaggMemory
    ? (disaggMemory.decodeSide.perRank?.fits ?? disaggMemory.decodeSide.fits)
    : true
  $: disaggFits = disaggPrefillFits && disaggDecodeFits
  $: disaggFailingSides =
    !disaggPrefillFits && !disaggDecodeFits ? 'both' :
    !disaggPrefillFits ? 'prefill' :
    !disaggDecodeFits  ? 'decode'  : null
```

- [ ] **Step 2: Update the OOM rendering**

Find the existing OOM block (around lines 121-128, the `{:else if memory && !fits}` branch). Split it into per-block conditionals. The render order: monolithic block (uses monolithic gate); disagg block (uses disagg gate).

The current monolithic block is rendered conditionally; the disagg block follows. Inside each block's `{#if}` ladder, gate on the side-appropriate fits flag.

Concretely, replace the existing monolithic+disagg branch with this structure:

```svelte
  {#if $simError}
    <div class="error">⚠ {$simError}</div>
  {:else if monolithicMemory && !monolithicFits}
    <div class="oom">
      <strong>✗ Out of memory.</strong>
      Model + KV cache + activations exceed HBM capacity on the selected
      configuration. Pick a larger SKU, add parallelism (TP/PP), or trim the
      workload. See the Calculator tab's Memory panel for the breakdown.
    </div>
  {:else if rowsMonolithic.length > 0}
    <h3 class="config-header">Single request, monolithic</h3>
    {@render resultBlock(rowsMonolithic)}

    {#if $disaggKvTransferFabricId}
      <h3 class="config-header">Single request, PD-disagg</h3>
      <DisaggInputPanel />
      {#if disaggMemory && !disaggFits}
        <div class="oom">
          <strong>✗ Out of memory on {disaggFailingSides} cluster{disaggFailingSides === 'both' ? 's' : ''}.</strong>
          {#if !disaggPrefillFits}
            Prefill side: weights + prefill activations exceed HBM. Try a larger prefill SKU
            or trim promptTokens (prefill activations scale with prompt × hidden).
          {/if}
          {#if !disaggDecodeFits}
            Decode side: weights + KV cache exceed HBM. Try a larger decode SKU,
            add parallelism on the decode cluster, or reduce maxContext-bound KV growth.
          {/if}
        </div>
      {:else if rowsDisagg.length > 0}
        {@render resultBlock(rowsDisagg)}
      {/if}
    {:else if !$disaggKvTransferFabricId}
      <div class="disagg-empty">
        <DisaggInputPanel />
        <p>Pick a KV transfer fabric above to add a PD-disagg comparison block.</p>
      </div>
    {/if}
  {/if}
```

(The existing top-level `{#if $simError}` branch stays. The monolithic OOM gate becomes side-specific; the disagg block has its own internal OOM check that distinguishes the failing side(s).)

- [ ] **Step 3: Update op-pair labels in `resultBlock`**

The pair id from the engine is either `prefillOp.id` (matched) or `"prefillId/decodeId"` (cross-fallback). The resultBlock snippet displays `row.id` as the op-name. The slash-composite format already renders correctly (the string includes the slash), so no template change is needed — the engine\'s pair id naming already produces the right label.

Verify by reading the resultBlock snippet (around lines 80-114). If the existing template renders `{row.id}` directly, no edit is needed. If it transforms the id (e.g. strips slashes), restore raw rendering.

- [ ] **Step 4: Run typecheck + full suite + build**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check 2>&1 | tail -3
npm test 2>&1 | grep -E "Tests " | tail -1
npm run build 2>&1 | tail -5
```
Expected: 0 type errors; tests green; clean build.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/Simulator.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): Simulator — two-sided OOM messaging for disagg block"
```

---

### Task 9: Smoke verification

**Files:** none (verification only).

- [ ] **Step 1: Final test + check + build**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
npm run build 2>&1 | tail -5
```
Expected: all green.

- [ ] **Step 2: Dev-server HTTP smoke**

```bash
cd /Users/yao/workspace/llm-perf/calc
(npm run dev > /tmp/d.log 2>&1 &) ; sleep 4
P=$(grep -oE "localhost:[0-9]+" /tmp/d.log | head -1)
curl -s "http://$P/" -o /dev/null -w "HTTP %{http_code}\n"
pkill -f "node.*vite" 2>/dev/null
```
Expected: HTTP 200.

- [ ] **Step 3: Interactive checks (controller's job)**

- Toggle "Use different hardware for decode cluster" on → decode-side selectors appear, pre-populated with prefill values.
- Change decode accelerator to H200 (or any other) → disagg KPI cards update; ttftS reflects PREFILL hardware\'s decode-step time (not decode cluster\'s).
- Switch to a multi-device decode system → ParallelismPicker (decode side) appears, defaults to system-appropriate parallelism.
- Provoke prefill-side OOM (huge prompt + tiny prefill SKU) → amber notice reads "Out of memory on prefill cluster."
- Provoke decode-side OOM (tiny decode SKU paired with normal prefill) → amber notice reads "Out of memory on decode cluster."
- Provoke both → "Out of memory on both clusters."
- Toggle off → decode-side selectors hidden; URL drops `het`/`a2`/`v2`/`s2`/`p2` keys.
- Share URL with `het=1&a=h100&v=sxm-80&a2=h200&v2=sxm-141` → paste in new tab → restores heterogeneous mode.

No commit step — pure verification. Fix any issues with a focused follow-up commit.

---

## Self-Review

**1. Spec coverage:**
- Spec §Engine refactor (CalcInput decode fields, calculate() refactor, pairOpPoints helper) → Tasks 2, 3, 4. ✓
- Spec §Two-phase memory model (prefillSide/decodeSide/decodeActivationsPeak) → Task 1. ✓
- Spec §Stores (5 new writables + decodeMultiDevice + simInputDisagg) → Task 5. ✓
- Spec §URL state (het, a2, v2, s2, p2) → Task 6. ✓
- Spec §UI DisaggInputPanel (heterogeneous toggle + decode hw + parallelism) → Task 7. ✓
- Spec §UI Simulator.svelte (two-sided OOM + op-pair labels) → Task 8. ✓
- Spec §Testing — engine math: opPoints tests (Task 2), memory two-phase tests (Task 1), calc.ts heterogeneous tests (Task 4). Stores: sim-disagg-stores tests (Task 5). share.ts: round-trip tests (Task 6). In-browser smoke (Task 9). ✓
- Spec §Non-goals — none implemented. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases." Step 2 of Task 7 says "If `ParallelismPicker` is too tangled to retrofit cleanly with the `side` prop, fall back to extracting…" — this is contextual guidance the implementer needs since I haven\'t read the existing ParallelismPicker; both branches are in-scope and the choice is data-driven. Pass.

**3. Type consistency:**
- `CalcInput.decodeAccelerator?: AcceleratorSpec`, `decodeAcceleratorVariantId?: string`, `decodeMultiDevice?: MultiDeviceConfig` (Task 3) — consumed in `calculate()` (Task 4) and `computeMemory()` (Task 1) by name. ✓
- `MemoryResult.prefillSide: MemorySide`, `decodeSide: MemorySide`, `decodeActivationsPeak: number` (Task 1) — consumed in `Simulator.svelte` (Task 8) as `disaggMemory.prefillSide.fits` etc. ✓
- `MemorySide.perRank?.fits` (Task 1) — consumed in `Simulator.svelte` (Task 8) with the `?? memory.fits` fallback. ✓
- `pairOpPoints` returns `Array<{ prefillOp, decodeOp, id }>` (Task 2) — consumed in `calc.ts` (Task 4) as `pair.prefillOp` / `pair.decodeOp` / `pair.id`. ✓
- `computePrefill(input, opPoint, memory, multiDeviceOverride?)` and `computeDecode(...)` (Task 3) — invoked in `calc.ts` (Task 4) with and without the override. ✓
- Stores `decodeAcceleratorId`, `decodeVariantId`, `decodeSystemId`, `decodeParallelismOverride`, `heterogeneous`, `decodeMultiDevice` (Task 5) — consumed in `share.ts` (Task 6) and `DisaggInputPanel.svelte` (Task 7) with exact names. ✓
- `ShareableState` gains `heterogeneous`, `decodeAcceleratorId`, `decodeVariantId`, `decodeSystemId`, `decodeParallelismOverride` (Task 6) — symmetric across encode/decode/readStoreState/applyToStores. ✓

**4. Known follow-ups (parked, not blockers):** Per-side quant, per-side fabric, asymmetric workload semantics, multi-stream/batched disagg — all documented in spec.

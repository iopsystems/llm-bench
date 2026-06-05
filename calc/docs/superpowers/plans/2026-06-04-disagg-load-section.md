# Disagg Load Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Under load (disagg)" section to the Simulator tab that sweeps decode-batch size N from 1 to N_max, charts throughput-vs-N and latency-vs-N, and exposes a slider for the operating point. Calc tab's concurrency field defaults to N_max.

**Architecture:** A pure-engine `queueModel.ts` computes `nMax` (KV-cap ceiling) and `loadCurve` (per-N KPIs) by reusing the existing `computeDecode`/`computePrefill`/`computeMemory` primitives with `workload.concurrency` overridden per-iteration. Stores expose `concurrencyOverride` (writable), `nMaxCalc`/`nMaxDecode` (derived), `effectiveConcurrency` (= override ?? nMaxCalc). The Calc tab's concurrency input binds to these. A new `LoadSection.svelte` mounts in the Sim tab below the existing disagg block, with two SVG charts and a slider that share the `concurrencyOverride` store.

**Tech Stack:** TypeScript, Svelte 5, Vite, Vitest, inline SVG (no charting dep), existing engine primitives in `src/engine/`.

---

## File Structure

**New files:**
- `src/engine/queueModel.ts` — pure functions: `computeNMax`, `loadCurve`, `pdInstanceRatio`. No Svelte/DOM/store deps.
- `src/ui/LoadSection.svelte` — slider + KPI block + LoadCharts. Mounted in Simulator.svelte.
- `src/ui/LoadCharts.svelte` — two side-by-side SVG charts.
- `test/engine/queueModel.test.ts` — engine unit tests.
- `test/ui/sim-load-stores.test.ts` — store derivation tests.

**Modified files:**
- `src/ui/stores.ts` — add `concurrencyOverride`, `nMaxCalc`, `nMaxDecode`, `effectiveConcurrency`; rewire `input` to use effective concurrency.
- `src/ui/InputPanel.svelte` — concurrency input binds to `concurrencyOverride`, displays `nMaxCalc` when override null.
- `src/ui/Simulator.svelte` — mount `<LoadSection />` below the disagg block.
- `src/ui/share.ts` — no semantic change; verify `c=` continues to encode the override.
- `test/ui/share.test.ts` / `test/ui/share-route.test.ts` — add tests for omit/honor semantics.

---

## Task 1: `computeNMax` — KV-cap ceiling for a CalcInput

**Files:**
- Create: `src/engine/queueModel.ts`
- Test: `test/engine/queueModel.test.ts`

- [ ] **Step 1: Write the failing test**

Create `test/engine/queueModel.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { computeNMax } from '../../src/engine/queueModel'
import { ACCELERATORS, MODELS } from '../../src/data'
import type { CalcInput } from '../../src/engine/types'

function inputFor(acceleratorId: string, variantId: string, modelId: string): CalcInput {
  const accelerator = ACCELERATORS.find(a => a.id === acceleratorId)!
  const model = MODELS.find(m => m.id === modelId)!
  return {
    accelerator,
    acceleratorVariantId: variantId,
    model,
    quant: { weights: 'bf16', kv: 'fp16', activations: 'bf16' },
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
  }
}

describe('computeNMax', () => {
  it('returns a positive integer for a model that fits with headroom', () => {
    // H100 SXM-80 (80 GB HBM), Llama-3.3-70B at bf16: weights ≈ 140 GB → doesn't fit
    // single-chip; need multi-device or a bigger chip. Use H200 SXM-141 instead.
    const r = computeNMax(inputFor('h200', 'sxm-141', 'llama-3.3-70b'))
    expect(r.boundBy).toBe('kv')
    expect(r.nMax).toBeGreaterThan(0)
    expect(Number.isInteger(r.nMax)).toBe(true)
  })

  it('returns {nMax: 0, boundBy: weights} when weights alone exceed HBM', () => {
    // Llama-3.3-70B bf16 ≈ 140 GB > 80 GB on H100 SXM-80.
    const r = computeNMax(inputFor('h100', 'sxm-80', 'llama-3.3-70b'))
    expect(r.boundBy).toBe('weights')
    expect(r.nMax).toBe(0)
  })
})
```

- [ ] **Step 2: Run test, verify failure**

Run: `npx vitest run test/engine/queueModel.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `computeNMax`**

Create `src/engine/queueModel.ts`:

```ts
import type { CalcInput } from './types'
import { computeMemory } from './memory'

export interface NMaxResult {
  nMax: number
  boundBy: 'kv' | 'weights'
}

// KV-cap ceiling: how many concurrent in-flight requests can be served before
// HBM exhausts. Decode side (uses decodeMultiDevice/decodeAccelerator when
// present, else falls back to prefill — same as computeMemory). Honest answer
// at the per-rank granularity when multiDevice is configured.
export function computeNMax(input: CalcInput): NMaxResult {
  // Compute memory at concurrency=1 to get per-request sizes (kvCachePerRequest
  // and activations both scale linearly with concurrency). All subsequent
  // arithmetic is in bytes-per-rank when multiDevice is set, bytes-total
  // otherwise — keeps the comparison apples-to-apples.
  const probe = { ...input, workload: { ...input.workload, concurrency: 1 } }
  const memory = computeMemory(probe)
  const side = memory.decodeSide

  // perRank present iff multiDevice is configured on the decode side. Use
  // per-rank when available because that's where HBM actually lives.
  const usingPerRank = side.perRank !== undefined
  const capacityBytes = side.hbmCapacityGB * 1024 * 1024 * 1024
  const weightsBytes = usingPerRank ? side.perRank!.weights : side.weights
  const perReqKvBytes = usingPerRank
    ? side.perRank!.kvCachePerRequest
    : memory.kvCachePerRequest
  // decodeActivationsPeak is at concurrency=1 (we set it above); divide by
  // divisors implicitly captured in perRank.activations when applicable.
  const perReqActBytes = usingPerRank
    ? side.perRank!.activations
    : memory.decodeActivationsPeak

  const free = capacityBytes - weightsBytes
  if (free <= 0) return { nMax: 0, boundBy: 'weights' }

  const perReqBytes = perReqKvBytes + perReqActBytes
  if (perReqBytes <= 0) return { nMax: 0, boundBy: 'weights' }

  const nMax = Math.floor(free / perReqBytes)
  return { nMax: Math.max(0, nMax), boundBy: 'kv' }
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `npx vitest run test/engine/queueModel.test.ts`
Expected: PASS — both tests green.

- [ ] **Step 5: Commit**

```bash
git add src/engine/queueModel.ts test/engine/queueModel.test.ts
git commit -m "feat(calc): queueModel.computeNMax — KV-cap ceiling for concurrency"
```

---

## Task 2: `loadCurve` — per-N KPIs by sweeping concurrency

**Files:**
- Modify: `src/engine/queueModel.ts`
- Test: `test/engine/queueModel.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `test/engine/queueModel.test.ts`:

```ts
import { loadCurve } from '../../src/engine/queueModel'
import { calculate } from '../../src/engine'

describe('loadCurve', () => {
  it('returns one LoadPoint per N with monotonic non-decreasing tpot', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const points = loadCurve(input, [1, 2, 4, 8])
    expect(points).toHaveLength(4)
    expect(points.map(p => p.n)).toEqual([1, 2, 4, 8])
    for (let i = 1; i < points.length; i++) {
      // tpot is non-decreasing because larger batch → more KV reads per step.
      expect(points[i].tpotS).toBeGreaterThanOrEqual(points[i - 1].tpotS)
    }
  })

  it('N=1 LoadPoint matches single-request calculate() for the same input', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const [point] = loadCurve(input, [1])
    const result = calculate({ ...input, workload: { ...input.workload, concurrency: 1 } })
    const tier = Object.values(result.perf)[0]  // first op-point pair
    expect(point.tpotS).toBeCloseTo(tier.decode.timePerTokenS, 12)
    expect(point.prefillS).toBeCloseTo(tier.prefill.timeS, 12)
    // totalS = prefill + kvTransfer + outputTokens × tpot
    const expectedTotal = tier.prefill.timeS + tier.kvTransferS + 512 * tier.decode.timePerTokenS
    expect(point.totalS).toBeCloseTo(expectedTotal, 12)
  })

  it('throughput is bottleneck-bound (min of prefill-rate and decode-rate)', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const [point] = loadCurve(input, [16])
    const decodeRate = 16 / (512 * point.tpotS)
    const prefillRate = 1 / point.prefillS
    const expected = Math.min(decodeRate, prefillRate)
    expect(point.throughputReqS).toBeCloseTo(expected, 12)
    expect(point.throughputTokS).toBeCloseTo(expected * 512, 12)
  })

  it('pdRatio = N × prefillS / (outputTokens × tpot(N))', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const [point] = loadCurve(input, [8])
    const expected = (8 * point.prefillS) / (512 * point.tpotS)
    expect(point.pdRatio).toBeCloseTo(expected, 12)
  })
})
```

- [ ] **Step 2: Run test, verify failure**

Run: `npx vitest run test/engine/queueModel.test.ts`
Expected: FAIL — `loadCurve` not exported.

- [ ] **Step 3: Implement `loadCurve` and `pdInstanceRatio`**

Append to `src/engine/queueModel.ts`:

```ts
import { computePrefill } from './prefill'
import { computeDecode } from './decode'
import { INTERCONNECTS } from '../data/interconnects'
import { pairOpPoints } from './opPoints'

export interface LoadPoint {
  n: number
  tpotS: number
  prefillS: number
  kvTransferS: number
  totalS: number
  throughputTokS: number
  throughputReqS: number
  pdRatio: number
}

// Per-N KPIs computed by reusing the engine's prefill/decode primitives with
// workload.concurrency overridden. Caller passes the disagg-side input
// (concurrency clamped to 1); loadCurve re-introduces N per iteration.
//
// Closed-loop, deterministic, identical-request model: no queue dynamics, no
// percentiles to compute — the math is direct.
export function loadCurve(input: CalcInput, ns: number[]): LoadPoint[] {
  // Resolve op-points the same way calc.ts does, so the chosen perf tier
  // matches what the single-request blocks above show.
  const prefillVariant = input.accelerator.variants.find(v => v.id === input.acceleratorVariantId)
  if (!prefillVariant) return []
  const decodeAccelerator = input.decodeAccelerator ?? input.accelerator
  const decodeVariantId = input.decodeAcceleratorVariantId ?? input.acceleratorVariantId
  const decodeVariant = decodeAccelerator.variants.find(v => v.id === decodeVariantId)
  if (!decodeVariant) return []
  const pairs = pairOpPoints(prefillVariant, decodeVariant)
  if (pairs.length === 0) return []
  const pair = pairs[0]  // v1: use the first (canonical) op-point pair

  // KV transfer cost is independent of N (per-request shipment).
  let kvTransferS = 0
  if (input.disaggKvTransferFabricId) {
    const fab = INTERCONNECTS.find(i => i.id === input.disaggKvTransferFabricId)
    if (fab) {
      const bw = fab.perDirectionGBs ?? fab.perGpuBandwidthGBs / 2
      // Need memory at concurrency=1 to get kvCachePerRequest.
      const probeMem = computeMemory({ ...input, workload: { ...input.workload, concurrency: 1 } })
      kvTransferS = probeMem.kvCachePerRequest / (bw * 1e9)
    }
  }

  // prefillS is independent of N (prefill cluster runs one request serially).
  const probeMem = computeMemory({ ...input, workload: { ...input.workload, concurrency: 1 } })
  const prefillTier = computePrefill(input, pair.prefillOp, probeMem)
  const prefillS = prefillTier.timeS

  const outputTokens = input.workload.outputTokens

  return ns.map(n => {
    // Build a per-iteration input with concurrency = N. Memory recomputes
    // because decode-step bytes scale with N (KV cache per step).
    const inputN = { ...input, workload: { ...input.workload, concurrency: n } }
    const memN = computeMemory(inputN)
    const decode = computeDecode(inputN, pair.decodeOp, memN)
    const tpotS = decode.timePerTokenS

    const totalS = prefillS + kvTransferS + outputTokens * tpotS
    const decodeReqRate = n / (outputTokens * tpotS)
    const prefillReqRate = 1 / prefillS
    const throughputReqS = Math.min(decodeReqRate, prefillReqRate)
    const throughputTokS = throughputReqS * outputTokens
    const pdRatio = (n * prefillS) / (outputTokens * tpotS)

    return { n, tpotS, prefillS, kvTransferS, totalS, throughputTokS, throughputReqS, pdRatio }
  })
}

export function pdInstanceRatio(prefillS: number, outputTokens: number, tpotS: number, n: number): number {
  return (n * prefillS) / (outputTokens * tpotS)
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `npx vitest run test/engine/queueModel.test.ts`
Expected: PASS — all six tests green.

- [ ] **Step 5: Commit**

```bash
git add src/engine/queueModel.ts test/engine/queueModel.test.ts
git commit -m "feat(calc): queueModel.loadCurve — per-N KPIs for disagg under load"
```

---

## Task 3: Stores — `concurrencyOverride`, `nMaxCalc`, `nMaxDecode`, `effectiveConcurrency`

**Files:**
- Modify: `src/ui/stores.ts`
- Test: `test/ui/sim-load-stores.test.ts`

- [ ] **Step 1: Write the failing test**

Create `test/ui/sim-load-stores.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  acceleratorId, variantId, systemId, modelId, workload, quant,
  concurrencyOverride, nMaxCalc, nMaxDecode, effectiveConcurrency,
  heterogeneous, decodeAcceleratorId, decodeVariantId, decodeSystemId,
  prefillAcceleratorId, prefillVariantId, prefillSystemId,
} from '../../src/ui/stores'

function resetStores() {
  acceleratorId.set('h200')
  variantId.set('sxm-141')
  systemId.set('')
  modelId.set('llama-3.3-70b')
  quant.set({ weights: 'bf16', kv: 'fp16', activations: 'bf16' })
  workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 1 })
  concurrencyOverride.set(null)
  heterogeneous.set(false)
  prefillAcceleratorId.set('')
  prefillVariantId.set('')
  prefillSystemId.set('')
  decodeAcceleratorId.set('')
  decodeVariantId.set('')
  decodeSystemId.set('')
}

describe('concurrencyOverride + effectiveConcurrency', () => {
  beforeEach(resetStores)

  it('default override is null; effective tracks nMaxCalc', () => {
    expect(get(concurrencyOverride)).toBeNull()
    expect(get(nMaxCalc)).toBeGreaterThan(0)
    expect(get(effectiveConcurrency)).toBe(get(nMaxCalc))
  })

  it('setting override to N makes effective return N', () => {
    concurrencyOverride.set(7)
    expect(get(effectiveConcurrency)).toBe(7)
  })

  it('clearing override (set to null) reverts effective to nMaxCalc', () => {
    concurrencyOverride.set(7)
    concurrencyOverride.set(null)
    expect(get(effectiveConcurrency)).toBe(get(nMaxCalc))
  })

  it('changing hardware re-derives nMaxCalc; effective follows when override is null', () => {
    const before = get(nMaxCalc)
    acceleratorId.set('h100')
    variantId.set('sxm-80')
    // Llama-3.3-70B at bf16 doesn't fit H100-80; nMax should be 0.
    expect(get(nMaxCalc)).toBe(0)
    expect(get(nMaxCalc)).not.toBe(before)
    // Effective floors at 1 even when nMax = 0, so the engine never sees concurrency=0.
    expect(get(effectiveConcurrency)).toBe(1)
  })

  it('override stays sticky when hardware changes', () => {
    concurrencyOverride.set(5)
    acceleratorId.set('h100')
    variantId.set('sxm-80')
    expect(get(effectiveConcurrency)).toBe(5)
  })
})

describe('nMaxDecode vs nMaxCalc under het=on', () => {
  beforeEach(resetStores)

  it('with symmetric hw (het off), nMaxDecode tracks nMaxCalc', () => {
    expect(get(nMaxDecode)).toBe(get(nMaxCalc))
  })

  it('with het=on + smaller decode hw, nMaxDecode < nMaxCalc', () => {
    // Calc/prefill stays on H200; decode cluster moves to H100-80.
    heterogeneous.set(true)
    prefillAcceleratorId.set('h200')
    prefillVariantId.set('sxm-141')
    decodeAcceleratorId.set('h100')
    decodeVariantId.set('sxm-80')
    // H100-80 can't fit 70B weights → nMaxDecode = 0; nMaxCalc still positive.
    expect(get(nMaxDecode)).toBe(0)
    expect(get(nMaxCalc)).toBeGreaterThan(0)
  })
})
```

- [ ] **Step 2: Run test, verify failure**

Run: `npx vitest run test/ui/sim-load-stores.test.ts`
Expected: FAIL — `concurrencyOverride`, `nMaxCalc`, `nMaxDecode`, `effectiveConcurrency` not exported.

- [ ] **Step 3: Add stores + rewire `input` to use effective concurrency**

Edit `src/ui/stores.ts`. Find the `workload` writable around line 65 — leave it as-is. After the `disaggFirstTokenOnPrefill` declaration (around line 30), add:

```ts
// User override for in-flight count. null ⇒ "use computed nMax". The Calc-tab
// concurrency input and the Sim-tab LoadSection slider both bind to this
// store; their displayed defaults differ (nMaxCalc vs nMaxDecode) but the
// override is shared.
export const concurrencyOverride = writable<number | null>(null)
```

Add an import at the top:

```ts
import { computeNMax } from '../engine/queueModel'
```

After `simInputDisagg` (around line 161+ depending on current state), add the derived stores:

```ts
// nMaxCalc: KV-cap ceiling computed against the Calc-tab (shared) input.
// Drives the Calc-tab concurrency default.
export const nMaxCalc: Readable<number> = derived(
  [input],
  ([$input]) => $input ? computeNMax($input).nMax : 0
)

// nMaxDecode: KV-cap ceiling for the disagg decode cluster (heterogeneity
// aware via simInputDisagg). Drives the LoadSection slider default and clamp.
export const nMaxDecode: Readable<number> = derived(
  [simInputDisagg],
  ([$d]) => $d ? computeNMax($d).nMax : 0
)

// Effective concurrency for Calc-tab consumers. Floor at 1 so the engine
// never sees concurrency=0 (which would zero out tokens-per-step math).
export const effectiveConcurrency: Readable<number> = derived(
  [concurrencyOverride, nMaxCalc],
  ([$override, $nMax]) => $override ?? Math.max(1, $nMax)
)
```

Now rewire `input` to use `effectiveConcurrency`. Find the `input` derived (around line 101) — its current signature is `derived([acceleratorId, variantId, ..., workload, ...], ...)`. Change `workload` consumption inside the deriver so the returned `workload` field uses `effectiveConcurrency` for the concurrency value:

```ts
export const input: Readable<CalcInput | null> = derived(
  [acceleratorId, variantId, modelId, quant, workload, multiDevice, systemId,
   disaggKvTransferFabricId, disaggFirstTokenOnPrefill, effectiveConcurrency],
  ([$acceleratorId, $variantId, $modelId, $quant, $workload, $multiDevice, $systemId,
    $disagg, $firstTokenOnPrefill, $effectiveConcurrency]) => {
    let accelerator
    let resolvedVariantId: string
    if ($systemId && $multiDevice) {
      accelerator = ACCELERATORS.find(a => a.id === $multiDevice.system.accelerator.id)
      resolvedVariantId = $multiDevice.system.accelerator.variantId
    } else {
      accelerator = ACCELERATORS.find(a => a.id === $acceleratorId)
      resolvedVariantId = $variantId
    }
    const model = MODELS.find(m => m.id === $modelId)
    if (!accelerator || !model) return null
    if (!accelerator.variants.find(v => v.id === resolvedVariantId)) return null
    return {
      accelerator,
      acceleratorVariantId: resolvedVariantId,
      model,
      quant: $quant,
      workload: { ...$workload, concurrency: $effectiveConcurrency },
      ...($multiDevice && { multiDevice: $multiDevice }),
      ...($disagg && {
        disaggKvTransferFabricId: $disagg,
        disaggFirstTokenOnPrefill: $firstTokenOnPrefill,
      }),
    }
  }
)
```

Note: `nMaxCalc` depends on `input` and `effectiveConcurrency` depends on `nMaxCalc`, but `input` now depends on `effectiveConcurrency`. This is a circular derivation. Resolve it by computing `nMaxCalc` against a probe input that uses `workload` directly (not effective concurrency). The simplest path:

Change `nMaxCalc` to compute against a workload-clamped input independent of effective concurrency:

```ts
export const nMaxCalc: Readable<number> = derived(
  [acceleratorId, variantId, modelId, quant, workload, multiDevice, systemId],
  ([$acceleratorId, $variantId, $modelId, $quant, $workload, $multiDevice, $systemId]) => {
    let accelerator
    let resolvedVariantId: string
    if ($systemId && $multiDevice) {
      accelerator = ACCELERATORS.find(a => a.id === $multiDevice.system.accelerator.id)
      resolvedVariantId = $multiDevice.system.accelerator.variantId
    } else {
      accelerator = ACCELERATORS.find(a => a.id === $acceleratorId)
      resolvedVariantId = $variantId
    }
    const model = MODELS.find(m => m.id === $modelId)
    if (!accelerator || !model) return 0
    if (!accelerator.variants.find(v => v.id === resolvedVariantId)) return 0
    const probe: CalcInput = {
      accelerator, acceleratorVariantId: resolvedVariantId, model,
      quant: $quant, workload: { ...$workload, concurrency: 1 },
      ...($multiDevice && { multiDevice: $multiDevice }),
    }
    return computeNMax(probe).nMax
  }
)
```

`nMaxDecode` already uses `simInputDisagg`, which clamps concurrency to 1 — no circular dep.

- [ ] **Step 4: Run tests, verify pass**

Run: `npx vitest run test/ui/sim-load-stores.test.ts`
Expected: PASS — all 7 tests green.

Also run full test suite to catch breakage in other tests that depend on `workload.concurrency` flowing through unchanged:

Run: `npm test`
Expected: 311 → still all green. If some tests fail because they relied on `concurrency=1` literally, that's a regression we need to fix — most likely `share.test.ts` round-trips that set `c=1` should still round-trip (override is the value, effective is what matters).

- [ ] **Step 5: Commit**

```bash
git add src/ui/stores.ts test/ui/sim-load-stores.test.ts
git commit -m "feat(calc): concurrencyOverride + nMax stores; input uses effective concurrency"
```

---

## Task 4: Calc-tab concurrency input → bind to `concurrencyOverride`

**Files:**
- Modify: `src/ui/InputPanel.svelte`

- [ ] **Step 1: Inspect the current concurrency input**

Read `src/ui/InputPanel.svelte` around lines 67-97 (the concurrency state + handler) and lines 231-244 (the input markup). The handler today writes to `workload.concurrency`.

- [ ] **Step 2: Replace the concurrency handler + state to use `concurrencyOverride`**

Edit `src/ui/InputPanel.svelte`. Update the imports at line 4:

```ts
import { acceleratorId, variantId, systemId, modelId, quant, workload, disaggKvTransferFabricId, disaggFirstTokenOnPrefill, concurrencyOverride, nMaxCalc } from './stores'
```

Replace the concurrency local state (around line 67) and handler (around line 90-97):

```ts
// Concurrency: override-based. When override is null, the input shows nMaxCalc
// as a placeholder-ish value (the "auto" default). User types → override sticks.
// Clearing the field → override resets to null → display reverts to nMaxCalc.
let concurrencyInput = $concurrencyOverride === null ? '' : String($concurrencyOverride)
let concurrencyInvalid = false

// Keep the textbox in sync with the store when the store changes externally
// (e.g. URL load, LoadSection slider drag).
$: concurrencyInput = $concurrencyOverride === null ? '' : String($concurrencyOverride)

function onConcurrencyInput(e: Event) {
  const v = (e.target as HTMLInputElement).value
  concurrencyInput = v
  if (v.trim() === '') {
    concurrencyInvalid = false
    concurrencyOverride.set(null)
    return
  }
  const n = parseTokenCount(v)
  if (n === null || n <= 0) { concurrencyInvalid = true; return }
  concurrencyInvalid = false
  concurrencyOverride.set(n)
}
```

Update the markup around line 235 — the input should show `nMaxCalc` as a placeholder when override is null:

```svelte
{#if !hideConcurrency}
  <label>
    Concurrency
    <input
      type="text"
      value={concurrencyInput}
      placeholder={`auto (${$nMaxCalc})`}
      class:invalid={concurrencyInvalid}
      on:input={onConcurrencyInput}
    />
  </label>
{/if}
```

Find any other reads of `$workload.concurrency` in InputPanel and replace with `$effectiveConcurrency` where they're displaying the "live" value (probably none in InputPanel itself, but search to confirm).

- [ ] **Step 3: Verify no other writers / displays of concurrency in InputPanel**

Run: `grep -n "concurrency" src/ui/InputPanel.svelte`
Expected: only the import (line 4), the `concurrencyInput` local + `onConcurrencyInput` handler we just rewrote, the `hideConcurrency` prop, and the markup block. No stray reads of `$workload.concurrency` for display.

Run: `grep -rn "workload\\.update.*concurrency\\|workload\\.set.*concurrency" src/ui/`
Expected: no matches (no other component writes the workload's concurrency directly — they all go through the override).

- [ ] **Step 4: Run full test suite + type check + build**

Run: `npm run check 2>&1 | tail -3 && npm test 2>&1 | grep "Tests " | tail -1 && npm run build 2>&1 | tail -3`
Expected: 0 errors / 311+ tests pass / build succeeds. Some existing tests may fail because the URL `c=` decoding now writes to `concurrencyOverride` instead of `workload.concurrency` — fix in Task 5.

- [ ] **Step 5: Commit**

```bash
git add src/ui/InputPanel.svelte
git commit -m "feat(calc): concurrency input binds to override store; placeholder shows nMaxCalc"
```

---

## Task 5: URL state — `c=` encodes override, not raw workload.concurrency

**Files:**
- Modify: `src/ui/share.ts`
- Modify: `test/ui/share.test.ts` and `test/ui/share-route.test.ts`

- [ ] **Step 1: Inspect current share.ts handling of `c=`**

Read the current encode/decode paths for the `c` URL key. Today encoding emits `c=${workload.concurrency}` and decoding writes to `workload.concurrency`.

- [ ] **Step 2: Write failing tests for the new semantic**

Add to `test/ui/share.test.ts`:

```ts
describe('concurrencyOverride URL encoding', () => {
  it('omits c= when override is null', () => {
    const state: ShareableState = {
      ...singleChipState,
      concurrencyOverride: null,
    }
    expect(encodeState(state)).not.toMatch(/(^|&)c=/)
  })

  it('emits c=N when override is set', () => {
    const state: ShareableState = {
      ...singleChipState,
      concurrencyOverride: 7,
    }
    expect(encodeState(state)).toContain('c=7')
  })

  it('decodes c=5 to concurrencyOverride=5', () => {
    expect(decodeState('c=5').concurrencyOverride).toBe(5)
  })

  it('decodes missing c= to concurrencyOverride undefined (recipient default null)', () => {
    expect(decodeState('a=h100&v=sxm-80').concurrencyOverride).toBeUndefined()
  })

  it('backward compat: old URL with c=1 sets override to 1', () => {
    expect(decodeState('c=1').concurrencyOverride).toBe(1)
  })
})
```

Existing tests reference `workload.concurrency` from `singleChipState` / `multiDeviceState`. Update these fixtures to include `concurrencyOverride: null` (so the tests stay valid TypeScript) and remove the `concurrency` field from the `workload` they pass in OR keep it (workload still carries it; the override is a separate axis). Inspect the existing tests around lines 100-180 of share.test.ts and update fixture shapes — add `concurrencyOverride: null` to both state literals.

- [ ] **Step 3: Run tests, verify failures**

Run: `npx vitest run test/ui/share.test.ts`
Expected: FAIL — `concurrencyOverride` not on ShareableState; encodeState doesn't emit/parse it correctly.

- [ ] **Step 4: Update share.ts**

Edit `src/ui/share.ts`:

1. Add `concurrencyOverride: number | null` to `ShareableState`.
2. Import `concurrencyOverride` from stores; remove direct write to `workload.concurrency` in `applyToStores` for the `c=` key.
3. In `readStoreState`, read `concurrencyOverride`.
4. In `applyToStores`, set `concurrencyOverride` from the decoded state.
5. In `encodeState`, change the `c=` emission:

```ts
if (state.concurrencyOverride !== null) {
  p.set('c', String(state.concurrencyOverride))
}
```

Replace the old emission that pulls from `state.workload.concurrency`. Also drop the `c` parse path from the `workload` block in `decodeState` — `c` is now its own top-level key:

```ts
// Standalone parse for c (concurrencyOverride).
const c = params.get('c')
if (c !== null) {
  const n = parseInt(c, 10)
  if (Number.isFinite(n) && n > 0) out.concurrencyOverride = n
}
```

Remove `c` from the workload-extraction block (workload no longer encodes concurrency in the URL — only pt/ot do).

5. In `startUrlSync`, subscribe to `concurrencyOverride` too so the URL updates when the slider/textbox changes.

- [ ] **Step 5: Run tests, verify pass**

Run: `npx vitest run test/ui/share.test.ts test/ui/share-route.test.ts`
Expected: PASS — all share tests green.

Run full suite: `npm test`
Expected: 311+ pass.

- [ ] **Step 6: Commit**

```bash
git add src/ui/share.ts test/ui/share.test.ts test/ui/share-route.test.ts
git commit -m "feat(calc): URL c= encodes concurrencyOverride (omit when null)"
```

---

## Task 6: `LoadCharts.svelte` — two side-by-side SVG charts

**Files:**
- Create: `src/ui/LoadCharts.svelte`

- [ ] **Step 1: Create the chart component**

Create `src/ui/LoadCharts.svelte`:

```svelte
<script lang="ts">
  import type { LoadPoint } from '../engine/queueModel'

  export let points: LoadPoint[]
  export let selectedN: number
  export let nMax: number

  // Chart dimensions: each panel 280×140 px, with 32-px left margin for y-axis
  // labels and 24-px bottom margin for x-axis labels. Inline SVG, no charting
  // dep — keeps bundle small and matches the gantt's style.
  const W = 280
  const H = 140
  const ML = 36
  const MB = 24
  const MT = 8
  const MR = 8
  const PW = W - ML - MR
  const PH = H - MT - MB

  // Derive both y-axis maxima from the data; round up to a "nice" number so
  // tick labels are readable.
  $: throughputMax = niceMax(points.map(p => p.throughputTokS))
  $: latencyMax    = niceMax(points.map(p => p.totalS))

  function niceMax(values: number[]): number {
    const max = Math.max(...values, 0)
    if (max === 0) return 1
    const exp = Math.floor(Math.log10(max))
    const base = Math.pow(10, exp)
    const norm = max / base
    const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10
    return nice * base
  }

  function xPx(n: number): number { return ML + (n - 1) / Math.max(1, nMax - 1) * PW }
  function yPxThru(v: number): number { return MT + PH - (v / throughputMax) * PH }
  function yPxLat(v: number):  number { return MT + PH - (v / latencyMax)    * PH }

  $: thruPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${xPx(p.n).toFixed(2)},${yPxThru(p.throughputTokS).toFixed(2)}`
  ).join(' ')
  $: latPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${xPx(p.n).toFixed(2)},${yPxLat(p.totalS).toFixed(2)}`
  ).join(' ')

  $: selectedPoint = points.find(p => p.n === selectedN) ?? points[points.length - 1]

  function fmtThru(v: number): string {
    if (v >= 1e6) return `${(v / 1e6).toPrecision(3)}M`
    if (v >= 1e3) return `${(v / 1e3).toPrecision(3)}k`
    return v.toPrecision(3)
  }
  function fmtLat(v: number): string {
    if (v >= 1) return `${v.toPrecision(3)}s`
    return `${(v * 1000).toPrecision(3)}ms`
  }
</script>

<div class="charts">
  <div class="chart">
    <div class="title">Throughput (tok/s)</div>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
      <!-- y-axis -->
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#bbb" stroke-width="1" />
      <text x={ML - 4} y={MT + 4} text-anchor="end" font-size="9" fill="#666">{fmtThru(throughputMax)}</text>
      <text x={ML - 4} y={MT + PH} text-anchor="end" font-size="9" fill="#666">0</text>
      <!-- x-axis -->
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#bbb" stroke-width="1" />
      <text x={ML} y={H - 6} text-anchor="start" font-size="9" fill="#666">N=1</text>
      <text x={ML + PW} y={H - 6} text-anchor="end" font-size="9" fill="#666">N={nMax}</text>
      <!-- curve -->
      <path d={thruPath} fill="none" stroke="#2b6cb0" stroke-width="1.5" />
      <!-- selected marker -->
      {#if selectedPoint}
        <line x1={xPx(selectedPoint.n)} y1={MT} x2={xPx(selectedPoint.n)} y2={MT + PH}
              stroke="#fcd34d" stroke-width="2.5" />
        <circle cx={xPx(selectedPoint.n)} cy={yPxThru(selectedPoint.throughputTokS)}
                r="3.5" fill="#2b6cb0" />
      {/if}
    </svg>
  </div>

  <div class="chart">
    <div class="title">Per-request latency</div>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#bbb" stroke-width="1" />
      <text x={ML - 4} y={MT + 4} text-anchor="end" font-size="9" fill="#666">{fmtLat(latencyMax)}</text>
      <text x={ML - 4} y={MT + PH} text-anchor="end" font-size="9" fill="#666">0</text>
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#bbb" stroke-width="1" />
      <text x={ML} y={H - 6} text-anchor="start" font-size="9" fill="#666">N=1</text>
      <text x={ML + PW} y={H - 6} text-anchor="end" font-size="9" fill="#666">N={nMax}</text>
      <path d={latPath} fill="none" stroke="#c05621" stroke-width="1.5" />
      {#if selectedPoint}
        <line x1={xPx(selectedPoint.n)} y1={MT} x2={xPx(selectedPoint.n)} y2={MT + PH}
              stroke="#fcd34d" stroke-width="2.5" />
        <circle cx={xPx(selectedPoint.n)} cy={yPxLat(selectedPoint.totalS)}
                r="3.5" fill="#c05621" />
      {/if}
    </svg>
  </div>
</div>

<style>
  .charts {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;
  }
  .chart {
    padding: 0.6rem 0.9rem;
    background: #fff; border: 1px solid #d4d4d4; border-radius: 0.4rem;
  }
  .title {
    font-size: 0.8rem; font-weight: 600; color: #555; margin-bottom: 0.3rem;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  @media (max-width: 800px) {
    .charts { grid-template-columns: 1fr; }
  }
</style>
```

- [ ] **Step 2: Verify type check**

Run: `npm run check 2>&1 | tail -3`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```bash
git add src/ui/LoadCharts.svelte
git commit -m "feat(calc): LoadCharts — two side-by-side SVG charts for under-load view"
```

---

## Task 7: `LoadSection.svelte` — slider + KPI block + charts wrapper

**Files:**
- Create: `src/ui/LoadSection.svelte`

- [ ] **Step 1: Create the section component**

Create `src/ui/LoadSection.svelte`:

```svelte
<script lang="ts">
  import { simInputDisagg, concurrencyOverride, nMaxDecode } from './stores'
  import { loadCurve } from '../engine/queueModel'
  import LoadCharts from './LoadCharts.svelte'

  // Sweep range: 1 to nMaxDecode, capped at 256 sample points so very large
  // nMax doesn't blow the chart with thousands of <path> nodes. Use a uniform
  // stride for now — log stride is overkill at this scale.
  $: nMax = $nMaxDecode
  $: ns = (() => {
    if (nMax <= 0) return []
    const cap = 256
    if (nMax <= cap) return Array.from({ length: nMax }, (_, i) => i + 1)
    const stride = Math.ceil(nMax / cap)
    const out: number[] = []
    for (let n = 1; n <= nMax; n += stride) out.push(n)
    if (out[out.length - 1] !== nMax) out.push(nMax)
    return out
  })()

  $: points = ($simInputDisagg && ns.length > 0) ? loadCurve($simInputDisagg, ns) : []

  // Selected N: user's override (if set), else nMaxDecode (= run at the cap).
  // Clamp to nMaxDecode for display — the user's override might be larger
  // (legitimately, for the Calc-tab context); we don't mutate the store.
  $: rawSelected = $concurrencyOverride ?? nMax
  $: selectedN = nMax > 0 ? Math.max(1, Math.min(nMax, rawSelected)) : 1
  $: clamped = ($concurrencyOverride !== null) && ($concurrencyOverride > nMax)

  $: selectedPoint = points.find(p => p.n === selectedN)
    ?? (points.length > 0 ? points.reduce((acc, p) => (Math.abs(p.n - selectedN) < Math.abs(acc.n - selectedN) ? p : acc)) : null)

  function onSliderInput(e: Event) {
    const v = parseInt((e.target as HTMLInputElement).value, 10)
    if (Number.isFinite(v) && v >= 1) {
      concurrencyOverride.set(v)
    }
  }

  function fmt(v: number, unit: string): string {
    if (unit === 's' && v < 1) return `${(v * 1000).toPrecision(3)} ms`
    if (unit === 'tok/s' && v >= 1e3) return `${(v / 1e3).toPrecision(3)} k tok/s`
    return `${v.toPrecision(3)} ${unit}`
  }
</script>

{#if nMax > 0 && points.length > 0 && selectedPoint}
  <div class="load-section">
    <h3 class="section-header">Under load</h3>

    <div class="slider-row">
      <label class="slider-label">
        <span>N (in-flight decode batch)</span>
        <input
          type="range"
          min="1" max={nMax} step="1"
          value={selectedN}
          on:input={onSliderInput}
        />
      </label>
      <div class="readout">
        <strong>{selectedN}</strong> / {nMax}
        {#if clamped}
          <span class="clamped">(override {$concurrencyOverride} clamped to decode-cluster cap)</span>
        {/if}
      </div>
    </div>

    <div class="kpi-row">
      <div class="kpi">
        <div class="label">Aggregate throughput</div>
        <div class="value">{fmt(selectedPoint.throughputTokS, 'tok/s')}</div>
        <div class="caption">{selectedPoint.throughputReqS.toPrecision(3)} req/s</div>
      </div>
      <div class="kpi">
        <div class="label">Per-request total</div>
        <div class="value">{fmt(selectedPoint.totalS, 's')}</div>
        <div class="caption">TPOT {fmt(selectedPoint.tpotS, 's')}</div>
      </div>
      <div class="kpi pd">
        <div class="label">P:D instance ratio</div>
        <div class="value">{selectedPoint.pdRatio.toPrecision(3)}</div>
        <div class="caption">
          {#if selectedPoint.pdRatio > 1}
            prefill-bound: need {selectedPoint.pdRatio.toPrecision(3)} prefill nodes per decode node
          {:else}
            decode-bound: {selectedPoint.pdRatio.toPrecision(3)} prefill nodes per decode node sustain the batch
          {/if}
        </div>
      </div>
    </div>

    <LoadCharts {points} {selectedN} {nMax} />
  </div>
{:else if nMax === 0}
  <div class="load-section">
    <h3 class="section-header">Under load</h3>
    <div class="oom-hint">
      Decode cluster can't fit any in-flight requests at this configuration
      (weights alone exceed HBM, or per-request KV overhead does after weights).
      Pick a larger decode SKU or add parallelism on the decode cluster.
    </div>
  </div>
{/if}

<style>
  .load-section { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.75rem; }
  .section-header { margin: 0; font-size: 1rem; font-weight: 600; color: #333; }
  .slider-row { display: flex; flex-direction: row; align-items: center; gap: 1rem; }
  .slider-label { display: flex; flex-direction: column; gap: 0.3rem; flex: 1; font-size: 0.85rem; color: #555; }
  .slider-label input[type=range] { width: 100%; }
  .readout { font-size: 0.95rem; color: #333; min-width: 8rem; }
  .readout strong { font-size: 1.2rem; }
  .clamped { display: block; font-size: 0.75rem; color: #8a3f00; font-style: italic; }
  .kpi-row { display: grid; grid-template-columns: 1fr 1fr 1.5fr; gap: 0.75rem; }
  .kpi {
    padding: 0.6rem 0.9rem; background: #fff;
    border: 1px solid #d4d4d4; border-radius: 0.4rem;
  }
  .kpi .label {
    font-size: 0.8rem; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .kpi .value { font-size: 1.4rem; font-weight: 700; color: #222; margin-top: 0.2rem; }
  .kpi .caption { font-size: 0.78rem; color: #666; margin-top: 0.3rem; }
  .oom-hint {
    padding: 0.7rem 0.9rem;
    background: #fff7ec; color: #8a3f00;
    border: 1px solid #f0c890; border-radius: 0.3rem;
    font-size: 0.9rem; line-height: 1.4;
  }
  @media (max-width: 700px) {
    .kpi-row { grid-template-columns: 1fr; }
  }
</style>
```

- [ ] **Step 2: Verify type check**

Run: `npm run check 2>&1 | tail -3`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```bash
git add src/ui/LoadSection.svelte
git commit -m "feat(calc): LoadSection — slider + KPI block + LoadCharts for disagg under load"
```

---

## Task 8: Mount `LoadSection` in `Simulator.svelte`

**Files:**
- Modify: `src/ui/Simulator.svelte`

- [ ] **Step 1: Add import + mount in the disagg block**

Edit `src/ui/Simulator.svelte`. Add the import at the top:

```ts
import LoadSection from './LoadSection.svelte'
```

Find the existing disagg block render — specifically the `{:else if rowsDisagg.length > 0}` branch that renders `{@render resultBlock(rowsDisagg)}`. Add `<LoadSection />` immediately after that render call, still inside the `{#if $disaggKvTransferFabricId}` branch:

```svelte
{:else if rowsDisagg.length > 0}
  {@render resultBlock(rowsDisagg)}
  <LoadSection />
{/if}
```

- [ ] **Step 2: Verify the gating is right**

The new section should appear:
- Only when a disagg KV-transfer fabric is selected (already-gated by enclosing `{#if $disaggKvTransferFabricId}`)
- Only when the disagg single-request block is also rendering (not when OOM or error blocks are showing — by sharing the same `{:else if rowsDisagg.length > 0}` branch)
- Only when `nMaxDecode > 0` (gated internally by LoadSection)

- [ ] **Step 3: Run check + build + tests**

Run: `npm run check 2>&1 | tail -3 && npm test 2>&1 | grep "Tests " | tail -1 && npm run build 2>&1 | tail -3`
Expected: 0 errors / all tests pass / build succeeds.

- [ ] **Step 4: Manual smoke test in dev server**

Run: `npm run dev` (background)
Open http://localhost:5173/#sim
Verify in the browser:
1. Pick H200 SXM-141 + Llama-3.3-70B at bf16. No KV fabric → no LoadSection visible.
2. Pick IB-NDR fabric → LoadSection appears below the single-request disagg block.
3. Slider min=1, max=nMaxDecode, default at nMaxDecode. KPI block populated. Two charts render.
4. Drag slider → KPIs and chart markers update live.
5. Switch to Calc tab → concurrency field shows the slider's value (or placeholder `auto (N)` when override null).
6. Clear concurrency field on Calc → both contexts revert to their respective nMax defaults.
7. Enable heterogeneous in Sim and pick a smaller decode SKU → nMaxDecode drops; if override > new nMaxDecode, slider visually clamps + "override … clamped to decode-cluster cap" hint appears.

If anything doesn't work, fix and retest. Stop the dev server when done.

- [ ] **Step 5: Commit**

```bash
git add src/ui/Simulator.svelte
git commit -m "feat(calc): mount LoadSection in Simulator below disagg block"
```

---

## Task 9: Final whole-feature verification

**Files:** all of the above (no new changes — verification pass)

- [ ] **Step 1: Re-run full verification suite**

Run: `npm run check 2>&1 | tail -3 && npm test 2>&1 | grep "Tests " | tail -1 && npm run build 2>&1 | tail -3`
Expected: 0 type errors / all tests pass (should be 311 + ~12 new = ~323) / build succeeds.

- [ ] **Step 2: URL round-trip smoke**

Manually craft a URL with all the new state and verify a fresh load decodes it cleanly:

```
#sim?a=h200&v=sxm-141&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=12&dk=ib-ndr
```

Open in a fresh browser tab. Verify:
- Sim tab loads
- Single-request disagg block shows TPOT computed at concurrency=12 (via `effectiveConcurrency`)
- LoadSection slider sits at 12
- Calc tab concurrency input reads "12"
- Refreshing URL doesn't lose state

Stop dev server.

- [ ] **Step 3: Push and open PR**

Run finishing-a-development-branch skill to wrap up. Per the project convention, ask the user before pushing/PR'ing — the user expects to approve git-write actions.

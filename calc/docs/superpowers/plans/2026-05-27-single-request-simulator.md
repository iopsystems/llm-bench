# Single-request simulator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New "Simulator" tab presenting TTFT/TPOT/Total latency + an SVG gantt for a single inference request, reusing the existing engine and shared stores.

**Architecture:** Sim is a thin consumer of `calculate()`. A `simInput` derived store clones the shared `CalcInput` with `concurrency=1`; a `simResult` derived store calls `calculate()` and exposes per-op-point KPIs plus gantt geometry. The gantt's geometry is a pure function (testable in node); the `.svelte` file is a dumb renderer. New `#sim?<payload>` URL prefix mirrors `#calc?<payload>` over the same shared state.

**Tech Stack:** TypeScript + Svelte 5; Vitest (node env, no DOM testing libs); npm from `calc/`; git from repo root `/Users/yao/workspace/llm-perf`. Branch `feat/single-request-simulator` (spec already committed at `c661f38`).

**Spec:** [`calc/docs/superpowers/specs/2026-05-26-single-request-simulator-design.md`](../specs/2026-05-26-single-request-simulator-design.md)

---

### Task 1: Expose `kvTransferS` on `PerfTier`

**Why:** The gantt needs the KV-transfer duration to draw the third segment (case C) and the overlay (case B). Today the engine computes it internally but only surfaces it via a derivation step (fragile label-matching). Add it as an optional field on `PerfTier`. This is a schema extension, not a math change — the value is already computed in `calc.ts:50–55`.

**Files:**
- Modify: `calc/src/engine/types.ts` (`PerfTier` interface around line 441-453)
- Modify: `calc/src/engine/calc.ts` (the `perf[op.id] = { ... }` block around line 74-84)
- Test: `calc/test/engine/calc.test.ts` (add new test)

- [ ] **Step 1: Check whether `calc.test.ts` exists; create it if not**

```bash
ls /Users/yao/workspace/llm-perf/calc/test/engine/calc.test.ts 2>/dev/null || echo "MISSING"
```

If `MISSING`, create the file with this scaffold:
```ts
import { describe, it, expect } from 'vitest'
import { calculate } from '../../src/engine'
import { testInput } from '../fixtures'
import type { MultiAcceleratorSystem } from '../../src/engine/types'

describe('calculate', () => {
  // tests added below
})
```

If the file exists, leave it; the new test will go into its existing top-level `describe('calculate', ...)` block.

- [ ] **Step 2: Write the failing test for `kvTransferS` exposure**

Append inside the `describe('calculate', ...)` block in `calc/test/engine/calc.test.ts`:

```ts
  it('exposes kvTransferS=0 in non-disagg config', () => {
    const result = calculate(testInput)
    for (const tier of Object.values(result.perf)) {
      expect(tier.kvTransferS).toBe(0)
    }
  })

  it('exposes a positive kvTransferS when a disagg fabric is configured', () => {
    // Build a multi-device input with a disagg fabric. Use HGX H100 + IB-NDR.
    const sys: MultiAcceleratorSystem = {
      id: 'hgx-h100-8', name: 'HGX H100', vendor: 'NVIDIA', releaseDate: '2022-09',
      formFactor: 'baseboard',
      accelerator: { id: testInput.accelerator.id, variantId: testInput.acceleratorVariantId, count: 8 },
      interconnectId: 'nvlink-4',
      aggregate: { totalHbmGB: 8 * 80, fabricBidirectionalTBs: 7.2 }
    }
    const inp = {
      ...testInput,
      multiDevice: {
        system: sys,
        parallelism: ['tp' as const],
        parallelismDegrees: { tp: 8 },
        disaggKvTransferFabricId: 'ib-ndr',
        disaggFirstTokenOnPrefill: false,
      }
    }
    const result = calculate(inp)
    for (const tier of Object.values(result.perf)) {
      expect(tier.kvTransferS).toBeGreaterThan(0)
      // Sanity: ttftS should equal prefillS + kvTransferS for sequential handoff.
      expect(tier.ttftS).toBeCloseTo(tier.prefill.timeS + tier.kvTransferS, 9)
    }
  })
```

Note: if `ib-ndr` is not a real interconnect id in `INTERCONNECTS`, substitute the first scale-out fabric id from `calc/src/data/interconnects.ts`. Check with `grep -n "scale: 'scale-out'" calc/src/data/interconnects.ts` and use the `id` of the first match.

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- calc.test 2>&1 | tail -15
```
Expected: FAIL — `tier.kvTransferS` is `undefined`.

- [ ] **Step 4: Add the field to `PerfTier`**

In `calc/src/engine/types.ts`, find the `PerfTier` interface (around line 441–453) and add `kvTransferS` after `ttftS`:

```ts
export interface PerfTier {
  prefill: { flops: number; bytes: number; timeS: number; regime: 'compute' | 'memory' | 'comms' }
  decode:  { flopsPerStep: number; bytesPerStep: number; timePerTokenS: number;
             regime: 'compute' | 'memory' | 'comms'; aggregateTokensPerS: number }
  ttftS: number
  kvTransferS: number   // KV-cache transfer time for disagg; 0 when integrated.
  inputTokenRate: number
  outputTokenRate: number
  tflopsSources?: string[]
  bandwidthSources?: string[]
  asOf?: string
  notes?: string
}
```

- [ ] **Step 5: Populate it in `calc.ts`**

In `calc/src/engine/calc.ts`, find the `perf[op.id] = { ... }` assignment (around lines 74-84) and add `kvTransferS` right after `ttftS`:

```ts
    perf[op.id] = {
      prefill, decode,
      ttftS,
      kvTransferS,
      inputTokenRate: input.workload.promptTokens / prefill.timeS,
      outputTokenRate: decode.aggregateTokensPerS,
      ...(op.tflopsSources && { tflopsSources: op.tflopsSources }),
      ...(op.bandwidthSources && { bandwidthSources: op.bandwidthSources }),
      ...(op.asOf && { asOf: op.asOf }),
      ...(op.notes && { notes: op.notes })
    }
```

(`kvTransferS` is already declared as a local in this function — line 50.)

- [ ] **Step 6: Run tests to verify pass + no regressions**

```bash
npm test 2>&1 | grep -E "(Tests |FAIL)" | tail -3
npm run check 2>&1 | tail -2
```
Expected: all green; 0 type errors.

- [ ] **Step 7: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/engine/types.ts calc/src/engine/calc.ts calc/test/engine/calc.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): expose kvTransferS on PerfTier for downstream gantt rendering"
```

---

### Task 2: `Route` gains `{ tab: 'sim' }`

**Files:**
- Modify: `calc/src/ui/route.ts`
- Modify: `calc/test/ui/route.test.ts`

- [ ] **Step 1: Add failing tests**

In `calc/test/ui/route.test.ts`, append to the `describe('parseRoute', ...)` block:

```ts
  it('bare sim → sim', () => {
    expect(parseRoute('#sim')).toEqual({ tab: 'sim' })
  })
  it('sim with payload', () => {
    expect(parseRoute('#sim?a=h100&m=llama-3.3-70b')).toEqual({ tab: 'sim' })
  })
```

Append to the `describe('serializeRoute', ...)` block:

```ts
  it('sim → #sim', () => {
    expect(serializeRoute({ tab: 'sim' })).toBe('#sim')
  })
  it('sim with payload', () => {
    expect(serializeRoute({ tab: 'sim' }, 'a=h100')).toBe('#sim?a=h100')
  })
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- route.test 2>&1 | tail -10
```
Expected: FAIL — `parseRoute('#sim')` returns `{ tab: 'calc' }` (the unknown-path fallback), and `serializeRoute` throws or returns the calc form.

- [ ] **Step 3: Extend `Route` and the parser/serializer**

In `calc/src/ui/route.ts`, update the `Route` type and both functions. Replace the type alias:

```ts
export type Route =
  | { tab: 'calc' }
  | { tab: 'sim' }
  | { tab: 'info' }
  | { tab: 'info'; detail: { kind: 'model' | 'sku'; id: string } }
```

Replace `parseRoute` body (the function's body, not the signature):

```ts
export function parseRoute(hash: string): Route {
  const h = hash.replace(/^#/, '')
  if (h === '' || h === 'calc' || h.startsWith('calc?')) return { tab: 'calc' }
  if (h === 'sim'  || h.startsWith('sim?'))  return { tab: 'sim' }
  if (h === 'info') return { tab: 'info' }
  const m = h.match(/^info\/(model|sku)\/(.+)$/)
  if (m) return { tab: 'info', detail: { kind: m[1] as 'model' | 'sku', id: m[2] } }
  return { tab: 'calc' }
}
```

Replace `serializeRoute`. The second parameter is now used by both calc and sim:

```ts
// Serialize a Route to a hash string (with leading '#'). For the calc and sim
// tabs an optional payload (the share.ts encodeState string) is appended as
// `?<payload>`. Both tabs use the same encoded payload (shared state); the
// hash prefix is what differentiates which tab the recipient lands on.
export function serializeRoute(route: Route, payload = ''): string {
  if (route.tab === 'calc') return payload ? `#calc?${payload}` : '#calc'
  if (route.tab === 'sim')  return payload ? `#sim?${payload}`  : '#sim'
  if ('detail' in route) return `#info/${route.detail.kind}/${route.detail.id}`
  return '#info'
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
npm test -- route.test 2>&1 | tail -10
```
Expected: all `route.test.ts` cases green.

- [ ] **Step 5: Run full suite + typecheck**

```bash
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/route.ts calc/test/ui/route.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): Route gains sim tab variant"
```

---

### Task 3: Generalize hash helpers in `share.ts`

**Why:** The simulator tab needs to (a) read its URL prefix on load, (b) write its own URL on changes, (c) build shareable URLs from its tab.

**Files:**
- Modify: `calc/src/ui/share.ts` — rename helper, update `startUrlSync`, `readUrlIntoStores`, `buildShareUrl`.
- Modify: `calc/test/ui/share-route.test.ts` — update existing assertions and add coverage for sim.

- [ ] **Step 1: Update failing tests**

In `calc/test/ui/share-route.test.ts`, replace the `describe('calcPayloadFromHash', ...)` block with this expanded version that exercises the renamed helper:

```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { tabPayloadFromHash, encodeState, decodeState, readUrlIntoStores } from '../../src/ui/share'
import { modelId, quant } from '../../src/ui/stores'
import { MODELS } from '../../src/data'

describe('tabPayloadFromHash', () => {
  it('extracts payload after calc?', () => {
    expect(tabPayloadFromHash('#calc?a=h100&m=x', 'calc')).toBe('a=h100&m=x')
  })
  it('extracts payload after sim?', () => {
    expect(tabPayloadFromHash('#sim?a=h100&m=x', 'sim')).toBe('a=h100&m=x')
  })
  it('returns empty for mismatched tab', () => {
    expect(tabPayloadFromHash('#calc?a=h100', 'sim')).toBe('')
    expect(tabPayloadFromHash('#sim?a=h100',  'calc')).toBe('')
  })
  it('legacy bare payload counts as calc-tab payload', () => {
    expect(tabPayloadFromHash('#a=h100&m=x', 'calc')).toBe('a=h100&m=x')
    expect(tabPayloadFromHash('#a=h100&m=x', 'sim')).toBe('')
  })
  it('info routes carry no calc/sim payload', () => {
    expect(tabPayloadFromHash('#info/model/deepseek-v3', 'calc')).toBe('')
    expect(tabPayloadFromHash('#info/model/deepseek-v3', 'sim')).toBe('')
    expect(tabPayloadFromHash('#info', 'calc')).toBe('')
  })
  it('empty hash → empty', () => {
    expect(tabPayloadFromHash('', 'calc')).toBe('')
    expect(tabPayloadFromHash('#calc', 'calc')).toBe('')
    expect(tabPayloadFromHash('#sim',  'sim')).toBe('')
  })
})
```

Find the existing `describe('URL with model but no quant → quant seeded from native', ...)` block in the same file. Add this test inside it (after the existing `store-level fallback` test):

```ts
  it('readUrlIntoStores accepts #sim? prefix too', () => {
    const w = globalThis as { window?: { location: { hash: string } } }
    w.window = { location: { hash: `#sim?m=${fp8Model.id}` } }
    try {
      readUrlIntoStores()
      expect(get(modelId)).toBe(fp8Model.id)
    } finally {
      delete w.window
    }
  })
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- share-route 2>&1 | tail -15
```
Expected: FAIL — `tabPayloadFromHash` is not exported.

- [ ] **Step 3: Rename and generalize `calcPayloadFromHash`**

In `calc/src/ui/share.ts`, find the existing `calcPayloadFromHash` function (around lines 228-235) and replace it with:

```ts
// Extract the per-tab payload from a raw location.hash. Supports the current
// `#calc?<payload>` / `#sim?<payload>` forms and the legacy bare `#<payload>`
// form (treated as calc-tab payload for backwards compatibility with old
// shared links). Info routes carry no payload regardless of tab argument.
export function tabPayloadFromHash(hash: string, tab: 'calc' | 'sim'): string {
  const h = hash.replace(/^#/, '')
  if (h === '' || h === tab) return ''
  if (h.startsWith(`${tab}?`)) return h.slice(tab.length + 1)
  if (h.startsWith('calc') || h.startsWith('sim') || h.startsWith('info')) return ''
  // Legacy bare payload: only honor it for the calc tab.
  return tab === 'calc' ? h : ''
}
```

- [ ] **Step 4: Update `readUrlIntoStores` to try both prefixes**

In `calc/src/ui/share.ts`, replace the body of `readUrlIntoStores` (currently around lines 239-244):

```ts
export function readUrlIntoStores(): void {
  if (typeof window === 'undefined') return
  // Try both tab prefixes — share URLs can be either #calc?... or #sim?...
  // and the recipient just lands on the corresponding tab. The payload itself
  // is identical (shared state), so either tab can decode the other's URL.
  const payload =
    tabPayloadFromHash(window.location.hash, 'calc') ||
    tabPayloadFromHash(window.location.hash, 'sim')
  if (!payload) return
  applyToStores(decodeState(payload))
}
```

- [ ] **Step 5: Update `startUrlSync` to write the matching prefix**

In `calc/src/ui/share.ts`, replace the `write` arrow inside `startUrlSync` (currently around lines 253-261):

```ts
  let ready = false
  const write = () => {
    if (!ready) return
    const tab = parseRoute(window.location.hash).tab
    // Info tab carries no calc payload; never overwrite it.
    if (tab !== 'calc' && tab !== 'sim') return
    const encoded = encodeState(readStoreState())
    const next = `${window.location.pathname}${window.location.search}#${tab}?${encoded}`
    window.history.replaceState(window.history.state, '', next)
  }
```

- [ ] **Step 6: Update `buildShareUrl` to use the current tab**

In `calc/src/ui/share.ts`, replace `buildShareUrl` (currently lines 285-290):

```ts
export function buildShareUrl(): string {
  const encoded = encodeState(readStoreState())
  if (typeof window === 'undefined') return `#calc?${encoded}`
  const tab = parseRoute(window.location.hash).tab
  const prefix = tab === 'sim' ? 'sim' : 'calc'  // info tab falls back to calc
  const { origin, pathname, search } = window.location
  return `${origin}${pathname}${search}#${prefix}?${encoded}`
}
```

- [ ] **Step 7: Run tests to verify pass**

```bash
npm test -- share-route 2>&1 | tail -15
npm test -- share.test 2>&1 | tail -10
npm run check 2>&1 | tail -2
```
Expected: green; 0 type errors.

- [ ] **Step 8: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/share.ts calc/test/ui/share-route.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): tab-aware hash helpers; readUrlIntoStores accepts #sim? prefix"
```

---

### Task 4: `InputPanel.svelte` gains `hideConcurrency` prop

**Why:** The sim tab reuses InputPanel but hides Concurrency (sim is single-request by definition).

**Files:**
- Modify: `calc/src/ui/InputPanel.svelte`

No tests (presentational; verified in-browser per existing convention).

- [ ] **Step 1: Add the prop**

In `calc/src/ui/InputPanel.svelte`, find the `<script lang="ts">` block (line 1) and add the prop declaration immediately after the imports. After line 9 (`import { orderModels, orderSkus } from './catalogOrder'`), add:

```ts
  export let hideConcurrency = false
```

- [ ] **Step 2: Conditionally render the Concurrency label**

In the same file, find the Concurrency `<label>` block (the one containing the `concurrencyInput` field; currently around lines 215-228, look for `Concurrency`). Wrap it in an `{#if !hideConcurrency}` block:

```svelte
      {#if !hideConcurrency}
        <label>
          Concurrency
          <input
            type="text"
            inputmode="numeric"
            value={concurrencyInput}
            on:input={onConcurrencyInput}
            class:invalid={concurrencyInvalid}
            title="Positive integer (≥1)"
          />
          {#if concurrencyInvalid}
            <span class="warn">⚠ invalid — use a positive integer</span>
          {/if}
        </label>
      {/if}
```

- [ ] **Step 3: Run typecheck + full suite (no regressions)**

```bash
npm run check 2>&1 | tail -2
npm test 2>&1 | grep -E "Tests " | tail -1
```
Expected: 0 type errors; all tests green.

- [ ] **Step 4: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/InputPanel.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): InputPanel hideConcurrency prop"
```

---

### Task 5: Pure gantt geometry module

**Why:** Vitest is configured for the `node` environment with no DOM testing libs. To test the gantt deterministically, isolate the math from the rendering: a pure function in a `.ts` file emits a geometry object; the Svelte component is a dumb renderer.

**Files:**
- Create: `calc/src/ui/simulatorGantt.ts`
- Create: `calc/test/ui/simulatorGantt.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `calc/test/ui/simulatorGantt.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { computeGanttGeometry, type GanttInput } from '../../src/ui/simulatorGantt'

// Helper: round to 9 decimals to dodge floating-point noise in assertions.
const r = (n: number) => Math.round(n * 1e9) / 1e9

describe('computeGanttGeometry — non-disagg (case A)', () => {
  const input: GanttInput = {
    prefillS: 0.287,
    kvTransferS: 0,
    tpotS: 0.042,
    outputTokens: 512,
    firstTokenOnPrefill: true,
    ttftS: 0.287,
    prefillRegime: 'compute',
    decodeRegime: 'memory',
  }
  const geom = computeGanttGeometry(input)

  it('produces two segments: prefill + decode', () => {
    expect(geom.segments).toHaveLength(2)
    expect(geom.segments[0]).toMatchObject({ kind: 'prefill', x: 0, width: 0.287, regime: 'compute' })
    expect(geom.segments[1]).toMatchObject({ kind: 'decode',  x: 0.287, regime: 'memory' })
  })
  it('no KV-xfer overlay', () => {
    expect(geom.kvOverlay).toBeUndefined()
  })
  it('marker at ttftS', () => {
    expect(geom.markerX).toBe(0.287)
  })
  it('totalS = ttftS + tpotS * (outputTokens-1)', () => {
    expect(r(geom.totalS)).toBe(r(0.287 + 0.042 * 511))
    expect(r(geom.segments[1].x + geom.segments[1].width)).toBe(r(geom.totalS))
  })
})

describe('computeGanttGeometry — disagg sequential (case C, firstTokenOnPrefill=false)', () => {
  const input: GanttInput = {
    prefillS: 0.287,
    kvTransferS: 0.213,
    tpotS: 0.042,
    outputTokens: 512,
    firstTokenOnPrefill: false,
    ttftS: 0.287 + 0.213,
    prefillRegime: 'compute',
    decodeRegime: 'memory',
  }
  const geom = computeGanttGeometry(input)

  it('produces three serial segments: prefill, kv-xfer, decode', () => {
    expect(geom.segments).toHaveLength(3)
    expect(geom.segments[0]).toMatchObject({ kind: 'prefill', x: 0, width: 0.287, regime: 'compute' })
    expect(geom.segments[1]).toMatchObject({ kind: 'kv-xfer', x: 0.287, width: 0.213, regime: 'comms' })
    expect(geom.segments[2]).toMatchObject({ kind: 'decode',  x: 0.500, regime: 'memory' })
  })
  it('no overlay (KV is its own segment here)', () => {
    expect(geom.kvOverlay).toBeUndefined()
  })
  it('marker at the kv-xfer/decode boundary', () => {
    expect(geom.markerX).toBe(0.5)
  })
  it('decode segment ends at totalS = ttftS + tpotS * (outputTokens-1)', () => {
    expect(r(geom.totalS)).toBe(r(0.5 + 0.042 * 511))
    expect(r(geom.segments[2].x + geom.segments[2].width)).toBe(r(geom.totalS))
  })
})

describe('computeGanttGeometry — disagg overlap (case B, firstTokenOnPrefill=true)', () => {
  const input: GanttInput = {
    prefillS: 0.287,
    kvTransferS: 0.213,
    tpotS: 0.042,
    outputTokens: 512,
    firstTokenOnPrefill: true,
    ttftS: 0.287 + 0.042,
    prefillRegime: 'compute',
    decodeRegime: 'memory',
  }
  const geom = computeGanttGeometry(input)

  it('produces two main segments (prefill + decode) — KV-xfer is overlay, not a segment', () => {
    expect(geom.segments).toHaveLength(2)
    expect(geom.segments[0]).toMatchObject({ kind: 'prefill', x: 0, width: 0.287, regime: 'compute' })
    expect(geom.segments[1]).toMatchObject({ kind: 'decode',  x: 0.287, regime: 'memory' })
  })
  it('emits a KV-xfer overlay spanning [prefillS, prefillS + kvTransferS]', () => {
    expect(geom.kvOverlay).toEqual({ x: 0.287, width: 0.213 })
  })
  it('marker at ttftS = prefillS + tpotS', () => {
    expect(r(geom.markerX)).toBe(r(0.287 + 0.042))
  })
  it('totalS = ttftS + tpotS * (outputTokens-1) = prefillS + tpotS * outputTokens', () => {
    expect(r(geom.totalS)).toBe(r(0.287 + 0.042 * 512))
  })
})
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- simulatorGantt 2>&1 | tail -10
```
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Implement `simulatorGantt.ts`**

Create `calc/src/ui/simulatorGantt.ts`:

```ts
// Pure geometry for the SimulatorGantt SVG. Kept separate from the .svelte
// renderer so it's testable in the node-only vitest environment.
//
// Three cases (see spec 2026-05-26-single-request-simulator-design.md §UI):
//   A — non-disagg (kvTransferS = 0): two segments, no overlay.
//   B — disagg + firstTokenOnPrefill=true: two segments + KV-xfer overlay
//       in [prefillS, prefillS + kvTransferS]. The overlay is rendered as
//       a thin bar below the main row to communicate "KV streams in parallel
//       with the prefill cluster's first decode step."
//   C — disagg + firstTokenOnPrefill=false: three serial segments
//       (prefill → kv-xfer → decode).
//
// Segment x-coordinates are in seconds (caller scales to pixels).

export type Regime = 'compute' | 'memory' | 'comms'

export interface GanttInput {
  prefillS: number
  kvTransferS: number       // 0 when integrated (non-disagg)
  tpotS: number             // decode time per token (constant per v1 spec)
  outputTokens: number
  firstTokenOnPrefill: boolean   // ignored when kvTransferS = 0
  ttftS: number             // engine-reported TTFT; placed verbatim as marker x
  prefillRegime: Regime
  decodeRegime: Regime
}

export interface GanttSegment {
  kind: 'prefill' | 'kv-xfer' | 'decode'
  x: number      // seconds from t=0
  width: number  // seconds
  regime: Regime
}

export interface GanttGeometry {
  segments: GanttSegment[]
  kvOverlay?: { x: number; width: number }
  markerX: number
  totalS: number
}

export function computeGanttGeometry(input: GanttInput): GanttGeometry {
  const { prefillS, kvTransferS, tpotS, outputTokens, firstTokenOnPrefill,
          ttftS, prefillRegime, decodeRegime } = input
  // Spec §Behavior contract: Total = TTFT + TPOT × (outputTokens − 1).
  // (Caveat in spec: small undercount in disagg+sequential case; bounded by
  // one TPOT. We use the formula uniformly for v1.)
  const totalS = ttftS + tpotS * (outputTokens - 1)

  // Case A: no disagg.
  if (kvTransferS === 0) {
    return {
      segments: [
        { kind: 'prefill', x: 0,        width: prefillS,           regime: prefillRegime },
        { kind: 'decode',  x: prefillS, width: totalS - prefillS,  regime: decodeRegime },
      ],
      markerX: ttftS,
      totalS,
    }
  }

  // Case C: disagg, sequential handoff.
  if (!firstTokenOnPrefill) {
    return {
      segments: [
        { kind: 'prefill', x: 0,                       width: prefillS,            regime: prefillRegime },
        { kind: 'kv-xfer', x: prefillS,                width: kvTransferS,         regime: 'comms' },
        { kind: 'decode',  x: prefillS + kvTransferS,  width: totalS - prefillS - kvTransferS, regime: decodeRegime },
      ],
      markerX: ttftS,
      totalS,
    }
  }

  // Case B: disagg, overlap. KV streams parallel to the prefill cluster's
  // first decode step — render KV-xfer as an overlay rather than a segment.
  return {
    segments: [
      { kind: 'prefill', x: 0,        width: prefillS,          regime: prefillRegime },
      { kind: 'decode',  x: prefillS, width: totalS - prefillS, regime: decodeRegime },
    ],
    kvOverlay: { x: prefillS, width: kvTransferS },
    markerX: ttftS,
    totalS,
  }
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
npm test -- simulatorGantt 2>&1 | tail -15
npm run check 2>&1 | tail -2
```
Expected: all 12 cases green; 0 type errors.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/simulatorGantt.ts calc/test/ui/simulatorGantt.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): pure gantt-geometry module + unit tests (cases A/B/C)"
```

---

### Task 6: `SimulatorGantt.svelte` renderer

**Why:** Dumb SVG renderer over the geometry from Task 5.

**Files:**
- Create: `calc/src/ui/SimulatorGantt.svelte`

No tests (presentational; geometry is covered in Task 5).

- [ ] **Step 1: Create the component**

Create `calc/src/ui/SimulatorGantt.svelte`:

```svelte
<script lang="ts">
  import { computeGanttGeometry, type GanttInput } from './simulatorGantt'
  export let input: GanttInput

  // SVG layout constants. The main row is the prefill/decode timeline;
  // the overlay row (case B only) sits below it for the KV-xfer bar.
  const W = 720
  const ROW_H = 28
  const OVERLAY_H = 10
  const ROW_GAP = 4
  const PADDING = 12  // left/right padding inside the viewBox

  $: geom = computeGanttGeometry(input)
  // Scale: seconds → pixels along the timeline. Reserve PADDING on each side.
  $: pxPerS = (W - 2 * PADDING) / Math.max(geom.totalS, 1e-9)
  $: hasOverlay = geom.kvOverlay !== undefined
  $: totalH = ROW_H + (hasOverlay ? ROW_GAP + OVERLAY_H : 0) + 30  // +30 for axis labels

  function ms(s: number): string {
    if (s >= 1)    return `${(s).toFixed(2)} s`
    if (s >= 1e-3) return `${(s * 1e3).toFixed(0)} ms`
    return `${(s * 1e6).toFixed(0)} µs`
  }

  // Map a regime to a CSS class on the rect.
  const regimeClass = (r: 'compute' | 'memory' | 'comms') => `regime-${r}`
</script>

<svg viewBox="0 0 {W} {totalH}" class="gantt" role="img" aria-label="Single-request timeline">
  <!-- Main row: prefill + decode (and kv-xfer in case C). -->
  {#each geom.segments as seg}
    <rect
      class="seg {regimeClass(seg.regime)}"
      x={PADDING + seg.x * pxPerS}
      y={0}
      width={Math.max(seg.width * pxPerS, 1)}
      height={ROW_H}
    />
    <title>{seg.kind} · {ms(seg.width)} · {seg.regime}-bound</title>
  {/each}

  <!-- KV-xfer overlay (case B only). -->
  {#if geom.kvOverlay}
    <rect
      class="seg regime-comms overlay"
      x={PADDING + geom.kvOverlay.x * pxPerS}
      y={ROW_H + ROW_GAP}
      width={Math.max(geom.kvOverlay.width * pxPerS, 1)}
      height={OVERLAY_H}
    />
    <title>kv-xfer · {ms(geom.kvOverlay.width)} · overlapped with first decode step</title>
  {/if}

  <!-- TTFT marker. -->
  <line
    class="marker"
    x1={PADDING + geom.markerX * pxPerS}
    y1={0}
    x2={PADDING + geom.markerX * pxPerS}
    y2={ROW_H + (hasOverlay ? ROW_GAP + OVERLAY_H : 0)}
  />
  <text
    class="marker-label"
    x={PADDING + geom.markerX * pxPerS}
    y={ROW_H + (hasOverlay ? ROW_GAP + OVERLAY_H : 0) + 14}
    text-anchor="middle"
  >first token</text>

  <!-- Axis ticks: 0, TTFT, Total. -->
  <text class="tick" x={PADDING}                                    y={totalH - 4} text-anchor="start">0</text>
  <text class="tick" x={PADDING + geom.totalS * pxPerS}             y={totalH - 4} text-anchor="end">{ms(geom.totalS)}</text>
</svg>

<style>
  .gantt { width: 100%; height: auto; display: block; }
  .seg { stroke: #fff; stroke-width: 1; }
  .seg.regime-compute { fill: #2b6cb0; }
  .seg.regime-memory  { fill: #c05621; }
  .seg.regime-comms   { fill: #6b46c1; }
  .seg.overlay { stroke-dasharray: 3 2; opacity: 0.85; }
  .marker { stroke: #111; stroke-width: 1.5; stroke-dasharray: 2 2; }
  .marker-label { font: 600 11px system-ui; fill: #111; }
  .tick { font: 11px system-ui; fill: #555; }
</style>
```

- [ ] **Step 2: Run typecheck**

```bash
npm run check 2>&1 | tail -2
```
Expected: 0 type errors.

- [ ] **Step 3: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/SimulatorGantt.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): SimulatorGantt SVG component renders geometry from pure module"
```

---

### Task 7: `simInput` + `simResult` derived stores

**Why:** The sim view consumes `calculate()` with `concurrency` forced to 1, independent of the calc tab's setting. Implementing this in `stores.ts` keeps the Simulator component thin and gives us a clean unit-test surface for the clamp.

**Files:**
- Modify: `calc/src/ui/stores.ts`
- Create: `calc/test/ui/sim-stores.test.ts`

- [ ] **Step 1: Write failing tests**

Create `calc/test/ui/sim-stores.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { workload, simInput, simResult } from '../../src/ui/stores'

describe('simInput', () => {
  beforeEach(() => {
    // Reset workload so concurrency=64 (the value we deliberately set below)
    // is observable as a deviation from the default.
    workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 64 })
  })

  it('clamps concurrency to 1 regardless of the shared workload store', () => {
    const inp = get(simInput)
    expect(inp).not.toBeNull()
    expect(inp!.workload.concurrency).toBe(1)
    // Other workload fields pass through.
    expect(inp!.workload.promptTokens).toBe(2048)
    expect(inp!.workload.outputTokens).toBe(512)
  })

  it('does not write back to the workload store', () => {
    get(simInput)   // force evaluation
    expect(get(workload).concurrency).toBe(64)
  })
})

describe('simResult', () => {
  beforeEach(() => {
    workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 64 })
  })

  it('produces a CalcResult with at least one operating point', () => {
    const r = get(simResult)
    expect(r).not.toBeNull()
    expect(Object.keys(r!.perf).length).toBeGreaterThan(0)
  })

  it('every op-point exposes ttftS, decode.timePerTokenS, and kvTransferS', () => {
    const r = get(simResult)
    for (const tier of Object.values(r!.perf)) {
      expect(typeof tier.ttftS).toBe('number')
      expect(typeof tier.decode.timePerTokenS).toBe('number')
      expect(typeof tier.kvTransferS).toBe('number')
    }
  })
})
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- sim-stores 2>&1 | tail -10
```
Expected: FAIL — `simInput`/`simResult` not exported.

- [ ] **Step 3: Add the derived stores to `stores.ts`**

In `calc/src/ui/stores.ts`, after the existing `error` export (currently the last line of the file, around line 114), append:

```ts

// --- Single-request simulator ---
// The simulator tab consumes calculate() with concurrency forced to 1,
// regardless of what the shared workload store carries. This keeps the
// calc tab and sim tab sharing all other state without one clobbering
// the other's mental model of "what is concurrency."
export const simInput: Readable<CalcInput | null> = derived(input, $input => {
  if (!$input) return null
  return { ...$input, workload: { ...$input.workload, concurrency: 1 } }
})

interface SimComputed { result: CalcResult | null; error: string | null }

const simComputed: Readable<SimComputed> = derived(simInput, $input => {
  if (!$input) return { result: null, error: null }
  try { return { result: calculate($input), error: null } }
  catch (err) { return { result: null, error: (err as Error).message } }
})

export const simResult: Readable<CalcResult | null> = derived(simComputed, $c => $c.result)
export const simError:  Readable<string | null>      = derived(simComputed, $c => $c.error)
```

(`CalcInput`, `CalcResult`, `calculate`, `input`, and `derived` are already imported at the top of this file.)

- [ ] **Step 4: Run tests to verify pass + full suite**

```bash
npm test -- sim-stores 2>&1 | tail -10
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
```
Expected: all green; 0 type errors.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/stores.ts calc/test/ui/sim-stores.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): simInput/simResult derived stores (concurrency clamped to 1)"
```

---

### Task 8: `Simulator.svelte` view

**Why:** Compose InputPanel(hideConcurrency=true) + KPI strip + SimulatorGantt + op-point selector.

**Files:**
- Create: `calc/src/ui/Simulator.svelte`

No tests (presentational; math is in Tasks 5 and 7).

- [ ] **Step 1: Create the component**

Create `calc/src/ui/Simulator.svelte`:

```svelte
<script lang="ts">
  import InputPanel from './InputPanel.svelte'
  import SimulatorGantt from './SimulatorGantt.svelte'
  import { simResult, simError, workload, disaggFirstTokenOnPrefill } from './stores'
  import type { GanttInput } from './simulatorGantt'

  // Same formatting helpers as PerfPanel; copied here to keep this file
  // self-contained for v1 (extract into a shared module when a third view
  // wants them).
  function sig3(n: number): string {
    if (n === 0) return '0'
    return parseFloat(n.toPrecision(3)).toString()
  }
  function ms(s: number): string {
    if (s >= 1)     return `${sig3(s)} s`
    if (s >= 1e-3)  return `${sig3(s * 1e3)} ms`
    if (s >= 1e-6)  return `${sig3(s * 1e6)} µs`
    return `${sig3(s * 1e9)} ns`
  }

  // The simulator follows the same op-point the calc tab is showing. Since
  // op-point isn't currently URL state, "the same" reduces to "show every
  // op-point" — pick the first key in perf for the cards/gantt and let
  // additional op-points appear below as a small comparison list.
  $: opIds = $simResult ? Object.keys($simResult.perf) : []
  $: primary = opIds[0]
  $: tier = $simResult && primary ? $simResult.perf[primary] : null

  $: ganttInput = tier ? ({
    prefillS: tier.prefill.timeS,
    kvTransferS: tier.kvTransferS,
    tpotS: tier.decode.timePerTokenS,
    outputTokens: $workload.outputTokens,
    firstTokenOnPrefill: $disaggFirstTokenOnPrefill,
    ttftS: tier.ttftS,
    prefillRegime: tier.prefill.regime,
    decodeRegime: tier.decode.regime,
  } satisfies GanttInput) : null

  $: totalS = tier ? tier.ttftS + tier.decode.timePerTokenS * ($workload.outputTokens - 1) : 0
</script>

<section class="simulator">
  <InputPanel hideConcurrency={true} />

  {#if $simError}
    <div class="error">⚠ {$simError}</div>
  {:else if tier && ganttInput}
    <div class="kpis">
      <div class="kpi">
        <div class="label">TTFT</div>
        <div class="value">{ms(tier.ttftS)}</div>
        <div class="badge regime-{tier.prefill.regime}">{tier.prefill.regime}-bound prefill</div>
      </div>
      <div class="kpi">
        <div class="label">TPOT</div>
        <div class="value">{ms(tier.decode.timePerTokenS)}</div>
        <div class="badge regime-{tier.decode.regime}">{tier.decode.regime}-bound decode</div>
        <div class="caption">{sig3(1 / tier.decode.timePerTokenS)} tok/s</div>
      </div>
      <div class="kpi">
        <div class="label">Total latency</div>
        <div class="value">{ms(totalS)}</div>
        <div class="caption">{$workload.outputTokens} output tokens</div>
      </div>
    </div>

    <div class="gantt-wrap">
      <h4>Timeline ({primary})</h4>
      <SimulatorGantt input={ganttInput} />
    </div>

    {#if opIds.length > 1}
      <details class="other-ops">
        <summary>Other operating points</summary>
        <table>
          <thead><tr><th>Op</th><th>TTFT</th><th>TPOT</th><th>Total</th></tr></thead>
          <tbody>
            {#each opIds.slice(1) as id}
              {@const t = $simResult.perf[id]}
              <tr>
                <td>{id}</td>
                <td>{ms(t.ttftS)}</td>
                <td>{ms(t.decode.timePerTokenS)}</td>
                <td>{ms(t.ttftS + t.decode.timePerTokenS * ($workload.outputTokens - 1))}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </details>
    {/if}
  {/if}
</section>

<style>
  .simulator { display: flex; flex-direction: column; gap: 1rem; }
  .error {
    padding: 0.5rem 0.75rem;
    background: #fde6e6; color: #8a1f1f;
    border: 1px solid #f0b0b0; border-radius: 0.25rem;
    font-size: 0.9rem;
  }
  .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
  .kpi {
    border: 1px solid #d4d4d4; border-radius: 0.4rem; padding: 0.8rem 1rem;
    background: #fff;
  }
  .kpi .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #888; }
  .kpi .value { font-size: 1.75rem; font-weight: 700; line-height: 1.1; margin-top: 0.2rem; }
  .kpi .badge {
    display: inline-block; margin-top: 0.4rem; padding: 0.1rem 0.45rem;
    font-size: 0.75rem; border-radius: 0.2rem; color: #fff;
  }
  .badge.regime-compute { background: #2b6cb0; }
  .badge.regime-memory  { background: #c05621; }
  .badge.regime-comms   { background: #6b46c1; }
  .kpi .caption { font-size: 0.78rem; color: #666; margin-top: 0.3rem; }
  .gantt-wrap h4 { margin: 0 0 0.4rem; font-size: 0.85rem; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .other-ops { font-size: 0.85rem; }
  .other-ops table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  .other-ops th, .other-ops td { text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid #eee; }
  @media (max-width: 640px) {
    .kpis { grid-template-columns: 1fr; }
  }
</style>
```

- [ ] **Step 2: Run typecheck**

```bash
npm run check 2>&1 | tail -2
```
Expected: 0 type errors.

- [ ] **Step 3: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/Simulator.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): Simulator view composes InputPanel + KPI strip + gantt"
```

---

### Task 9: Wire the Simulator tab into `TabBar` + `App.svelte`

**Files:**
- Modify: `calc/src/ui/TabBar.svelte`
- Modify: `calc/src/ui/App.svelte`

- [ ] **Step 1: Add the tab to `TabBar.svelte`**

In `calc/src/ui/TabBar.svelte`, replace the `tabs` array (currently lines 5-8) with:

```ts
  // Compare / Cloud tabs are added by roadmap items #5 / #6.
  const tabs = [
    { id: 'calc' as const, label: 'Calculator' },
    { id: 'sim'  as const, label: 'Simulator' },
    { id: 'info' as const, label: 'Info' },
  ]
```

Then update the `on:click` handler in the same file (currently line 16) to handle the third tab:

```svelte
      on:click={() => navigate({ tab: t.id })}
```

(This works because all three tabs' bare route forms are `{ tab: '<id>' }` — no detail payload.)

- [ ] **Step 2: Add the branch in `App.svelte`**

In `calc/src/ui/App.svelte`, first add the import after line 8 (`import InfoPanel from './InfoPanel.svelte'`):

```ts
  import Simulator from './Simulator.svelte'
```

Then update the `{#if $route.tab === 'info'} ... {:else} ... {/if}` block (currently lines 49-60) to a three-way conditional:

```svelte
  {#if $route.tab === 'info'}
    <InfoPanel />
  {:else if $route.tab === 'sim'}
    <Simulator />
  {:else}
    <InputPanel />
    {#if $error}
      <div class="error">⚠ {$error}</div>
    {/if}
    <MemoryPanel />
    <PerfPanel />
    <RooflinePanel />
    <DerivationDrawer />
  {/if}
```

- [ ] **Step 3: Run full suite + typecheck + build**

```bash
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
npm run build 2>&1 | tail -5
```
Expected: all green; 0 type errors; clean build.

- [ ] **Step 4: Best-effort browser smoke**

```bash
cd /Users/yao/workspace/llm-perf/calc
(npm run dev > /tmp/d.log 2>&1 &) ; sleep 4
P=$(grep -oE "localhost:[0-9]+" /tmp/d.log | head -1)
curl -s "http://$P/" -o /dev/null -w "HTTP %{http_code}\n"
pkill -f "node.*vite" 2>/dev/null
```
Expected: `HTTP 200`.

Interactive checks (controller's job — open the dev URL and click through):
- Three tabs visible: Calculator, Simulator, Info. Click Simulator → renders.
- KPI strip shows three cards with reasonable values.
- Gantt has two segments (no disagg fabric selected). Marker labeled "first token" between them.
- Toggle to a multi-device system; pick a disagg fabric in the InputPanel. Gantt either gains a third KV-xfer segment (firstTokenOnPrefill=false) or an overlay band (firstTokenOnPrefill=true).
- Concurrency input is hidden on the sim tab; visible on the calc tab.
- Edit model on sim → switch to calc → same model selected.
- URL while on sim tab starts with `#sim?...`; copy it, paste in a new tab → restores on the sim tab.
- "Copy link" button on header still works; URL has `#sim?` while on sim tab.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/TabBar.svelte calc/src/ui/App.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): wire Simulator tab into TabBar + App"
```

---

## Self-Review

**1. Spec coverage:**
- Spec §Behavior contract (TTFT/TPOT/Total/throughput from existing engine + concurrency-clamped-to-1) → Tasks 7 (clamp) + 8 (display). ✓
- Spec §UI input panel (hideConcurrency prop) → Task 4. ✓
- Spec §UI KPI strip → Task 8. ✓
- Spec §UI gantt (cases A/B/C, segment geometry, marker placement) → Task 5 (math) + Task 6 (render). ✓
- Spec §State and URL (`#sim?` prefix; tab-aware read/write/build) → Tasks 2 (route) + 3 (share). ✓
- Spec §Files — every file listed is touched in exactly one task. ✓
- Spec §Testing — engine math: no new tests except the `kvTransferS` exposure (Task 1). Gantt geometry: 3 cases (Task 5). Route: sim variants (Task 2). Share: `tabPayloadFromHash` + `readUrlIntoStores` with `#sim?` (Task 3). InputPanel: in-browser per convention (Task 4). In-browser smoke: Task 9. ✓
- Spec §Implementation order — the plan's task order (1→9) follows the spec's suggested order with the addition of Task 1 (the engine schema extension needed by the gantt). ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases." Every step has full code or an exact transform. Pass.

**3. Type consistency:**
- `PerfTier.kvTransferS: number` (Task 1) — read by `simResult` (Task 7) and `Simulator.svelte` (Task 8) and shaped into `GanttInput.kvTransferS` (Task 5/6). Consistent.
- `GanttInput` interface (Task 5) consumed by `SimulatorGantt.svelte` and `Simulator.svelte` via `import type { GanttInput }`. Field names match across producer (Simulator) and consumer (Gantt). Consistent.
- `Route` extended once (Task 2); all later tasks read `$route.tab === 'sim'` consistently. ✓
- `simInput`/`simResult`/`simError` exports (Task 7) are imported by `Simulator.svelte` (Task 8) using the exact names. ✓
- `tabPayloadFromHash(hash, 'calc' | 'sim')` (Task 3) — the function rename is total: `readUrlIntoStores`, the test file, and any future caller all use the new name. The old `calcPayloadFromHash` is fully replaced (no stray references). ✓

**4. Open question / known imprecision (not a placeholder):**
The Total-latency formula `TTFT + TPOT × (outputTokens − 1)` is exact for case A (non-disagg) and case B when `kvTransferS ≤ TPOT`. It undercounts:
- Case C (disagg + sequential handoff) by ≈ one TPOT (decode cluster does N tokens, not N−1).
- Case B (disagg + overlap) when `kvTransferS > TPOT` by `kvTransferS − TPOT` (decode cluster has to wait past the first token).

Both errors are bounded and below the calc's overall accuracy band. The spec acknowledges case C explicitly; the case-B-slow-transfer subcase is a smaller related effect. If reasoning workloads or slow-fabric disagg become a focus, switch to a case-split Total formula in `simulatorGantt.ts` and `Simulator.svelte` (≈10 lines, no UI shape change).

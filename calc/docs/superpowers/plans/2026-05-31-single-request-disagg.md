# Single-request simulator — PD-disagg configuration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PD-disagg as the simulator tab's second stacked configuration block (below the monolithic one), with a dedicated `DisaggInputPanel`, a hoisted-from-MultiDeviceConfig engine schema for disagg, and 5 new RoCE/Spectrum-X catalog entries with accelerator-family eligibility filtering.

**Architecture:** Three layers of change. (1) **Engine**: move `disaggKvTransferFabricId` / `disaggFirstTokenOnPrefill` from `MultiDeviceConfig` to `CalcInput` top-level so disagg works without a multi-device system. (2) **Catalog**: 5 new Ethernet-family entries + a `compatibleAcceleratorIds` field on `InterconnectSpec` for scale-up fabrics. (3) **UI**: a new `DisaggInputPanel.svelte` lives inside the simulator's disagg block; `Simulator.svelte` renders two stacked blocks (monolithic + disagg) from separate derived stores `simResultMonolithic` / `simResultDisagg`.

**Tech Stack:** TypeScript + Svelte 5; Vitest (node env, no DOM testing libs); npm from `calc/`; git from repo root `/Users/yao/workspace/llm-perf`. Branch `feat/single-request-disagg` (spec committed at `68274a3`).

**Spec:** [`calc/docs/superpowers/specs/2026-05-31-single-request-disagg-design.md`](../specs/2026-05-31-single-request-disagg-design.md)

---

### Task 1: Hoist disagg fields out of `MultiDeviceConfig`

**Why:** Today the engine reads `input.multiDevice?.disaggKvTransferFabricId`, which forces the UI to gate disagg on system selection. We want two single-chip nodes over RoCE-400 to be expressible, so the fields move to the top of `CalcInput`. Math is unchanged — pure schema migration.

**Files:**
- Modify: `calc/src/engine/types.ts` (`MultiDeviceConfig` lines 397-411, `CalcInput` lines 413-420)
- Modify: `calc/src/engine/calc.ts` (lines 50-61)
- Modify: `calc/src/ui/stores.ts` (`multiDevice` derived at line 56-76, `input` derived at line 78-103)
- Modify: `calc/test/engine/calc.test.ts` (the existing disagg test that constructs `multiDevice.disaggKvTransferFabricId`)

- [ ] **Step 1: Update the existing test (TDD — change the contract first)**

In `calc/test/engine/calc.test.ts`, find the existing test `exposes a positive kvTransferS when a disagg fabric is configured`. It constructs an input like:

```ts
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
```

Replace the inner `multiDevice` block to drop disagg fields, and add them at the top level:

```ts
const inp = {
  ...testInput,
  multiDevice: {
    system: sys,
    parallelism: ['tp' as const],
    parallelismDegrees: { tp: 8 },
  },
  disaggKvTransferFabricId: 'ib-ndr',
  disaggFirstTokenOnPrefill: false,
}
```

Then append a brand-new test (after the existing disagg test, before the closing `})` of the `describe`):

```ts
  it('disagg works without a multiDevice config (single-chip + scale-out fabric)', () => {
    // Two single-chip nodes connected by a scale-out fabric — no system selected.
    const inp = {
      ...testInput,
      disaggKvTransferFabricId: 'roce-400',
      disaggFirstTokenOnPrefill: false,
    }
    const result = calculate(inp)
    for (const tier of Object.values(result.perf)) {
      expect(tier.kvTransferS).toBeGreaterThan(0)
      expect(tier.ttftS).toBeCloseTo(tier.prefill.timeS + tier.kvTransferS, 9)
    }
  })
```

If `roce-400` doesn't exist in `INTERCONNECTS` yet (Task 2 adds it), substitute `'ib-ndr'` here too — Task 2 will swap it back. Or skip the test until Task 2 lands; both options OK. The simplest path: use `'ib-ndr'` now and update to `'roce-400'` as part of Task 2.

- [ ] **Step 2: Run tests to verify the updated suite fails**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- calc.test 2>&1 | tail -15
```
Expected: FAIL — type error or runtime error on `disaggKvTransferFabricId` (the field isn't on `CalcInput` yet at top level).

- [ ] **Step 3: Move the fields in `types.ts`**

In `calc/src/engine/types.ts`, find `MultiDeviceConfig` (line 397) and delete the two disagg field lines and their comments:

```ts
export interface MultiDeviceConfig {
  system: MultiAcceleratorSystem
  parallelism: ParallelismMode['id'][]
  parallelismDegrees: Partial<Record<ParallelismMode['id'], number>>
}
```

Then find `CalcInput` (line 413) and add the two fields after `multiDevice?:`:

```ts
export interface CalcInput {
  accelerator: AcceleratorSpec
  acceleratorVariantId: string
  model: ModelArch
  quant: Quantization
  workload: Workload
  multiDevice?: MultiDeviceConfig
  // PD-disagg: prefill ships KV to decode over this fabric (InterconnectSpec.id).
  // Undefined = integrated serving (no transfer cost). Independent of multiDevice
  // — disagg is a deployment topology, not a property of one cluster's parallelism.
  disaggKvTransferFabricId?: string
  // When disagg is active, whether prefill emits the first decoded token locally
  // while KV transfer streams in parallel. Defaults true; setting false models the
  // worst-case sequential handoff.
  disaggFirstTokenOnPrefill?: boolean
}
```

- [ ] **Step 4: Update read sites in `calc.ts`**

In `calc/src/engine/calc.ts`, find lines 50-61 (the `kvTransferS` block and the `firstTokenOnPrefill` constant) and replace:

```ts
  // Disaggregated serving: KV cache ships from prefill cluster to decode
  // cluster over a separate fabric. Adds a one-shot transfer time to TTFT.
  // For integrated serving (single cluster) this is 0.
  let kvTransferS = 0
  if (input.disaggKvTransferFabricId) {
    const fab = INTERCONNECTS.find(i => i.id === input.disaggKvTransferFabricId)
    if (fab) {
      const bw = fab.perDirectionGBs ?? fab.perGpuBandwidthGBs / 2
      kvTransferS = memory.kvCachePerRequest / (bw * 1e9)
    }
  }
  // Production-standard: prefill node emits the first decoded token locally
  // while KV transfer streams in parallel. Defaults true when disagg is on.
  const firstTokenOnPrefill =
    input.disaggFirstTokenOnPrefill ?? true
```

(Two changes: `input.multiDevice?.disaggKvTransferFabricId` → `input.disaggKvTransferFabricId`; same for the `firstTokenOnPrefill` read.)

- [ ] **Step 5: Update `multiDevice` and `input` derived stores in `stores.ts`**

In `calc/src/ui/stores.ts`, find the `multiDevice` derived (line 56) and remove the disagg dependencies and spread:

```ts
export const multiDevice: Readable<MultiDeviceConfig | undefined> = derived(
  [systemId, modelId, parallelismOverride],
  ([$systemId, $modelId, $override]) => {
    if (!$systemId) return undefined
    const system = SYSTEMS.find(s => s.id === $systemId)
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

Then find the `input` derived (line 78) and add the disagg dependencies + top-level spread:

```ts
export const input: Readable<CalcInput | null> = derived(
  [acceleratorId, variantId, modelId, quant, workload, multiDevice, systemId,
   disaggKvTransferFabricId, disaggFirstTokenOnPrefill],
  ([$acceleratorId, $variantId, $modelId, $quant, $workload, $multiDevice, $systemId,
    $disagg, $firstTokenOnPrefill]) => {
    // When a system is selected, resolve accelerator from the system's chip ref.
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
      workload: $workload,
      ...($multiDevice && { multiDevice: $multiDevice }),
      ...($disagg && {
        disaggKvTransferFabricId: $disagg,
        disaggFirstTokenOnPrefill: $firstTokenOnPrefill,
      }),
    }
  }
)
```

(Three changes: add `disaggKvTransferFabricId` and `disaggFirstTokenOnPrefill` to the deps array; destructure them in the callback args; spread them into the return when `$disagg` is non-empty.)

- [ ] **Step 6: Run full suite + typecheck**

```bash
npm test 2>&1 | grep -E "(Tests |FAIL)" | tail -3
npm run check 2>&1 | tail -2
```
Expected: all tests green (including the two updated/new disagg tests); 0 type errors.

- [ ] **Step 7: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/engine/types.ts calc/src/engine/calc.ts calc/src/ui/stores.ts calc/test/engine/calc.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "refactor(calc): hoist disagg fields from MultiDeviceConfig to CalcInput

PD-disagg is a deployment topology, not a property of one cluster's
parallelism. Two single-chip nodes over RoCE-400 is a valid disagg
setup that the old schema couldn't express. Math unchanged."
```

(No Co-Authored-By footer — project convention.)

---

### Task 2: Interconnect catalog updates

**Why:** Spec calls for (a) a new optional field `compatibleAcceleratorIds` on `InterconnectSpec` populated for the 4 scale-up fabrics eligible as disagg "wires", and (b) 5 new high-end Ethernet entries (RoCEv2 200/400/800, Spectrum-X 400/800).

**Files:**
- Modify: `calc/src/engine/types.ts` (`InterconnectSpec` interface, line 61-105)
- Modify: `calc/src/data/interconnects.ts` (4 scale-up entries get the new field; 5 new entries appended)

No unit tests for the data add directly (the next task's `disaggFabrics.ts` helper will exercise this data).

- [ ] **Step 1: Add the `compatibleAcceleratorIds` field to `InterconnectSpec`**

In `calc/src/engine/types.ts`, find the `InterconnectSpec` interface (line 61). Insert a new optional field after `maxScaleUpGpus` (around line 79), before the existing `hopLatencyNs`:

```ts
  maxScaleUpGpus?: number      // size of the largest non-blocking domain

  // PD-disagg eligibility: which accelerator families can host this fabric as
  // the wire between prefill and decode clusters. Only populated for scale-up
  // fabrics whose use as a disagg medium implies a specific accelerator
  // family (e.g. NVL72 requires Blackwell). Scale-out fabrics leave this
  // undefined — any GPU can be on IB/EFA/RoCE.
  compatibleAcceleratorIds?: string[]

  // Round-trip latency for a single hop, ns. Optional; many vendors don't disclose.
  hopLatencyNs?: number
```

- [ ] **Step 2: Populate the field on the 4 eligible scale-up fabrics**

In `calc/src/data/interconnects.ts`, find each of these four entries by `id` and add the new field. The exact insertion point is right before the `sources` field of each (or before `notes` if there's no `sources`):

**`nvlink-4-nvl-256`** (DGX H100 SuperPOD):
```ts
    maxScaleUpGpus: 256,
    compatibleAcceleratorIds: ['h100', 'h200'],
    sources: ['nvidia-nvlink'],
```

**`nvlink-5-nvl72`** (NVL72):
```ts
    maxScaleUpGpus: 72,
    compatibleAcceleratorIds: ['gb200', 'b100', 'b200'],
    sources: ['nvidia-nvlink'],
```

**`tpu-ici-v5p`**:
```ts
    maxScaleUpGpus: 8960,
    compatibleAcceleratorIds: ['tpu-v5p'],
    sources: ['google-tpu-v5p-docs'],
```

**`tpu-ici-trillium`**:
```ts
    maxScaleUpGpus: 256,
    compatibleAcceleratorIds: ['tpu-trillium'],
    sources: ['google-tpu-v6e-docs'],
```

- [ ] **Step 3: Add the 5 new Ethernet entries**

In `calc/src/data/interconnects.ts`, find the closing `]` of the `INTERCONNECTS` array (around line 272, after `aws-efa-v3`). Insert these entries just before the closing bracket. Maintain the existing comment-section style (`// === ... ===`):

```ts
  // === RoCEv2 (Ethernet scale-out) ===
  {
    id: 'roce-200',
    name: 'RoCEv2 200 GbE',
    vendor: 'IBTA / Ethernet',
    generation: 'ConnectX-6 / 200 GbE',
    perGpuBandwidthGBs: 50,
    perDirectionGBs: 25,
    topology: 'fat-tree',
    scale: 'scale-out',
    notes: 'Same physical layer as IB-HDR (200 Gb/s SerDes); different transport (RoCEv2 lossy IP). Pairs with ConnectX-6.'
  },
  {
    id: 'roce-400',
    name: 'RoCEv2 400 GbE',
    vendor: 'IBTA / Ethernet',
    generation: 'ConnectX-7 / 400 GbE',
    perGpuBandwidthGBs: 100,
    perDirectionGBs: 50,
    topology: 'fat-tree',
    scale: 'scale-out',
    notes: 'ConnectX-7 generation. The modal Ethernet AI fabric in production today — most non-HPC NVIDIA deployments use RoCEv2 over Ethernet rather than IB.'
  },
  {
    id: 'roce-800',
    name: 'RoCEv2 800 GbE',
    vendor: 'IBTA / Ethernet',
    generation: 'ConnectX-8 / 800 GbE',
    perGpuBandwidthGBs: 200,
    perDirectionGBs: 100,
    topology: 'fat-tree',
    scale: 'scale-out',
    notes: 'ConnectX-8 generation. Matches IB-XDR per-port BW; same SerDes.'
  },

  // === NVIDIA Spectrum-X (lossless Ethernet for AI) ===
  {
    id: 'spectrum-x-400',
    name: 'NVIDIA Spectrum-X 400G',
    vendor: 'NVIDIA',
    generation: 'Spectrum-X SN5000 / 400 GbE',
    perGpuBandwidthGBs: 100,
    perDirectionGBs: 50,
    topology: 'fat-tree',
    scale: 'scale-out',
    notes: 'RoCEv2 + adaptive routing + Spectrum-X congestion control. Same peak BW as vanilla RoCE-400; differs in tail latency / contention behavior (not modeled by the engine today).'
  },
  {
    id: 'spectrum-x-800',
    name: 'NVIDIA Spectrum-X 800G',
    vendor: 'NVIDIA',
    generation: 'Spectrum-X SN5600 / 800 GbE',
    perGpuBandwidthGBs: 200,
    perDirectionGBs: 100,
    topology: 'fat-tree',
    scale: 'scale-out',
    notes: 'SN5600 + ConnectX-8. Blackwell-era pairing for AI clusters opting into Ethernet over IB.'
  }
```

- [ ] **Step 4: If Task 1 used `'ib-ndr'` as a placeholder in the new test, swap it now**

If the new test added in Task 1 Step 1 (`disagg works without a multiDevice config`) used `'ib-ndr'` as a placeholder, change the fabric to `'roce-400'` now:

```ts
    const inp = {
      ...testInput,
      disaggKvTransferFabricId: 'roce-400',
      disaggFirstTokenOnPrefill: false,
    }
```

If `'roce-400'` was used directly in Task 1, nothing to do here.

- [ ] **Step 5: Run typecheck + tests**

```bash
npm run check 2>&1 | tail -2
npm test 2>&1 | grep -E "Tests " | tail -1
```
Expected: 0 type errors; all tests green.

- [ ] **Step 6: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/engine/types.ts calc/src/data/interconnects.ts calc/test/engine/calc.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): add RoCE / Spectrum-X interconnect entries + compatibleAcceleratorIds field

5 new scale-out entries (RoCEv2 200/400/800, Spectrum-X 400/800). New
optional compatibleAcceleratorIds field on InterconnectSpec; populated
for the 4 scale-up fabrics eligible as PD-disagg media (NVL-256, NVL72,
TPU v5p, Trillium). The disagg picker will use this to filter scale-up
options to the user's selected accelerator family."
```

---

### Task 3: `share.ts` — unconditional disagg encoding

**Why:** The encoder currently gates `dk` / `df` emission on `state.systemId`. After Task 1's engine hoist, single-chip configs can also have disagg — the URL needs to encode it.

**Files:**
- Modify: `calc/src/ui/share.ts` (lines 72-77)
- Modify: `calc/test/ui/share.test.ts` or `calc/test/ui/share-route.test.ts` (add a single-chip+disagg roundtrip)

- [ ] **Step 1: Write the failing test**

In `calc/test/ui/share-route.test.ts`, add this new test at the bottom of the file (after the existing `describe` blocks):

```ts
describe('disagg URL encoding (single-chip + scale-out fabric)', () => {
  it('encodes dk/df when no system is selected', () => {
    const state = {
      acceleratorId: 'h100', variantId: 'sxm-80', systemId: '', modelId: 'llama-3.3-70b',
      quant: { weights: 'bf16', kv: 'fp16', activations: 'bf16' } as const,
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
      parallelismOverride: null,
      disaggKvTransferFabricId: 'roce-400',
      disaggFirstTokenOnPrefill: false,
    }
    const enc = encodeState(state)
    expect(enc).toContain('dk=roce-400')
    expect(enc).toContain('df=0')
  })

  it('round-trips single-chip + disagg state through decode', () => {
    const enc = 'a=h100&v=sxm-80&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=1&dk=roce-400'
    const decoded = decodeState(enc)
    expect(decoded.disaggKvTransferFabricId).toBe('roce-400')
    expect(decoded.disaggFirstTokenOnPrefill).toBe(true)   // omitted from URL → default true
  })
})
```

Note: the second test depends on `decodeState` parsing `dk`/`df` already. Check `share.ts` decode logic; it likely already does (the current decoder reads them unconditionally, only the encoder gates).

- [ ] **Step 2: Run tests to verify the first one fails**

```bash
npm test -- share-route 2>&1 | tail -10
```
Expected: FAIL on `encodes dk/df when no system is selected` — `dk` is missing from the encoded string because the gate suppresses it.

- [ ] **Step 3: Drop the `systemId` gate in `encodeState`**

In `calc/src/ui/share.ts`, find lines 72-77 (the `if (state.systemId && state.disaggKvTransferFabricId)` block):

```ts
  if (state.disaggKvTransferFabricId) {
    p.set('dk', state.disaggKvTransferFabricId)
    // `df=1` is the default — only emit when the user opted into the
    // worst-case sequential handoff.
    if (!state.disaggFirstTokenOnPrefill) p.set('df', '0')
  }
```

(One change: drop `state.systemId &&` from the condition.)

- [ ] **Step 4: Check the decoder for parity (probably no change)**

In `calc/src/ui/share.ts`, find the disagg decoding (search for `'dk'`). Confirm the decoder already reads `dk` unconditionally; if it has a `systemId`-style gate, drop that too. The current decode block looks like:

```ts
  if (params.has('dk')) {
    const dk = params.get('dk')!
    if (dk && INTERCONNECTS.find(i => i.id === dk)) {
      out.disaggKvTransferFabricId = dk
      out.disaggFirstTokenOnPrefill = params.get('df') !== '0'
    } else if (dk === '') {
      out.disaggKvTransferFabricId = ''
    }
  }
```

No system gate here; leave it as-is.

- [ ] **Step 5: Run tests to verify pass**

```bash
npm test -- share-route 2>&1 | tail -10
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
```
Expected: targeted tests green; full suite green; 0 type errors.

- [ ] **Step 6: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/share.ts calc/test/ui/share-route.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): encode dk/df unconditionally so single-chip + disagg round-trips"
```

---

### Task 4: `disaggFabrics.ts` pure helper + tests

**Why:** Both `InputPanel.svelte` (calc tab) and `DisaggInputPanel.svelte` (sim tab, created in Task 7) need the same logic for "which fabrics are eligible for the user's selected accelerator, and how should they be labeled and grouped." Factoring into a pure module makes it testable in node and reused by both panels.

**Files:**
- Create: `calc/src/ui/disaggFabrics.ts`
- Create: `calc/test/ui/disaggFabrics.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `calc/test/ui/disaggFabrics.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { groupedDisaggFabrics, formatFabricLabel } from '../../src/ui/disaggFabrics'

describe('groupedDisaggFabrics', () => {
  it('H100: scale-up shows NVL-256 (compatible), excludes NVL72 (Blackwell-only)', () => {
    const g = groupedDisaggFabrics('h100')
    const ids = g.scaleUp.map(f => f.id)
    expect(ids).toContain('nvlink-4-nvl-256')
    expect(ids).not.toContain('nvlink-5-nvl72')
  })

  it('GB200: scale-up shows NVL72 (compatible), excludes NVL-256 (Hopper-only)', () => {
    const g = groupedDisaggFabrics('gb200')
    const ids = g.scaleUp.map(f => f.id)
    expect(ids).toContain('nvlink-5-nvl72')
    expect(ids).not.toContain('nvlink-4-nvl-256')
  })

  it('non-NVIDIA / non-TPU accelerator: scale-up is empty', () => {
    const g = groupedDisaggFabrics('cerebras-wse3')
    expect(g.scaleUp).toEqual([])
  })

  it('scale-out is the same for any accelerator', () => {
    const h = groupedDisaggFabrics('h100').scaleOut.map(f => f.id)
    const mi = groupedDisaggFabrics('mi300x').scaleOut.map(f => f.id)
    expect(h).toEqual(mi)
    // Should include the new RoCE / Spectrum-X entries.
    expect(h).toContain('roce-400')
    expect(h).toContain('spectrum-x-800')
    expect(h).toContain('ib-ndr')
  })

  it('scale-out sorted by perGpuBandwidthGBs descending', () => {
    const g = groupedDisaggFabrics('h100')
    for (let i = 1; i < g.scaleOut.length; i++) {
      expect(g.scaleOut[i - 1].perGpuBandwidthGBs).toBeGreaterThanOrEqual(g.scaleOut[i].perGpuBandwidthGBs)
    }
  })

  it('does not include intra-node / die-to-die fabrics', () => {
    const g = groupedDisaggFabrics('h100')
    const allIds = [...g.scaleUp, ...g.scaleOut].map(f => f.id)
    expect(allIds).not.toContain('nvlink-4')   // intra-node HGX baseboard
    expect(allIds).not.toContain('pcie-gen5-x16')
    expect(allIds).not.toContain('ultrafusion')
  })
})

describe('formatFabricLabel', () => {
  it('appends " — N GB/s/GPU" to the name', () => {
    const fab = { name: 'RoCEv2 400 GbE', perGpuBandwidthGBs: 100 } as const
    expect(formatFabricLabel(fab as any)).toBe('RoCEv2 400 GbE — 100 GB/s/GPU')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
npm test -- disaggFabrics 2>&1 | tail -10
```
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create `disaggFabrics.ts`**

Create `calc/src/ui/disaggFabrics.ts`:

```ts
// Eligibility + labeling helpers for the PD-disagg fabric picker.
// Used by both the calc tab's InputPanel and the sim tab's DisaggInputPanel.
//
// Scope of "eligible" (per spec 2026-05-31-single-request-disagg-design.md):
//   • Scale-up fabrics — only those whose `compatibleAcceleratorIds` includes
//     the currently-selected accelerator. (Real disagg deployments inside a
//     single switch domain: NVL72 prefill+decode slices, TPU pods, etc.)
//   • Scale-out fabrics — always shown; any GPU can be on IB/EFA/RoCE.
//   • Intra-node and die-to-die fabrics — never shown; they're too small to
//     host meaningful PD-disagg.
//
// Within each group, sort by perGpuBandwidthGBs descending so the highest-BW
// option sits at the top.

import { INTERCONNECTS } from '../data/interconnects'
import type { InterconnectSpec } from '../engine/types'

export interface FabricGroups {
  scaleUp: InterconnectSpec[]
  scaleOut: InterconnectSpec[]
}

export function groupedDisaggFabrics(acceleratorId: string): FabricGroups {
  const byBwDesc = (a: InterconnectSpec, b: InterconnectSpec) =>
    b.perGpuBandwidthGBs - a.perGpuBandwidthGBs
  const scaleUp = INTERCONNECTS
    .filter(i => i.scale === 'scale-up' &&
                 (i.compatibleAcceleratorIds?.includes(acceleratorId) ?? false))
    .sort(byBwDesc)
  const scaleOut = INTERCONNECTS
    .filter(i => i.scale === 'scale-out')
    .sort(byBwDesc)
  return { scaleUp, scaleOut }
}

export function formatFabricLabel(f: InterconnectSpec): string {
  return `${f.name} — ${f.perGpuBandwidthGBs} GB/s/GPU`
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
npm test -- disaggFabrics 2>&1 | tail -15
npm run check 2>&1 | tail -2
```
Expected: all 7 cases green; 0 type errors.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/disaggFabrics.ts calc/test/ui/disaggFabrics.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): disaggFabrics helper (eligible-by-accelerator + label formatting)"
```

---

### Task 5: `stores.ts` — dual sim stores (monolithic vs disagg)

**Why:** The simulator renders TWO blocks (monolithic + disagg). Each needs its own `CalcResult`. The monolithic block must compute as if disagg were off (fields nulled); the disagg block keeps the user's current disagg state.

**Files:**
- Modify: `calc/src/ui/stores.ts` (the `simInput`/`simResult` block at lines 116-135)
- Create: `calc/test/ui/sim-disagg-stores.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `calc/test/ui/sim-disagg-stores.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  workload, disaggKvTransferFabricId, disaggFirstTokenOnPrefill,
  simInputMonolithic, simInputDisagg, simResultMonolithic, simResultDisagg
} from '../../src/ui/stores'

describe('simInputMonolithic / simInputDisagg', () => {
  beforeEach(() => {
    workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 64 })
    disaggKvTransferFabricId.set('roce-400')
    disaggFirstTokenOnPrefill.set(false)
  })

  it('both clamp workload.concurrency to 1', () => {
    expect(get(simInputMonolithic)!.workload.concurrency).toBe(1)
    expect(get(simInputDisagg)!.workload.concurrency).toBe(1)
  })

  it('simInputMonolithic clears disagg fields regardless of store state', () => {
    const inp = get(simInputMonolithic)!
    expect(inp.disaggKvTransferFabricId).toBeUndefined()
    expect(inp.disaggFirstTokenOnPrefill).toBeUndefined()
  })

  it('simInputDisagg preserves disagg fields from the store', () => {
    const inp = get(simInputDisagg)!
    expect(inp.disaggKvTransferFabricId).toBe('roce-400')
    expect(inp.disaggFirstTokenOnPrefill).toBe(false)
  })

  it('does not write back to the shared stores', () => {
    get(simInputMonolithic); get(simInputDisagg)
    expect(get(workload).concurrency).toBe(64)
    expect(get(disaggKvTransferFabricId)).toBe('roce-400')
  })
})

describe('simResultMonolithic / simResultDisagg', () => {
  beforeEach(() => {
    workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 1 })
    disaggKvTransferFabricId.set('roce-400')
    disaggFirstTokenOnPrefill.set(true)
  })

  it('monolithic result has zero kvTransferS even when a fabric is configured', () => {
    const r = get(simResultMonolithic)
    expect(r).not.toBeNull()
    for (const tier of Object.values(r!.perf)) {
      expect(tier.kvTransferS).toBe(0)
    }
  })

  it('disagg result has positive kvTransferS', () => {
    const r = get(simResultDisagg)
    expect(r).not.toBeNull()
    for (const tier of Object.values(r!.perf)) {
      expect(tier.kvTransferS).toBeGreaterThan(0)
    }
  })

  it('monolithic and disagg TTFT differ when disagg fabric is set', () => {
    const mono = get(simResultMonolithic)!
    const disagg = get(simResultDisagg)!
    const opId = Object.keys(mono.perf)[0]
    // firstTokenOnPrefill=true: disagg ttft = prefill + 1 decode step;
    // mono ttft = prefill. So disagg > mono by exactly tpot.
    expect(disagg.perf[opId].ttftS).toBeGreaterThan(mono.perf[opId].ttftS)
  })
})
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -- sim-disagg-stores 2>&1 | tail -10
```
Expected: FAIL — the new exports don't exist yet.

- [ ] **Step 3: Replace the existing sim-store block in `stores.ts`**

In `calc/src/ui/stores.ts`, find the existing `// --- Single-request simulator ---` block (line 116) and replace it entirely with:

```ts

// --- Single-request simulator ---
// The simulator tab renders two stacked configurations (monolithic + disagg)
// from the same shared inputs. Each block gets its own derived CalcInput +
// CalcResult; the monolithic side nulls the disagg fields, the disagg side
// passes them through. Concurrency is clamped to 1 in both (sim is by
// definition single-request, regardless of what the shared workload carries).
export const simInputMonolithic: Readable<CalcInput | null> = derived(input, $input => {
  if (!$input) return null
  return {
    ...$input,
    workload: { ...$input.workload, concurrency: 1 },
    disaggKvTransferFabricId: undefined,
    disaggFirstTokenOnPrefill: undefined,
  }
})

export const simInputDisagg: Readable<CalcInput | null> = derived(input, $input => {
  if (!$input) return null
  return { ...$input, workload: { ...$input.workload, concurrency: 1 } }
  // disagg fields flow through from $input as-is
})

interface SimComputed { result: CalcResult | null; error: string | null }
function safeCalc($input: CalcInput | null): SimComputed {
  if (!$input) return { result: null, error: null }
  try { return { result: calculate($input), error: null } }
  catch (err) { return { result: null, error: (err as Error).message } }
}

const simComputedMonolithic: Readable<SimComputed> = derived(simInputMonolithic, safeCalc)
const simComputedDisagg:     Readable<SimComputed> = derived(simInputDisagg,     safeCalc)

export const simResultMonolithic: Readable<CalcResult | null> = derived(simComputedMonolithic, $c => $c.result)
export const simResultDisagg:     Readable<CalcResult | null> = derived(simComputedDisagg,     $c => $c.result)
// Errors don't differ between the two variants (same hw/model/quant); surface
// monolithic's error as the canonical one.
export const simError: Readable<string | null> = derived(simComputedMonolithic, $c => $c.error)
```

This deletes the existing `simInput`, `simResult`, `simError` (and the local `simComputed`) and replaces them with the dual-store API.

- [ ] **Step 4: Update existing tests that reference the old API**

The previous task in the monolithic PR's task 7 created `calc/test/ui/sim-stores.test.ts` that imports `simInput`, `simResult`. Search for those imports and update:

```bash
grep -rn "simInput\|simResult\|simError" /Users/yao/workspace/llm-perf/calc/test /Users/yao/workspace/llm-perf/calc/src | grep -v node_modules
```

Update `calc/test/ui/sim-stores.test.ts`'s imports and assertions to use `simInputMonolithic` / `simResultMonolithic` (since those preserve the original semantics minus disagg). The existing test's behavior should still hold for monolithic, and `disaggKvTransferFabricId` should be reset to '' in `beforeEach` so the tests aren't sensitive to the new disagg-store state. If the old test asserted on `simInput` shape exactly, switch to `simInputMonolithic`.

Concrete: in `calc/test/ui/sim-stores.test.ts`, replace `import { workload, simInput, simResult }` with `import { workload, disaggKvTransferFabricId, simInputMonolithic, simResultMonolithic }`, and rename `simInput` → `simInputMonolithic` and `simResult` → `simResultMonolithic` in the test bodies. Add `disaggKvTransferFabricId.set('')` to the `beforeEach` to ensure a clean state.

- [ ] **Step 5: Update `Simulator.svelte` import to keep build green (deferred to Task 8, but typecheck needs it now)**

`Simulator.svelte` currently imports `simResult` and `simError`. After Step 3 those names no longer exist; svelte-check will error. The full rewrite of `Simulator.svelte` lands in Task 8 — for this task, just make the import non-failing.

Minimal patch in `calc/src/ui/Simulator.svelte`'s `<script lang="ts">` block — change the imports:

```ts
  import { simResultMonolithic as simResult, simError, workload, disaggFirstTokenOnPrefill } from './stores'
```

This is a deliberate alias to keep the file compiling without touching its body. Task 8 deletes this aliased import and rewrites the body to use both `simResultMonolithic` and `simResultDisagg` properly.

- [ ] **Step 6: Run targeted + full suite + typecheck**

```bash
npm test -- sim-disagg-stores 2>&1 | tail -10
npm test -- sim-stores 2>&1 | tail -10
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -2
```
Expected: all green; 0 type errors.

- [ ] **Step 7: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/stores.ts calc/src/ui/Simulator.svelte calc/test/ui/sim-stores.test.ts calc/test/ui/sim-disagg-stores.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): dual sim stores (simResultMonolithic / simResultDisagg)

Monolithic clears disagg fields before calling calculate(); disagg
preserves them. Both clamp concurrency to 1. Simulator.svelte gets a
temporary alias on simResultMonolithic to keep build green; Task 8
rewrites the component body."
```

---

### Task 6: `InputPanel.svelte` — `hideDisagg` prop + drop system gate + helper integration

**Why:** Calc tab keeps the disagg picker but it must now appear regardless of system selection (Task 1's hoist made disagg system-independent). Sim tab will hide the picker entirely via `hideDisagg={true}`. Both surfaces share the same filter+label logic via Task 4's helper.

**Files:**
- Modify: `calc/src/ui/InputPanel.svelte`

No new tests (presentational; verified in-browser per existing convention).

- [ ] **Step 1: Add the `hideDisagg` prop**

In `calc/src/ui/InputPanel.svelte`, find the existing `export let hideConcurrency = false` line (added in the prior PR). Add a sibling:

```ts
  export let hideConcurrency = false
  export let hideDisagg = false
```

- [ ] **Step 2: Replace the disagg fabric filter with the helper**

In the same script block, find `const disaggFabrics = INTERCONNECTS.filter(i => i.scale === 'scale-out')` (around line 13). Replace with imports + reactive helper call:

```ts
  import { groupedDisaggFabrics, formatFabricLabel } from './disaggFabrics'
```

Then in the script body (probably near `const DTYPES`), add:

```ts
  $: disaggGroups = groupedDisaggFabrics($acceleratorId)
```

Delete the now-unused `const disaggFabrics = ...` line and the `import { INTERCONNECTS } ...` if no other use remains. (Search for `INTERCONNECTS` references in this file before deleting.)

- [ ] **Step 3: Replace the disagg `<label>` block with the new markup**

In `calc/src/ui/InputPanel.svelte`, find the disagg block at line 131. Currently:

```svelte
      {#if $systemId}
        <label>
          Disagg KV transfer
          <select bind:value={$disaggKvTransferFabricId}>
            <option value="">— integrated —</option>
            {#each disaggFabrics as f}
              <option value={f.id}>{f.name}</option>
            {/each}
          </select>
        </label>
        {#if $disaggKvTransferFabricId}
          <label class="inline">
            <input type="checkbox" bind:checked={$disaggFirstTokenOnPrefill} />
            <span>1st token on prefill (hide transfer in TTFT)</span>
          </label>
        {/if}
      {/if}
```

Replace the entire block with:

```svelte
      {#if !hideDisagg}
        <label>
          Disagg KV transfer
          <select bind:value={$disaggKvTransferFabricId}>
            <option value="">— integrated —</option>
            {#if disaggGroups.scaleUp.length > 0}
              <optgroup label="Intra-domain (scale-up)">
                {#each disaggGroups.scaleUp as f}
                  <option value={f.id}>{formatFabricLabel(f)}</option>
                {/each}
              </optgroup>
            {/if}
            <optgroup label="Cross-rack (scale-out)">
              {#each disaggGroups.scaleOut as f}
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
        {/if}
      {/if}
```

Two changes: the outer gate is now `!hideDisagg` (was `$systemId`); the inner `<select>` uses optgroups + the helper for labels.

- [ ] **Step 4: Run typecheck + full suite**

```bash
npm run check 2>&1 | tail -2
npm test 2>&1 | grep -E "Tests " | tail -1
```
Expected: 0 type errors; tests green.

- [ ] **Step 5: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/InputPanel.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): InputPanel — hideDisagg prop + drop system gate + grouped fabric picker"
```

---

### Task 7: `DisaggInputPanel.svelte` (new component)

**Why:** Sim tab's disagg block owns its own inputs (per the spec). Future-proofs for asymmetric P/D (where this panel will grow separate prefill/decode hw selectors).

**Files:**
- Create: `calc/src/ui/DisaggInputPanel.svelte`

No tests (presentational; the filtering/labeling math is in `disaggFabrics.ts` and already tested).

- [ ] **Step 1: Create the component**

Create `calc/src/ui/DisaggInputPanel.svelte`:

```svelte
<script lang="ts">
  import { acceleratorId, disaggKvTransferFabricId, disaggFirstTokenOnPrefill } from './stores'
  import { groupedDisaggFabrics, formatFabricLabel } from './disaggFabrics'

  // V1: symmetric P=D — the disagg block inherits hw from the shared input
  // panel above. This component owns only the fabric and first-token toggle.
  // V2 (asymmetric P/D): this panel grows separate prefill / decode hw
  // selectors and per-side parallelism.
  $: groups = groupedDisaggFabrics($acceleratorId)
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
  {/if}
</div>

<style>
  .disagg-inputs {
    display: flex; flex-direction: row; flex-wrap: wrap;
    gap: 0.75rem; align-items: flex-end;
    padding: 0.6rem 0.9rem;
    background: #fafafa;
    border: 1px solid #e0e0e0; border-radius: 0.3rem;
    margin-bottom: 0.75rem;
  }
  label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.9rem; }
  label.inline { flex-direction: row; align-items: center; gap: 0.4rem; font-size: 0.85rem; }
  label.inline input[type=checkbox] { width: auto; }
  select { font-size: 1rem; padding: 0.25rem; min-width: 280px; }
</style>
```

- [ ] **Step 2: Run typecheck**

```bash
npm run check 2>&1 | tail -2
```
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/DisaggInputPanel.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): DisaggInputPanel — fabric picker + first-token toggle for the sim tab's disagg block"
```

---

### Task 8: `Simulator.svelte` — stacked monolithic + disagg blocks

**Why:** Render the disagg block below the existing monolithic block, gated by `$disaggKvTransferFabricId`. Pass `hideDisagg={true}` to the shared InputPanel so disagg controls live exclusively in the disagg block.

**Files:**
- Modify: `calc/src/ui/Simulator.svelte`

No new tests (presentational; the math is covered by Task 5's store tests + the existing gantt-geometry tests).

- [ ] **Step 1: Replace the entire file**

Replace the contents of `calc/src/ui/Simulator.svelte` with this new version. (Most of the structure is the same as the monolithic-only version that was just merged; the additions are: dual rows, dual blocks via a snippet, the DisaggInputPanel, `hideDisagg={true}` on the shared InputPanel.)

```svelte
<script lang="ts">
  import InputPanel from './InputPanel.svelte'
  import DisaggInputPanel from './DisaggInputPanel.svelte'
  import SimulatorGantt from './SimulatorGantt.svelte'
  import {
    simResultMonolithic, simResultDisagg, simError,
    workload, disaggFirstTokenOnPrefill, disaggKvTransferFabricId
  } from './stores'
  import type { GanttInput } from './simulatorGantt'
  import type { CalcResult } from '../engine/types'

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
  function rate(tps: number): string {
    if (tps >= 1e9) return `${sig3(tps / 1e9)} G tok/s`
    if (tps >= 1e6) return `${sig3(tps / 1e6)} M tok/s`
    if (tps >= 1e3) return `${sig3(tps / 1e3)} k tok/s`
    return `${sig3(tps)} tok/s`
  }

  // Memory fit is identical between monolithic and disagg (symmetric hw both
  // sides), so a single check applies to both blocks.
  $: memory = $simResultMonolithic?.memory
  $: fits = memory ? (memory.perRank?.fits ?? memory.fits) : false

  interface OpRow {
    id: string
    ttftS: number
    tpotS: number
    totalS: number
    inputTokenRate: number
    prefillRegime: 'compute' | 'memory' | 'comms'
    decodeRegime: 'compute' | 'memory' | 'comms'
    gantt: GanttInput
  }

  function rowsFrom(result: CalcResult | null, firstTokenOnPrefill: boolean, outputTokens: number): OpRow[] {
    if (!result) return []
    return Object.entries(result.perf).map(([id, t]): OpRow => ({
      id,
      ttftS: t.ttftS,
      tpotS: t.decode.timePerTokenS,
      totalS: t.ttftS + t.decode.timePerTokenS * (outputTokens - 1),
      inputTokenRate: t.inputTokenRate,
      prefillRegime: t.prefill.regime,
      decodeRegime: t.decode.regime,
      gantt: {
        prefillS: t.prefill.timeS,
        kvTransferS: t.kvTransferS,
        tpotS: t.decode.timePerTokenS,
        outputTokens,
        firstTokenOnPrefill,
        ttftS: t.ttftS,
        prefillRegime: t.prefill.regime,
        decodeRegime: t.decode.regime,
      },
    })).sort((a, b) => a.totalS - b.totalS)
  }

  $: rowsMonolithic = rowsFrom($simResultMonolithic, $disaggFirstTokenOnPrefill, $workload.outputTokens)
  $: rowsDisagg     = rowsFrom($simResultDisagg,     $disaggFirstTokenOnPrefill, $workload.outputTokens)
</script>

{#snippet resultBlock(rows: OpRow[])}
  <div class="kpis">
    <div class="kpi">
      <div class="label">TTFT</div>
      {#each rows as row, i}
        <div class="op" class:secondary={i > 0}>
          {#if rows.length > 1}<div class="op-name">{row.id}</div>{/if}
          <div class="value">{ms(row.ttftS)}</div>
          <div class="badge regime-{row.prefillRegime}">{row.prefillRegime}-bound prefill</div>
          <div class="caption">{rate(row.inputTokenRate)} input</div>
        </div>
      {/each}
    </div>
    <div class="kpi">
      <div class="label">TPOT</div>
      {#each rows as row, i}
        <div class="op" class:secondary={i > 0}>
          {#if rows.length > 1}<div class="op-name">{row.id}</div>{/if}
          <div class="value">{ms(row.tpotS)}</div>
          <div class="badge regime-{row.decodeRegime}">{row.decodeRegime}-bound decode</div>
          <div class="caption">{rate(1 / row.tpotS)} output</div>
        </div>
      {/each}
    </div>
    <div class="kpi">
      <div class="label">Total latency</div>
      {#each rows as row, i}
        <div class="op" class:secondary={i > 0}>
          {#if rows.length > 1}<div class="op-name">{row.id}</div>{/if}
          <div class="value">{ms(row.totalS)}</div>
          <div class="caption">{$workload.outputTokens} output tokens</div>
        </div>
      {/each}
    </div>
  </div>

  {#each rows as row}
    <div class="gantt-wrap">
      <h4>Timeline{rows.length > 1 ? ` (${row.id})` : ''}</h4>
      <SimulatorGantt input={row.gantt} />
    </div>
  {/each}
{/snippet}

<section class="simulator">
  <InputPanel hideConcurrency={true} hideDisagg={true} />

  {#if $simError}
    <div class="error">⚠ {$simError}</div>
  {:else if memory && !fits}
    <div class="oom">
      <strong>✗ Out of memory.</strong>
      Model + KV cache + activations exceed HBM capacity on the selected
      configuration. Pick a larger SKU, add parallelism (TP/PP), or trim the
      workload (prompt/output tokens). See the Calculator tab's Memory panel
      for the breakdown.
    </div>
  {:else if rowsMonolithic.length > 0}
    <h3 class="config-header">Single request, monolithic</h3>
    {@render resultBlock(rowsMonolithic)}

    {#if $disaggKvTransferFabricId && rowsDisagg.length > 0}
      <h3 class="config-header">Single request, PD-disagg</h3>
      <DisaggInputPanel />
      {@render resultBlock(rowsDisagg)}
    {:else if !$disaggKvTransferFabricId}
      <!-- Inline affordance for enabling disagg without going up to the
           shared inputs above. Lives in its own placeholder block. -->
      <div class="disagg-empty">
        <DisaggInputPanel />
        <p>Pick a KV transfer fabric above to add a PD-disagg comparison block.</p>
      </div>
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
  .oom {
    padding: 0.7rem 0.9rem;
    background: #fff7ec; color: #8a3f00;
    border: 1px solid #f0c890; border-radius: 0.3rem;
    font-size: 0.9rem; line-height: 1.4;
  }
  .oom strong { color: #b85b00; margin-right: 0.25rem; }
  .config-header {
    margin: 0.5rem 0 -0.25rem; font-size: 1rem; font-weight: 600; color: #333;
  }
  .disagg-empty p {
    margin: 0.25rem 0 0; font-size: 0.85rem; color: #666; font-style: italic;
  }
  .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; align-items: start; }
  .kpi {
    border: 1px solid #d4d4d4; border-radius: 0.4rem; padding: 0.8rem 1rem;
    background: #fff;
  }
  .kpi .label { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.02em; color: #888; }
  .op { padding-top: 0.2rem; }
  .op.secondary {
    margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px solid #eee;
  }
  .op-name {
    font-size: 0.85rem; font-style: italic; color: #555;
    margin-bottom: 0.1rem;
  }
  .op .value { font-size: 1.5rem; font-weight: 700; line-height: 1.1; margin-top: 0.1rem; }
  .op .badge {
    display: inline-block; margin-top: 0.35rem; padding: 0.1rem 0.45rem;
    font-size: 0.75rem; border-radius: 0.2rem; color: #fff;
  }
  .badge.regime-compute { background: #c05621; }
  .badge.regime-memory  { background: #2b6cb0; }
  .badge.regime-comms   { background: #6b46c1; }
  .op .caption { font-size: 0.78rem; color: #666; margin-top: 0.3rem; }
  .gantt-wrap h4 { margin: 0 0 0.4rem; font-size: 0.85rem; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  @media (max-width: 640px) {
    .kpis { grid-template-columns: 1fr; }
  }
</style>
```

Two notable choices in this layout:
- When `$disaggKvTransferFabricId` is empty, render a small empty-state block (`.disagg-empty`) that still hosts the `DisaggInputPanel` so the user has an inline affordance to enable disagg without scrolling up to the shared inputs.
- The `resultBlock` snippet is reused for both monolithic and disagg blocks; the only difference is the surrounding header + (for disagg) the `<DisaggInputPanel />`.

- [ ] **Step 2: Run typecheck + full suite + build**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check 2>&1 | tail -2
npm test 2>&1 | grep -E "Tests " | tail -1
npm run build 2>&1 | tail -5
```
Expected: 0 type errors; all tests green; clean build.

- [ ] **Step 3: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/Simulator.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): Simulator renders stacked monolithic + disagg blocks

Reads simResultMonolithic and simResultDisagg as separate stores. Disagg
block appears when a fabric is selected; otherwise a small empty-state
panel offers an inline affordance to enable disagg. Inputs panel above
uses hideDisagg=true so disagg controls live only in the disagg block."
```

---

### Task 9: Smoke verification

**Files:** none (verification only).

- [ ] **Step 1: Final integration test pass**

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

- [ ] **Step 3: Interactive checks (controller's job — open the dev URL)**

- Sim tab loads with monolithic block + an empty disagg placeholder containing the DisaggInputPanel.
- Pick a fabric from DisaggInputPanel → the empty placeholder becomes a full disagg block with KPI cards + gantt. Monolithic block's KPI strip is unchanged.
- Toggle `1st token on prefill` → disagg gantt reshapes (overlap layout vs. sequential 3-segment). TTFT updates.
- Switch to a non-Hopper / non-Blackwell accelerator (e.g. MI300X) → scale-up fabric options disappear from the picker; scale-out options remain.
- Pick a single-chip H100 (no system) and a RoCE-400 fabric → disagg block renders with no error (the Task 1 hoist enables this).
- URL update: a sim-tab URL with disagg state should encode `dk` / `df`. Copy it, paste in a new tab → restores on the sim tab with the disagg block visible.
- Calc tab still shows the disagg picker, now available regardless of system selection.

No commit step here — pure verification. If issues surface, fix and commit in a focused follow-up commit.

---

## Self-Review

**1. Spec coverage:**
- Spec §Engine refactor (hoist disagg fields) → Task 1. ✓
- Spec §Catalog additions (5 RoCE/Spectrum-X) → Task 2. ✓
- Spec §`compatibleAcceleratorIds` field + populate 4 scale-up → Task 2. ✓
- Spec §InputPanel changes (`hideDisagg` prop, drop system gate, optgroup + label) → Task 6. ✓
- Spec §`DisaggInputPanel.svelte` → Task 7. ✓
- Spec §`disaggFabrics.ts` helper → Task 4. ✓
- Spec §Simulator UI (stacked blocks, snippet reuse, empty-state) → Task 8. ✓
- Spec §share.ts unconditional encoding → Task 3. ✓
- Spec §stores.ts dual sim stores → Task 5. ✓
- Spec §Testing (engine, helper, share, gantt geometry already covered) → tests embedded in tasks 1, 3, 4, 5. ✓
- Spec §Non-goals — none implemented as features. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases" patterns. Every code step has full code or exact transform. Task 9 has interactive checks but no code (verification-only by design). Pass.

**3. Type consistency:**
- `CalcInput.disaggKvTransferFabricId: string | undefined` (Task 1) — consumed everywhere as optional. ✓
- `InterconnectSpec.compatibleAcceleratorIds: string[] | undefined` (Task 2) — read in `groupedDisaggFabrics` (Task 4) as `compatibleAcceleratorIds?.includes(...)`. ✓
- `FabricGroups { scaleUp: InterconnectSpec[]; scaleOut: InterconnectSpec[] }` (Task 4) — consumed in `InputPanel.svelte` (Task 6) and `DisaggInputPanel.svelte` (Task 7) as `disaggGroups.scaleUp` / `disaggGroups.scaleOut`. ✓
- `formatFabricLabel(f)` (Task 4) — consumed in both panels. ✓
- `simResultMonolithic` / `simResultDisagg` (Task 5) — consumed in `Simulator.svelte` (Task 8) with exact names. ✓
- `OpRow` interface (Task 8) — consistent shape across both calls to `rowsFrom`. ✓
- `rowsFrom(result, firstTokenOnPrefill, outputTokens)` signature consistent in both invocations (Task 8). ✓
- `hideDisagg: boolean` prop on InputPanel (Task 6) — set to `true` in Simulator.svelte (Task 8). Calc tab leaves it default `false`. ✓
- Snippet `resultBlock(rows: OpRow[])` (Task 8) — invoked twice with the matching parameter types. ✓

**4. Known-deferred items (not blockers, documented in spec):**
- Asymmetric P/D inputs in `DisaggInputPanel` — TODO; v2 grows the component.
- Achievable-vs-theoretical fabric model — TODO; engine fidelity follow-up.
- Ultra Ethernet / EFAv4 catalog entries — TODO; data refresh follow-up.

import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  workload, disaggKvTransferFabricId, disaggFirstTokenOnPrefill,
  simInputMonolithic, simInputDisagg, simResultMonolithic, simResultDisagg,
  decodeAcceleratorId, decodeVariantId, decodeSystemId,
  decodeParallelismOverride, heterogeneous
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

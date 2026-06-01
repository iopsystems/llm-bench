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

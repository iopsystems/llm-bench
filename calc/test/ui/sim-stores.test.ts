import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { workload, disaggKvTransferFabricId, simInputMonolithic, simResultMonolithic } from '../../src/ui/stores'

describe('simInputMonolithic', () => {
  beforeEach(() => {
    // Reset workload so concurrency=64 (the value we deliberately set below)
    // is observable as a deviation from the default.
    workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 64 })
    disaggKvTransferFabricId.set('')   // ensure clean monolithic state
  })

  it('clamps concurrency to 1 regardless of the shared workload store', () => {
    const inp = get(simInputMonolithic)
    expect(inp).not.toBeNull()
    expect(inp!.workload.concurrency).toBe(1)
    // Other workload fields pass through.
    expect(inp!.workload.promptTokens).toBe(2048)
    expect(inp!.workload.outputTokens).toBe(512)
  })

  it('does not write back to the workload store', () => {
    get(simInputMonolithic)   // force evaluation
    expect(get(workload).concurrency).toBe(64)
  })
})

describe('simResultMonolithic', () => {
  beforeEach(() => {
    workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 64 })
    disaggKvTransferFabricId.set('')   // ensure clean monolithic state
  })

  it('produces a CalcResult with at least one operating point', () => {
    const r = get(simResultMonolithic)
    expect(r).not.toBeNull()
    expect(Object.keys(r!.perf).length).toBeGreaterThan(0)
  })

  it('every op-point exposes ttftS, decode.timePerTokenS, and kvTransferS', () => {
    const r = get(simResultMonolithic)
    for (const tier of Object.values(r!.perf)) {
      expect(typeof tier.ttftS).toBe('number')
      expect(typeof tier.decode.timePerTokenS).toBe('number')
      expect(typeof tier.kvTransferS).toBe('number')
    }
  })
})

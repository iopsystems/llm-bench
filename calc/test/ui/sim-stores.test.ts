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

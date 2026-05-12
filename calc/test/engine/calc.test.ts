import { describe, it, expect } from 'vitest'
import { calculate } from '../../src/engine/calc'
import { testInput } from '../fixtures'

describe('calculate', () => {
  it('returns memory matching computeMemory', () => {
    const r = calculate(testInput)
    expect(r.memory.weights).toBe(2000)
    expect(r.memory.kvCachePerRequest).toBe(240)
    expect(r.memory.kvCacheTotal).toBe(480)
    expect(r.memory.activationsPeak).toBe(960)
    expect(r.memory.total).toBe(3440)
    expect(r.memory.fits).toBe(true)
  })

  it('produces one perf tier per operating point', () => {
    const r = calculate(testInput)
    expect(Object.keys(r.perf)).toEqual(['peak'])
  })

  it('perf.peak has all the expected fields', () => {
    const r = calculate(testInput)
    const p = r.perf['peak']
    expect(p.prefill.flops).toBe(21600)
    expect(p.decode.flopsPerStep).toBe(4400)
    expect(p.ttftS).toBe(p.prefill.timeS)
    expect(p.outputTokenRate).toBeCloseTo(p.decode.aggregateTokensPerS, 9)
    expect(p.inputTokenRate).toBeCloseTo(testInput.workload.promptTokens / p.prefill.timeS, 6)
  })

  it('derivation is non-empty and ends with the final memory total', () => {
    const r = calculate(testInput)
    expect(r.derivation.length).toBeGreaterThan(0)
    const memoryTotalStep = r.derivation.find(s => s.label === 'memory total')
    expect(memoryTotalStep?.value).toBe(r.memory.total)
  })

  it('throws on unknown variant id', () => {
    expect(() => calculate({ ...testInput, gpuVariantId: 'nope' })).toThrow()
  })
})

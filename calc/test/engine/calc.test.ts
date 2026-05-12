import { describe, it, expect } from 'vitest'
import { calculate } from '../../src/engine/calc'
import { testInput } from '../fixtures'
import { GPUS } from '../../src/data/gpus'
import { MODELS } from '../../src/data/models'
import type { CalcInput } from '../../src/engine/types'

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

describe('calculate — real data integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const llama70b = MODELS.find(m => m.id === 'llama-3.3-70b')!

  const input: CalcInput = {
    gpu: h100,
    gpuVariantId: 'sxm-80',
    model: llama70b,
    quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
  }

  it('Llama 3.3 70B on H100 SXM-80: weights are 141 GB (does not fit single-GPU)', () => {
    const r = calculate(input)
    // 70.55B params × 2 bytes = 141.1 GB
    expect(r.memory.weights / 1e9).toBeCloseTo(141.1, 0)
    expect(r.memory.fits).toBe(false)
  })

  it('Llama 3.3 70B prefill regime is compute-bound for batch=1, prompt=2048', () => {
    const r = calculate(input)
    // Long prefill on dense 70B model — compute term dominates
    expect(r.perf['peak'].prefill.regime).toBe('compute')
  })

  it('Llama 3.3 70B decode at batch=1 is memory-bound', () => {
    const r = calculate(input)
    // Classic single-stream decode: weight-load bandwidth dominates
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

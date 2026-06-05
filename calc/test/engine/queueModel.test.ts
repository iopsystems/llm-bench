import { describe, it, expect } from 'vitest'
import { computeNMax, loadCurve } from '../../src/engine/queueModel'
import { calculate } from '../../src/engine'
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

// Hardware rationale shared by these tests: Llama-3.3-70B at bf16 needs
// ~140 GB for weights, so a single H100 SXM-80 can't fit it (boundBy=weights),
// while an H200 SXM-141 has the headroom to fit weights + KV (boundBy=kv).
describe('computeNMax', () => {
  it('returns a positive integer for a model that fits with headroom', () => {
    const r = computeNMax(inputFor('h200', 'sxm-141', 'llama-3.3-70b'))
    expect(r.boundBy).toBe('kv')
    expect(r.nMax).toBeGreaterThan(0)
    expect(Number.isInteger(r.nMax)).toBe(true)
  })

  it('returns {nMax: 0, boundBy: weights} when weights alone exceed HBM', () => {
    const r = computeNMax(inputFor('h100', 'sxm-80', 'llama-3.3-70b'))
    expect(r.boundBy).toBe('weights')
    expect(r.nMax).toBe(0)
  })
})

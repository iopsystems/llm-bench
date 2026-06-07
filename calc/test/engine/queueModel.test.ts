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

  it('totalS matches engine overlap-mode formula when disagg fabric is set', () => {
    const input = {
      ...inputFor('h200', 'sxm-141', 'llama-3.3-70b'),
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: true,
    }
    const [point] = loadCurve(input, [1])
    // Overlap: totalS = prefill + outputTokens × tpot + max(0, kvTransfer - tpot)
    const stutter = Math.max(0, point.kvTransferS - point.tpotS)
    expect(point.totalS).toBeCloseTo(point.prefillS + 512 * point.tpotS + stutter, 12)
  })

  it('totalS matches sequential formula when firstTokenOnPrefill is false', () => {
    const input = {
      ...inputFor('h200', 'sxm-141', 'llama-3.3-70b'),
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: false,
    }
    const [point] = loadCurve(input, [1])
    expect(point.totalS).toBeCloseTo(point.prefillS + point.kvTransferS + 512 * point.tpotS, 12)
  })

  it('per-device throughput fields scale correctly with device count', () => {
    // Single-chip (count=1): per-device = aggregate.
    const single = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const [pSingle] = loadCurve(single, [8])
    expect(pSingle.prefillDevices).toBe(1)
    expect(pSingle.decodeDevices).toBe(1)
    expect(pSingle.prefillInputTokPerSPerDevice).toBeCloseTo(2048 / pSingle.prefillS, 6)
    expect(pSingle.decodeOutputTokPerSPerDevice).toBeCloseTo(8 / pSingle.tpotS, 6)
  })

  it('per-device output tok/s × decodeDevices recovers aggregate decode rate', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const [p] = loadCurve(input, [4])
    const aggregateDecode = p.decodeOutputTokPerSPerDevice * p.decodeDevices
    expect(aggregateDecode).toBeCloseTo(4 / p.tpotS, 6)
  })

  it('ttftS in overlap mode matches engine: prefillS + firstStepOnPrefillS (single-cluster)', () => {
    const input = {
      ...inputFor('h200', 'sxm-141', 'llama-3.3-70b'),
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: true,
    }
    // Symmetric cluster (no decode override): firstStepOnPrefillS equals the
    // decode-step time on the prefill cluster at batch=1, computed via the same
    // engine path calc.ts uses.
    const [point] = loadCurve(input, [1])
    expect(point.ttftS).toBeCloseTo(point.prefillS + point.tpotS, 12)
    // Sanity: ttftS < totalS
    expect(point.ttftS).toBeLessThan(point.totalS)
  })

  it('ttftS in sequential mode includes full kvTransferS', () => {
    const input = {
      ...inputFor('h200', 'sxm-141', 'llama-3.3-70b'),
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: false,
    }
    const [point] = loadCurve(input, [4])
    expect(point.ttftS).toBeCloseTo(point.prefillS + point.kvTransferS, 12)
  })

  it('ttftS for no-fabric case is just prefillS', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    // No disagg fabric → kvTransferS = 0 → ttftS = prefillS (no transfer, no overlap)
    const [point] = loadCurve(input, [1])
    expect(point.kvTransferS).toBe(0)
    expect(point.ttftS).toBeCloseTo(point.prefillS, 12)
  })

  it('ttftS is independent of N (TTFT is per-request, not batched)', () => {
    const input = {
      ...inputFor('h200', 'sxm-141', 'llama-3.3-70b'),
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: true,
    }
    const points = loadCurve(input, [1, 4, 16])
    // All three points should have the same ttftS — the just-arrived request's
    // first-token latency doesn't depend on what size batch the decode cluster
    // is running concurrently.
    expect(points[1].ttftS).toBeCloseTo(points[0].ttftS, 12)
    expect(points[2].ttftS).toBeCloseTo(points[0].ttftS, 12)
  })

  it('ttftMode classifies overlap / sequential / no-fabric correctly', () => {
    const base = inputFor('h200', 'sxm-141', 'llama-3.3-70b')

    // No fabric → no-fabric mode
    expect(loadCurve(base, [1])[0].ttftMode).toBe('no-fabric')

    // Fabric + overlap (default) → overlap mode
    const overlap = { ...base, disaggKvTransferFabricId: 'ib-ndr', disaggFirstTokenOnPrefill: true }
    expect(loadCurve(overlap, [1])[0].ttftMode).toBe('overlap')

    // Fabric + no overlap → sequential mode
    const sequential = { ...base, disaggKvTransferFabricId: 'ib-ndr', disaggFirstTokenOnPrefill: false }
    expect(loadCurve(sequential, [1])[0].ttftMode).toBe('sequential')
  })

  it('inputTokPerS = throughputReqS × promptTokens (aggregate input rate)', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const [point] = loadCurve(input, [8])
    expect(point.inputTokPerS).toBeCloseTo(point.throughputReqS * 2048, 6)
  })

  it('latencyS equals totalS in v1 deterministic model', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const [point] = loadCurve(input, [4])
    expect(point.latencyS).toBe(point.totalS)
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

  it('prefill side returns smaller nMax than decode side (prefill activations are larger)', () => {
    const input = inputFor('h200', 'sxm-141', 'llama-3.3-70b')
    const decodeSide = computeNMax(input)             // default 'decode'
    const prefillSide = computeNMax(input, 'prefill')
    expect(prefillSide.nMax).toBeLessThan(decodeSide.nMax)
    expect(prefillSide.boundBy).toBe('kv')
  })

  it('prefill side honors prompt-token sensitivity', () => {
    const small = { ...inputFor('h200', 'sxm-141', 'llama-3.3-70b'),
                    workload: { promptTokens: 512, outputTokens: 512, concurrency: 1 } }
    const big   = { ...inputFor('h200', 'sxm-141', 'llama-3.3-70b'),
                    workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 } }
    // Larger prompt → larger prefill activations → smaller nMax.
    expect(computeNMax(big, 'prefill').nMax).toBeLessThan(computeNMax(small, 'prefill').nMax)
  })

  it('weights-bound case returns 0 for both phases', () => {
    const input = inputFor('h100', 'sxm-80', 'llama-3.3-70b')  // 80 GB chip, 140 GB weights
    expect(computeNMax(input).nMax).toBe(0)
    expect(computeNMax(input, 'prefill').nMax).toBe(0)
  })
})

import { describe, it, expect } from 'vitest'
import { computePrefill } from '../../src/engine/prefill'
import { testInput } from '../fixtures'
import { computeMemory } from '../../src/engine/memory'

describe('computePrefill', () => {
  const opPoint = testInput.gpu.variants[0].operatingPoints[0]
  const memory = computeMemory(testInput)

  it('flops = 2 × params × prompt + 2 × layers × prompt² × hidden', () => {
    // 2 × 1000 × 10 + 2 × 2 × 100 × 4 = 20000 + 1600 = 21600
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.flops).toBe(21600)
  })

  it('bytes = weightBytes + activationsPeak', () => {
    // weights=2000, activations=960 → 2960
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.bytes).toBe(2960)
  })

  it('timeS = max(flops/tflops, bytes/bw)', () => {
    // flops/tflops = 21600 / 1e12 = 2.16e-8
    // bytes/bw    = 2960 / 1e9    = 2.96e-6  ← bigger
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.timeS).toBeCloseTo(2960 / 1e9, 12)
    expect(p.regime).toBe('memory')
  })

  it('uses activation dtype to pick tflops', () => {
    // testInput uses fp16; opPoint.tflops.fp16 = 1
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.timeS).toBeGreaterThan(0)
  })

  it('attention term caps at window for sliding attention', () => {
    // testModel: layers=2, hiddenDim=4. Prompt=10. With window=5:
    // attention term = 2 × 2 × 10 × min(10, 5) × 4 = 800 (vs full's 1600)
    // MLP term = 2 × 1000 × 10 = 20000 (unchanged)
    // total = 20800
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 5 }
    }
    const input = { ...testInput, model: slidingModel }
    const slidingMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, slidingMemory)
    expect(p.flops).toBe(20800)
  })
})

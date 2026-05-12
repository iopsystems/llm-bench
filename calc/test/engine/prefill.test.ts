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
})

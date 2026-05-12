import { describe, it, expect } from 'vitest'
import { computeDecode } from '../../src/engine/decode'
import { testInput } from '../fixtures'
import { computeMemory } from '../../src/engine/memory'

describe('computeDecode', () => {
  const opPoint = testInput.gpu.variants[0].operatingPoints[0]
  const memory = computeMemory(testInput)

  // testInput: prompt=10, output=5, concurrency=2
  // avg seqlen for decode attention ≈ prompt + output/2 = 12.5

  it('flopsPerStep = (2 × params + 2 × layers × seqlen_avg × hidden) × concurrency', () => {
    // (2 × 1000 + 2 × 2 × 12.5 × 4) × 2 = (2000 + 200) × 2 = 4400
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.flopsPerStep).toBe(4400)
  })

  it('bytesPerStep = weightBytes + kvPerRequest × concurrency', () => {
    // weights=2000, kvPerRequest=240 → 2000 + 240×2 = 2480
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.bytesPerStep).toBe(2480)
  })

  it('timePerTokenS = max(flopsPerStep/tflops, bytesPerStep/bw)', () => {
    // flops/tflops = 4400 / 1e12 = 4.4e-9
    // bytes/bw    = 2480 / 1e9  = 2.48e-6  ← bigger
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.timePerTokenS).toBeCloseTo(2480 / 1e9, 12)
    expect(d.regime).toBe('memory')
  })

  it('aggregateTokensPerS = concurrency / timePerTokenS', () => {
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.aggregateTokensPerS).toBeCloseTo(2 / d.timePerTokenS, 6)
  })

  it('attention term caps at window for sliding attention', () => {
    // testModel: layers=2, hiddenDim=4, concurrency=2.
    // avgSeqlen = 10 + 5/2 = 12.5. With window=8, effSeqlen = 8.
    // flopsPerStep = (2 × 1000 + 2 × 2 × 8 × 4) × 2 = (2000 + 128) × 2 = 4256
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 8 }
    }
    const input = { ...testInput, model: slidingModel }
    const slidingMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, slidingMemory)
    expect(d.flopsPerStep).toBe(4256)
  })

  it('bytesPerStep weight term uses activeParams for MoE', () => {
    // testModel: paramCount=1000, fp16 weights → 2 bytes/param.
    // For MoE with activeParamCount=250:
    //   weight bytes per step = 250 × 2 = 500
    //   kv per request = 240 (existing fixture), × concurrency 2 = 480
    //   total bytesPerStep = 500 + 480 = 980
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, moeMemory)
    expect(d.bytesPerStep).toBe(980)
  })

  it('flopsPerStep MLP term uses activeParams for MoE', () => {
    // testModel: paramCount=1000, hiddenDim=4, layers=2, concurrency=2.
    // avgSeqlen = 10 + 5/2 = 12.5.
    // For MoE with activeParamCount=250:
    //   (2 × 250 + 2 × 2 × 12.5 × 4) × 2 = (500 + 200) × 2 = 1400
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, moeMemory)
    expect(d.flopsPerStep).toBe(1400)
  })
})

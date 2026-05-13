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

  it('FLOPs MLP term uses activeParams for MoE', () => {
    // testModel: paramCount=1000, hiddenDim=4, layers=2, prompt=10.
    // For MoE with activeParamCount=250:
    //   MLP: 2 × 250 × 10 = 5000
    //   Attention: 2 × 2 × 10 × 10 × 4 = 1600 (full attention, unchanged)
    //   Total = 6600
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        numSharedExperts: 0,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, moeMemory)
    expect(p.flops).toBe(6600)
  })

  it('attention term uses attentionDim for MLA (kv_lora + rope, not hidden)', () => {
    // testModel: layers=2, hiddenDim=4, prompt=10, paramCount=1000.
    // numHeads=2, headDim=2 → 2×2=4 matches hiddenDim, so non-MLA path unaffected.
    // MLA with kvLoraRank=10, rope=2 → attentionDim = 12.
    // MLP: 2 × 1000 × 10 = 20000
    // Attention: 2 × 2 × 10 × 10 × 12 = 4800 (full attention, no sliding bound)
    // Total = 24800
    const mlaModel = {
      ...testInput.model,
      attention: { type: 'mla' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2 }
    }
    const input = { ...testInput, model: mlaModel }
    const mlaMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, mlaMemory)
    expect(p.flops).toBe(24800)
  })

  it('attention term uses hybrid formula in prefill flops', () => {
    // testModel: layers=2, hiddenDim=4, paramCount=1000, prompt=10.
    // numHeads=2, headDim=2 → attentionDim=4.
    // Hybrid with slidingWindow=5, numSlidingLayers=1, numGlobalLayers=1:
    //   attendedSeq(10) = 1 × min(10, 5) + 1 × 10 = 15
    //   MLP: 2 × 1000 × 10 = 20000
    //   Attention: 2 × 10 × 15 × 4 = 1200
    //   Total = 21200
    const hybridModel = {
      ...testInput.model,
      attention: {
        type: 'hybrid' as const,
        slidingWindow: 5,
        numSlidingLayers: 1,
        numGlobalLayers: 1
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, hybridMemory)
    expect(p.flops).toBe(21200)
  })

  it('attention term caps at topK for mla-dsa', () => {
    // testModel: layers=2, paramCount=1000, prompt=10.
    // MLA-DSA with kvLoraRank=10, rope=2 → attentionDim=12; topK=3 (< prompt=10).
    //   attendedSeq = 2 × min(10, 3) = 6
    //   MLP: 2 × 1000 × 10 = 20000
    //   Attention: 2 × 10 × 6 × 12 = 1440
    //   Total = 21440
    const dsaModel = {
      ...testInput.model,
      attention: { type: 'mla-dsa' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2, topK: 3 }
    }
    const input = { ...testInput, model: dsaModel }
    const dsaMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, dsaMemory)
    expect(p.flops).toBe(21440)
  })
})

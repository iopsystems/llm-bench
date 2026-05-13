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
        numSharedExperts: 0,
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
        numSharedExperts: 0,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, moeMemory)
    expect(d.flopsPerStep).toBe(1400)
  })

  it('attention term uses attentionDim for MLA (kv_lora + rope, not hidden)', () => {
    // testModel: layers=2, hiddenDim=4, paramCount=1000, concurrency=2.
    // numHeads=2, headDim=2 → 2×2=4 matches hiddenDim (non-MLA path unaffected).
    // avgSeqlen = 10 + 5/2 = 12.5.
    // MLA with kvLoraRank=10, rope=2 → attentionDim = 12.
    // flopsPerStep = (2×1000 + 2×2×12.5×12) × 2 = (2000 + 600) × 2 = 5200
    const mlaModel = {
      ...testInput.model,
      attention: { type: 'mla' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2 }
    }
    const input = { ...testInput, model: mlaModel }
    const mlaMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, mlaMemory)
    expect(d.flopsPerStep).toBe(5200)
  })

  it('attention term uses hybrid formula in decode flopsPerStep', () => {
    // testModel: layers=2, hiddenDim=4, paramCount=1000, concurrency=2.
    // numHeads=2, headDim=2 → attentionDim=4.
    // avgSeqlen = 10 + 5/2 = 12.5.
    // Hybrid with slidingWindow=5, numSlidingLayers=1, numGlobalLayers=1:
    //   attendedSeq(12.5) = 1 × min(12.5, 5) + 1 × 12.5 = 17.5
    //   flopsPerStep = (2×1000 + 2×17.5×4) × 2 = (2000 + 140) × 2 = 4280
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
    const d = computeDecode(input, opPoint, hybridMemory)
    expect(d.flopsPerStep).toBe(4280)
  })

  it('attention term caps at topK for mla-dsa', () => {
    // testModel: layers=2, paramCount=1000, concurrency=2.
    // avgSeqlen = 10 + 5/2 = 12.5.
    // MLA-DSA with kvLoraRank=10, rope=2 → attentionDim=12; topK=4 (< 12.5).
    //   attendedSeq = 2 × min(12.5, 4) = 8
    //   flopsPerStep = (2×1000 + 2×8×12) × 2 = (2000 + 192) × 2 = 4384
    const dsaModel = {
      ...testInput.model,
      attention: { type: 'mla-dsa' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2, topK: 4 }
    }
    const input = { ...testInput, model: dsaModel }
    const dsaMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, dsaMemory)
    expect(d.flopsPerStep).toBe(4384)
  })

  it('flopsPerStep and bytesPerStep for linear-mla-hybrid include KDA terms', () => {
    // testModel: layers=2, paramCount=1000, concurrency=2.
    // avgSeqlen = 10 + 5/2 = 12.5.
    // linear-mla-hybrid (kvLoraRank=5, rope=1, numLinear=1, numFull=1,
    //                    numLinearHeads=2, linearHeadDim=2):
    //   attentionDim = 6
    //   attendedSeqlen(12.5) = 1 × 12.5 = 12.5  (only the 1 full layer)
    //   KDA per-token FLOPs = 2 × 1 × 2 × 2² = 16
    //   flopsPerStep = (2 × 1000 + 2 × 12.5 × 6 + 16) × 2 = (2000 + 150 + 16) × 2 = 4332
    //   memory.kvCachePerRequest (from prompt+output=15) = 12 × 15 + 16 = 196
    //   KDA state bytes = 16
    //   bytesPerStep = 1000 × 2 + 196 × 2 + 16 × 2 = 2000 + 392 + 32 = 2424
    const hybridModel = {
      ...testInput.model,
      attention: {
        type: 'linear-mla-hybrid' as const,
        kvLoraRank: 5, qkRopeHeadDim: 1,
        qkNopeHeadDim: 1, vHeadDim: 1,
        numLinearLayers: 1, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 2
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, hybridMemory)
    expect(d.flopsPerStep).toBe(4332)
    expect(d.bytesPerStep).toBe(2424)
  })
})

import { describe, it, expect } from 'vitest'
import {
  activeParams,
  kvBytesPerTokenPerLayer,
  attentionDim,
  attendedSeqlenSummedOverLayers
} from '../../src/engine/memory'
import type { ModelArch } from '../../src/engine/types'

describe('activeParams', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 2, hiddenDim: 4, intermediateDim: 8,
    numHeads: 2, numKvHeads: 1, headDim: 2, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('returns paramCount for dense models', () => {
    expect(activeParams(base)).toBe(1000)
  })

  it('returns activeParamCount for MoE models', () => {
    const moe: ModelArch = {
      ...base,
      paramCount: 8000,
      architecture: {
        type: 'moe',
        numExperts: 8,
        numExpertsActive: 2,
        numSharedExperts: 0,
        activeParamCount: 2000
      }
    }
    expect(activeParams(moe)).toBe(2000)
  })
})

describe('kvBytesPerTokenPerLayer', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('GQA / full attention: 2 × kv_heads × head_dim × bytes (no layers factor)', () => {
    // 2 × 2 × 8 × 2 (fp16) = 64
    expect(kvBytesPerTokenPerLayer(base, 'fp16')).toBe(64)
  })

  it('sliding window uses same GQA formula', () => {
    const sliding: ModelArch = {
      ...base,
      attention: { type: 'sliding', window: 50 }
    }
    expect(kvBytesPerTokenPerLayer(sliding, 'fp16')).toBe(64)
  })

  it('MLA: (kv_lora + rope) × bytes (no factor of 2, no layers factor)', () => {
    const mla: ModelArch = {
      ...base,
      attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8 }
    }
    // (32 + 8) × 2 = 80
    expect(kvBytesPerTokenPerLayer(mla, 'fp16')).toBe(80)
  })
})

describe('attentionDim', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('returns numHeads × headDim for full attention', () => {
    // 8 × 8 = 64 (intentionally different from hiddenDim 16, so the test
    // would fail if the helper accidentally returned hiddenDim — c.f. PR #91)
    expect(attentionDim(base)).toBe(64)
  })

  it('returns numHeads × headDim for sliding window', () => {
    const sliding: ModelArch = {
      ...base,
      attention: { type: 'sliding', window: 50 }
    }
    expect(attentionDim(sliding)).toBe(64)
  })

  it('returns kvLoraRank + qkRopeHeadDim for MLA', () => {
    const mla: ModelArch = {
      ...base,
      attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8 }
    }
    expect(attentionDim(mla)).toBe(40)
  })
})

describe('attendedSeqlenSummedOverLayers', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('full: layers × seqlen', () => {
    expect(attendedSeqlenSummedOverLayers(base, 100)).toBe(400) // 4 × 100
    expect(attendedSeqlenSummedOverLayers(base, 0)).toBe(0)
  })

  it('sliding: layers × min(seqlen, window)', () => {
    const m: ModelArch = { ...base, attention: { type: 'sliding', window: 50 } }
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(200) // 4 × 50
    expect(attendedSeqlenSummedOverLayers(m, 30)).toBe(120)  // 4 × 30
    expect(attendedSeqlenSummedOverLayers(m, 50)).toBe(200)  // 4 × 50
  })

  it('mla: layers × seqlen (dimensional reduction is in attentionDim, not seqlen)', () => {
    const m: ModelArch = { ...base, attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8 } }
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(400) // 4 × 100
  })

  it('hybrid: numSliding × min(seqlen, window) + numGlobal × seqlen', () => {
    const m: ModelArch = {
      ...base,
      layers: 6,
      attention: {
        type: 'hybrid', slidingWindow: 50,
        numSlidingLayers: 5, numGlobalLayers: 1
      }
    }
    // seqlen > window: 5 × min(100, 50) + 1 × 100 = 250 + 100 = 350
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(350)
    // seqlen < window: 5 × 30 + 1 × 30 = 180 (sliding cap inactive)
    expect(attendedSeqlenSummedOverLayers(m, 30)).toBe(180)
    // seqlen == window: 5 × 50 + 1 × 50 = 300
    expect(attendedSeqlenSummedOverLayers(m, 50)).toBe(300)
  })

  it('hybrid: throws when numSlidingLayers + numGlobalLayers ≠ model.layers', () => {
    const m: ModelArch = {
      ...base,
      layers: 6,
      attention: {
        type: 'hybrid', slidingWindow: 50,
        numSlidingLayers: 4, numGlobalLayers: 1  // 4+1=5 ≠ 6
      }
    }
    expect(() => attendedSeqlenSummedOverLayers(m, 100)).toThrow(/sum to model\.layers/)
  })
})

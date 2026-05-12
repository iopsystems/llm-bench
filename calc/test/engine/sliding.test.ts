import { describe, it, expect } from 'vitest'
import { effectiveAttentionLength, activeParams, kvBytesPerToken, attentionDim } from '../../src/engine/memory'
import type { ModelArch } from '../../src/engine/types'

describe('effectiveAttentionLength', () => {
  it('returns rawSeqlen for full attention', () => {
    expect(effectiveAttentionLength(100, { type: 'full' })).toBe(100)
    expect(effectiveAttentionLength(0, { type: 'full' })).toBe(0)
  })

  it('caps at window for sliding attention when raw exceeds window', () => {
    expect(effectiveAttentionLength(100, { type: 'sliding', window: 50 })).toBe(50)
  })

  it('returns raw when sliding window is larger than raw', () => {
    expect(effectiveAttentionLength(30, { type: 'sliding', window: 50 })).toBe(30)
  })

  it('returns raw when equal to window', () => {
    expect(effectiveAttentionLength(50, { type: 'sliding', window: 50 })).toBe(50)
  })
})

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
        activeParamCount: 2000
      }
    }
    expect(activeParams(moe)).toBe(2000)
  })
})

describe('kvBytesPerToken', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('GQA / full attention: 2 × layers × kv_heads × head_dim × bytes', () => {
    // 2 × 4 × 2 × 8 × 2 (fp16) = 256
    expect(kvBytesPerToken(base, 'fp16')).toBe(256)
  })

  it('sliding window uses same GQA formula', () => {
    const sliding: ModelArch = {
      ...base,
      attention: { type: 'sliding', window: 50 }
    }
    expect(kvBytesPerToken(sliding, 'fp16')).toBe(256)
  })

  it('MLA: layers × (kv_lora + rope) × bytes (no factor of 2)', () => {
    const mla: ModelArch = {
      ...base,
      attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8 }
    }
    // 4 × (32 + 8) × 2 = 320
    expect(kvBytesPerToken(mla, 'fp16')).toBe(320)
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

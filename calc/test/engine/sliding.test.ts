import { describe, it, expect } from 'vitest'
import { effectiveAttentionLength, activeParams } from '../../src/engine/memory'
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

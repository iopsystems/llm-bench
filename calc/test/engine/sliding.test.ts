import { describe, it, expect } from 'vitest'
import { effectiveAttentionLength } from '../../src/engine/memory'

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

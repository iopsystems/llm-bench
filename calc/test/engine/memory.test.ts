import { describe, it, expect } from 'vitest'
import { computeMemory } from '../../src/engine/memory'
import { testInput } from '../fixtures'

describe('computeMemory', () => {
  it('weights = paramCount × bytes(weight_dtype)', () => {
    // paramCount=1000, fp16=2 bytes → 2000 bytes
    const m = computeMemory(testInput)
    expect(m.weights).toBe(2000)
  })

  it('kvCachePerRequest = 2 × layers × kv_heads × head_dim × bytes(kv_dtype) × (prompt + output)', () => {
    // 2 × 2 × 1 × 2 × 2 (fp16) = 16 bytes per token
    // × (10 + 5) = 240 bytes per request
    const m = computeMemory(testInput)
    expect(m.kvCachePerRequest).toBe(240)
  })

  it('kvCacheTotal = kvCachePerRequest × concurrency', () => {
    // 240 × 2 = 480
    const m = computeMemory(testInput)
    expect(m.kvCacheTotal).toBe(480)
  })
})

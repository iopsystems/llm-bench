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

  it('activationsPeak = concurrency × prompt × (hidden + intermediate) × bytes(act_dtype) × 2', () => {
    // 2 × 10 × (4 + 8) × 2 (fp16) × 2 = 960 bytes
    const m = computeMemory(testInput)
    expect(m.activationsPeak).toBe(960)
  })

  it('total = weights + kvCacheTotal + activationsPeak', () => {
    // 2000 + 480 + 960 = 3440
    const m = computeMemory(testInput)
    expect(m.total).toBe(3440)
  })

  it('hbmCapacityGB echoed from chosen variant', () => {
    const m = computeMemory(testInput)
    expect(m.hbmCapacityGB).toBe(1)
  })

  it('headroom = hbmCapacity_bytes − total, fits when ≥ 0', () => {
    // 1 GB = 1_073_741_824 bytes; headroom = 1_073_741_824 − 3440
    const m = computeMemory(testInput)
    expect(m.headroom).toBe(1_073_741_824 - 3440)
    expect(m.fits).toBe(true)
  })

  it('fits=false and negative headroom on OOM', () => {
    const bigModel = { ...testInput.model, paramCount: 10_000_000_000 }  // 10B params × 2B = 20GB
    const m = computeMemory({ ...testInput, model: bigModel })
    expect(m.fits).toBe(false)
    expect(m.headroom).toBeLessThan(0)
  })

  it('kvCachePerRequest caps at window for sliding attention', () => {
    // testModel uses full attention; build a sliding variant with window=8
    // (prompt+output=15, so should cap at 8 tokens instead of 15)
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 8 }
    }
    const input = { ...testInput, model: slidingModel }
    const m = computeMemory(input)
    // 16 bytes per token × 8 (window) = 128 bytes per request
    expect(m.kvCachePerRequest).toBe(128)
    // × concurrency 2 = 256 bytes
    expect(m.kvCacheTotal).toBe(256)
  })
})

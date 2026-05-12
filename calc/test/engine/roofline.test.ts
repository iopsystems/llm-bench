import { describe, it, expect } from 'vitest'
import { roofline } from '../../src/engine/roofline'

describe('roofline', () => {
  it('returns compute regime when flops/tflops > bytes/bw', () => {
    // flops/tflops = 4 / 1e12 / 1e-12 = ...
    // Use simpler numbers: tflops in TFLOPs = 1 (so 1e12 FLOP/s), bw in GB/s = 1 (so 1e9 B/s)
    //   flops = 2e12 → time = 2s ; bytes = 1e9 → time = 1s → compute wins
    const r = roofline({ flops: 2e12, bytes: 1e9, tflops: 1, bwGBs: 1 })
    expect(r.regime).toBe('compute')
    expect(r.timeS).toBe(2)
  })

  it('returns memory regime when bytes/bw > flops/tflops', () => {
    //   flops = 1e12 → time = 1s ; bytes = 2e9 → time = 2s → memory wins
    const r = roofline({ flops: 1e12, bytes: 2e9, tflops: 1, bwGBs: 1 })
    expect(r.regime).toBe('memory')
    expect(r.timeS).toBe(2)
  })

  it('ties classify as memory regime (defensive choice)', () => {
    const r = roofline({ flops: 1e12, bytes: 1e9, tflops: 1, bwGBs: 1 })
    expect(r.timeS).toBe(1)
    expect(r.regime).toBe('memory')
  })
})

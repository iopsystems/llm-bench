import { describe, it, expect } from 'vitest'
import { roofline } from '../../src/engine/roofline'

describe('roofline — single-accelerator (existing behavior)', () => {
  it('compute-bound when flops/tflops > bytes/bw', () => {
    const r = roofline({ flops: 1e12, bytes: 1e6, tflops: 1, bwGBs: 1000 })
    expect(r.regime).toBe('compute')
    expect(r.timeS).toBeCloseTo(1e12 / 1e12, 12)  // = 1 s
  })

  it('memory-bound when bytes/bw > flops/tflops', () => {
    const r = roofline({ flops: 1e6, bytes: 1e9, tflops: 1, bwGBs: 1 })
    expect(r.regime).toBe('memory')
    expect(r.timeS).toBeCloseTo(1e9 / 1e9, 12)  // = 1 s
  })
})

describe('roofline — multi-accelerator (comms ceiling)', () => {
  it('comms-bound when commsBytes/interconnect > both other terms', () => {
    const r = roofline({
      flops: 1, bytes: 1,
      tflops: 1, bwGBs: 1,
      commsBytes: 1e10,
      interconnectBwGBs: 100
    })
    expect(r.regime).toBe('comms')
    expect(r.timeS).toBeCloseTo(0.1, 6)
  })

  it('memory-bound still wins if HBM term exceeds comms term', () => {
    const r = roofline({
      flops: 1, bytes: 1e10,
      tflops: 1, bwGBs: 1000,
      commsBytes: 1e8,
      interconnectBwGBs: 100
    })
    expect(r.regime).toBe('memory')
    expect(r.timeS).toBeCloseTo(0.01, 6)
  })

  it('commsBytes undefined → no comms ceiling (back-compat)', () => {
    const r = roofline({ flops: 1, bytes: 1e9, tflops: 1, bwGBs: 1 })
    expect(r.regime).toBe('memory')
  })
})

import { describe, it, expect } from 'vitest'
import { pairOpPoints } from '../../src/engine/opPoints'
import type { AcceleratorVariant } from '../../src/engine/types'

const variant = (opIds: string[]): AcceleratorVariant => ({
  id: 'v', label: 'V', hbmCapacityGB: 80,
  operatingPoints: opIds.map(id => ({
    id, label: id, tflops: { fp16: 1 }, hbmBandwidthGBs: 1
  }))
})

describe('pairOpPoints', () => {
  it('pairs matched ids: peak/peak, achievable/achievable', () => {
    const pairs = pairOpPoints(variant(['peak', 'achievable']), variant(['peak', 'achievable']))
    expect(pairs).toHaveLength(2)
    expect(pairs[0]).toMatchObject({ id: 'peak' })
    expect(pairs[0].prefillOp.id).toBe('peak')
    expect(pairs[0].decodeOp.id).toBe('peak')
    expect(pairs[1]).toMatchObject({ id: 'achievable' })
  })

  it("symmetric (same variant on both sides) collapses to that variant's op list", () => {
    const v = variant(['peak'])
    const pairs = pairOpPoints(v, v)
    expect(pairs).toHaveLength(1)
    expect(pairs[0].id).toBe('peak')
    expect(pairs[0].prefillOp).toBe(pairs[0].decodeOp)
  })

  it("falls back to decode side's first op when prefill name has no match", () => {
    // Prefill has 'peak' + 'achievable'; decode only has 'peak'.
    // The 'achievable' prefill op pairs with decode's only op (peak).
    const pairs = pairOpPoints(variant(['peak', 'achievable']), variant(['peak']))
    expect(pairs).toHaveLength(2)
    expect(pairs[0]).toMatchObject({ id: 'peak' })
    expect(pairs[1].prefillOp.id).toBe('achievable')
    expect(pairs[1].decodeOp.id).toBe('peak')   // fallback to decode's first op
    expect(pairs[1].id).toBe('achievable/peak')  // composite id signals the cross-fallback
  })

  it('single op-point on prefill, multiple on decode: still iterates over prefill list', () => {
    const pairs = pairOpPoints(variant(['peak']), variant(['peak', 'achievable']))
    expect(pairs).toHaveLength(1)
    expect(pairs[0].prefillOp.id).toBe('peak')
    expect(pairs[0].decodeOp.id).toBe('peak')   // matched-by-name
    expect(pairs[0].id).toBe('peak')
  })
})

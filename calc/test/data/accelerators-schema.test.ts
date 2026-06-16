import { describe, it, expect } from 'vitest'
import { ACCELERATORS } from '../../src/data'

// powerCapW lives on AcceleratorVariant (different form factors of the same
// chip — SXM vs PCIe, OAM vs NVL — have different thermal envelopes).
// tier lives on AcceleratorSpec (the entire chip targets one market segment).

describe('AcceleratorSpec schema additions', () => {
  it('every accelerator has a tier', () => {
    for (const a of ACCELERATORS) {
      expect(['datacenter', 'consumer'], `accelerator ${a.id}: tier`).toContain(a.tier)
    }
  })

  it('powerCapW is a positive integer when present', () => {
    for (const a of ACCELERATORS) {
      for (const v of a.variants) {
        if (v.powerCapW !== undefined) {
          expect(v.powerCapW, `${a.id}/${v.id}: powerCapW positive`).toBeGreaterThan(0)
          expect(Number.isInteger(v.powerCapW), `${a.id}/${v.id}: powerCapW int`).toBe(true)
        }
      }
    }
  })
})

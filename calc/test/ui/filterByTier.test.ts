import { describe, it, expect } from 'vitest'
import { filterByTier } from '../../src/ui/catalogOrder'
import type { AcceleratorSpec } from '../../src/engine/types'

const make = (id: string, tier: 'datacenter' | 'consumer'): AcceleratorSpec => ({
  id,
  name: id.toUpperCase(),
  vendor: 'Test',
  releaseDate: '2024-01',
  tier,
  variants: [],
})

const fixtures: AcceleratorSpec[] = [
  make('dc1', 'datacenter'),
  make('dc2', 'datacenter'),
  make('con1', 'consumer'),
  make('con2', 'consumer'),
]

describe('filterByTier', () => {
  it('shows only datacenter when showConsumer=false and no alwaysShowIds', () => {
    expect(filterByTier(fixtures, false).map(a => a.id)).toEqual(['dc1', 'dc2'])
  })

  it('shows all when showConsumer=true', () => {
    expect(filterByTier(fixtures, true).map(a => a.id)).toEqual(['dc1', 'dc2', 'con1', 'con2'])
  })

  it('preserves consumer entries listed in alwaysShowIds when showConsumer=false', () => {
    expect(filterByTier(fixtures, false, ['con1']).map(a => a.id)).toEqual(['dc1', 'dc2', 'con1'])
  })

  it('does not duplicate datacenter entries that are also in alwaysShowIds', () => {
    expect(filterByTier(fixtures, false, ['dc1']).map(a => a.id)).toEqual(['dc1', 'dc2'])
  })

  it('returns all when showConsumer=true ignoring alwaysShowIds', () => {
    expect(filterByTier(fixtures, true, ['con1']).map(a => a.id)).toEqual(['dc1', 'dc2', 'con1', 'con2'])
  })

  it('ignores unknown ids in alwaysShowIds', () => {
    expect(filterByTier(fixtures, false, ['nonexistent']).map(a => a.id)).toEqual(['dc1', 'dc2'])
  })

  it('returns empty for empty input', () => {
    expect(filterByTier([], false)).toEqual([])
  })

  it('preserves input order (no sorting)', () => {
    const shuffled = [fixtures[2], fixtures[0], fixtures[3], fixtures[1]]
    expect(filterByTier(shuffled, true).map(a => a.id)).toEqual(['con1', 'dc1', 'con2', 'dc2'])
  })
})

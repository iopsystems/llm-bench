import { describe, it, expect } from 'vitest'
import { DerivationBuilder } from '../../src/engine/derivation'

describe('DerivationBuilder', () => {
  it('records steps in order', () => {
    const b = new DerivationBuilder()
    b.add('weights', 'paramCount × bytes(dtype)', 2000, 'bytes')
    b.add('kv/token', '2 × layers × kv_heads × head_dim × bytes(dtype)', 16, 'bytes')
    expect(b.steps()).toEqual([
      { label: 'weights', expression: 'paramCount × bytes(dtype)', value: 2000, unit: 'bytes' },
      { label: 'kv/token', expression: '2 × layers × kv_heads × head_dim × bytes(dtype)', value: 16, unit: 'bytes' }
    ])
  })

  it('returns a defensive copy from steps()', () => {
    const b = new DerivationBuilder()
    b.add('x', 'y', 1, 'z')
    const out = b.steps()
    out.push({ label: 'evil', expression: '', value: 0, unit: '' })
    expect(b.steps()).toHaveLength(1)
  })
})

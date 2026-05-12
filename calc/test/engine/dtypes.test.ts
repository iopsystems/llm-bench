import { describe, it, expect } from 'vitest'
import { bytesOf } from '../../src/engine/dtypes'

describe('bytesOf', () => {
  it('returns correct bytes per element for each dtype', () => {
    expect(bytesOf('fp32')).toBe(4)
    expect(bytesOf('fp16')).toBe(2)
    expect(bytesOf('bf16')).toBe(2)
    expect(bytesOf('fp8')).toBe(1)
    expect(bytesOf('int8')).toBe(1)
    expect(bytesOf('int4')).toBe(0.5)
  })
})

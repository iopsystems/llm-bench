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

  it('returns 0.5 for fp4 (Blackwell native fp4 tensor cores)', () => {
    expect(bytesOf('fp4')).toBe(0.5)
  })

  it('returns 0.5 for int4 (same width as fp4 but distinct Dtype)', () => {
    expect(bytesOf('int4')).toBe(0.5)
  })

  it('returns 1 for fp8 and int8', () => {
    expect(bytesOf('fp8')).toBe(1)
    expect(bytesOf('int8')).toBe(1)
  })
})

import { describe, it, expect } from 'vitest'
import { parseTokenCount, formatTokenCount } from '../../src/ui/parseTokens'

describe('parseTokenCount', () => {
  it('parses plain integers', () => {
    expect(parseTokenCount('8192')).toBe(8192)
    expect(parseTokenCount('1')).toBe(1)
  })

  it('parses k suffix as ×1024 (binary, matching HF context conventions)', () => {
    expect(parseTokenCount('40k')).toBe(40960)
    expect(parseTokenCount('8K')).toBe(8192)
    expect(parseTokenCount('128k')).toBe(131072)
  })

  it('parses M suffix as ×1024² (binary)', () => {
    expect(parseTokenCount('1M')).toBe(1048576)
    expect(parseTokenCount('1m')).toBe(1048576)
  })

  it('accepts decimals with suffixes', () => {
    expect(parseTokenCount('1.5k')).toBe(1536)
    expect(parseTokenCount('0.5M')).toBe(524288)
  })

  it('trims whitespace and allows space before suffix', () => {
    expect(parseTokenCount('  40k  ')).toBe(40960)
    expect(parseTokenCount('40 k')).toBe(40960)
  })

  it('rejects invalid input by returning null', () => {
    expect(parseTokenCount('')).toBeNull()
    expect(parseTokenCount('abc')).toBeNull()
    expect(parseTokenCount('40kk')).toBeNull()
    expect(parseTokenCount('-5')).toBeNull()
    expect(parseTokenCount('40g')).toBeNull()
  })

  it('rejects zero and negative results (min 1 token)', () => {
    expect(parseTokenCount('0')).toBeNull()
    expect(parseTokenCount('0k')).toBeNull()
  })
})

describe('formatTokenCount', () => {
  it('formats exact powers cleanly', () => {
    expect(formatTokenCount(8192)).toBe('8k')
    expect(formatTokenCount(40960)).toBe('40k')
    expect(formatTokenCount(1048576)).toBe('1M')
  })

  it('falls back to raw integer for non-round values', () => {
    expect(formatTokenCount(2048)).toBe('2k')
    expect(formatTokenCount(500)).toBe('500')
    expect(formatTokenCount(1)).toBe('1')
  })

  it('round-trips with parseTokenCount for round values', () => {
    for (const n of [1024, 8192, 40960, 131072, 262144, 1048576]) {
      expect(parseTokenCount(formatTokenCount(n))).toBe(n)
    }
  })
})

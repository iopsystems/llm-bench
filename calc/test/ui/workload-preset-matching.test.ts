import { describe, it, expect } from 'vitest'
import { matchPreset, type WorkloadPreset } from '../../src/data/workload-presets'

const fixtures: WorkloadPreset[] = [
  { id: 'p1', name: 'P1', group: 'code-gen', promptTokens: 100, outputTokens: 200,
    sourceUrl: 'https://example.com/p1', sourceAccessedAt: '2026-06-08', description: 'fixture 1' },
  { id: 'p2', name: 'P2', group: 'other', promptTokens: 500, outputTokens: 500,
    sourceUrl: 'https://example.com/p2', sourceAccessedAt: '2026-06-08', description: 'fixture 2' },
]

describe('matchPreset', () => {
  it('returns the preset id when prompt+output match exactly', () => {
    expect(matchPreset({ promptTokens: 100, outputTokens: 200 }, fixtures)).toBe('p1')
    expect(matchPreset({ promptTokens: 500, outputTokens: 500 }, fixtures)).toBe('p2')
  })

  it("returns 'custom' when prompt is off by 1", () => {
    expect(matchPreset({ promptTokens: 101, outputTokens: 200 }, fixtures)).toBe('custom')
  })

  it("returns 'custom' when output is off by 1", () => {
    expect(matchPreset({ promptTokens: 100, outputTokens: 201 }, fixtures)).toBe('custom')
  })

  it("returns 'custom' on empty preset list", () => {
    expect(matchPreset({ promptTokens: 100, outputTokens: 200 }, [])).toBe('custom')
  })

  it("returns 'custom' when prompt matches one preset and output matches another (no partial match)", () => {
    expect(matchPreset({ promptTokens: 100, outputTokens: 500 }, fixtures)).toBe('custom')
  })
})

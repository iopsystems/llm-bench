import { describe, it, expect } from 'vitest'
import { WORKLOAD_PRESETS } from '../../src/data/workload-presets'

describe('WORKLOAD_PRESETS schema', () => {
  it('registry has at least one entry', () => {
    expect(WORKLOAD_PRESETS.length).toBeGreaterThan(0)
  })

  it('every preset has the required shape', () => {
    for (const p of WORKLOAD_PRESETS) {
      expect(p.id, `preset ${JSON.stringify(p)}: id`).toMatch(/^[a-z0-9-]+$/)
      expect(p.name.length, `preset ${p.id}: name non-empty`).toBeGreaterThan(0)
      expect(['code-gen', 'other'], `preset ${p.id}: group enum`).toContain(p.group)
      expect(p.promptTokens, `preset ${p.id}: promptTokens > 0`).toBeGreaterThan(0)
      expect(Number.isInteger(p.promptTokens), `preset ${p.id}: promptTokens int`).toBe(true)
      expect(p.outputTokens, `preset ${p.id}: outputTokens > 0`).toBeGreaterThan(0)
      expect(Number.isInteger(p.outputTokens), `preset ${p.id}: outputTokens int`).toBe(true)
      expect(() => new URL(p.sourceUrl), `preset ${p.id}: sourceUrl parses`).not.toThrow()
      expect(p.sourceAccessedAt, `preset ${p.id}: sourceAccessedAt YYYY-MM-DD`).toMatch(/^\d{4}-\d{2}-\d{2}$/)
      expect(p.description.length, `preset ${p.id}: description non-empty`).toBeGreaterThan(0)
      expect(p.description.length, `preset ${p.id}: description ≤100 chars`).toBeLessThanOrEqual(100)
    }
  })

  it('preset ids are unique', () => {
    const ids = WORKLOAD_PRESETS.map(p => p.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('(promptTokens, outputTokens) pairs are unique (matchPreset needs deterministic mapping)', () => {
    const pairs = WORKLOAD_PRESETS.map(p => `${p.promptTokens}/${p.outputTokens}`)
    expect(new Set(pairs).size).toBe(pairs.length)
  })
})

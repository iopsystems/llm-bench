import { describe, it, expect } from 'vitest'
import { MODELS } from '../../src/data'
import { kvBytesPerTokenPerLayer } from '../../src/engine/memory'

// 2025–2026 additions to existing publisher families. Values sourced from
// each model's HuggingFace config.json (see commit message for repos).

describe('DeepSeek-R1 (= deepseek_v3 architecture)', () => {
  it('exists and mirrors DeepSeek-V3 architecture exactly', () => {
    const r1 = MODELS.find(m => m.id === 'deepseek-r1')!
    const v3 = MODELS.find(m => m.id === 'deepseek-v3')!
    expect(r1.publisher).toBe('DeepSeek')
    expect(r1.attention).toEqual(v3.attention)
    expect(r1.architecture).toEqual(v3.architecture)
    expect(r1.layers).toBe(v3.layers)
    expect(r1.hiddenDim).toBe(v3.hiddenDim)
    expect(r1.paramCount).toBe(v3.paramCount)
    // Same MLA geometry ⇒ identical KV bytes/token/layer at fp16.
    expect(kvBytesPerTokenPerLayer(r1, 'fp16')).toBe(kvBytesPerTokenPerLayer(v3, 'fp16'))
  })
})

describe('Mistral 2025 (dense GQA)', () => {
  it('Mistral Small 3.2 — 40L/5120, 32/8 heads, 128k ctx', () => {
    const m = MODELS.find(x => x.id === 'mistral-small-3.2-24b')!
    expect(m.publisher).toBe('Mistral AI')
    expect([m.layers, m.hiddenDim, m.intermediateDim]).toEqual([40, 5120, 32768])
    expect([m.numHeads, m.numKvHeads, m.headDim]).toEqual([32, 8, 128])
    expect([m.vocabSize, m.maxContext]).toEqual([131072, 131072])
    expect(m.attention.type).toBe('full')
    expect(m.architecture.type).toBe('dense')
  })
  it('Magistral Small — same backbone, 40960 ctx', () => {
    const m = MODELS.find(x => x.id === 'magistral-small')!
    expect([m.layers, m.hiddenDim, m.numHeads, m.numKvHeads]).toEqual([40, 5120, 32, 8])
    expect(m.maxContext).toBe(40960)
    expect(m.attention.type).toBe('full')
  })
})

describe('Phi-4 2025 (dense GQA)', () => {
  it('Phi-4-mini — 32L/3072, 24/8 heads, vocab 200064', () => {
    const m = MODELS.find(x => x.id === 'phi-4-mini')!
    expect(m.publisher).toBe('Microsoft')
    expect([m.layers, m.hiddenDim, m.intermediateDim]).toEqual([32, 3072, 8192])
    expect([m.numHeads, m.numKvHeads, m.headDim]).toEqual([24, 8, 128])
    expect([m.vocabSize, m.maxContext]).toEqual([200064, 131072])
    expect(m.attention.type).toBe('full')
    expect(m.architecture.type).toBe('dense')
  })
  it('Phi-4-reasoning — Phi-4 14B backbone, 32768 ctx', () => {
    const m = MODELS.find(x => x.id === 'phi-4-reasoning')!
    const p4 = MODELS.find(x => x.id === 'phi-4')!
    expect([m.layers, m.hiddenDim, m.intermediateDim]).toEqual([40, 5120, 17920])
    expect([m.numHeads, m.numKvHeads]).toEqual([40, 10])
    expect(m.vocabSize).toBe(100352)
    expect(m.maxContext).toBe(32768)
    expect(m.paramCount).toBe(p4.paramCount)
  })
})

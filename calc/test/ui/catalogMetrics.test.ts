import { describe, it, expect } from 'vitest'
import { modelMetrics, skuMetrics } from '../../src/ui/catalogMetrics'
import { MODELS, ACCELERATORS } from '../../src/data'
import { SYSTEMS } from '../../src/data/systems'

describe('modelMetrics', () => {
  it('full-attention model: KV per token per layer = 2·kvHeads·headDim·2 bytes (fp16)', () => {
    const m = MODELS.find(x => x.id === 'llama-3.3-70b')!
    const r = modelMetrics(m)
    expect(r.kvBytesPerTokenPerLayer).toBe(2 * m.numKvHeads * m.headDim * 2)
    expect(r.kvBytesPerToken).toBe(r.kvBytesPerTokenPerLayer * m.layers)
    expect(r.gqaRatio).toBeCloseTo(m.numHeads / m.numKvHeads)
    expect(r.attentionLabel).toMatch(/grouped-query|full/i)
  })
  it('MoE model exposes active/total ratio', () => {
    const m = MODELS.find(x => x.id === 'deepseek-v3')!
    const r = modelMetrics(m)
    expect(r.moeActiveRatio).toBeCloseTo(
      (m.architecture as { activeParamCount: number }).activeParamCount / m.paramCount
    )
  })
  it('dense model has no moeActiveRatio', () => {
    const m = MODELS.find(x => x.id === 'llama-3.3-70b')!
    expect(modelMetrics(m).moeActiveRatio).toBeUndefined()
  })
})

describe('skuMetrics', () => {
  it('accelerator ridge = peak FLOPS / HBM BW per dtype', () => {
    const a = ACCELERATORS.find(x => x.id === 'h100')!
    const r = skuMetrics(a)
    const v = a.variants[0]
    const peak = v.operatingPoints.find(o => o.id === 'peak')!
    if (r.kind !== 'accelerator') throw new Error('expected accelerator')
    const ridgeBf16 = r.variants[0].operatingPoints
      .find(o => o.id === 'peak')!.ridgeByDtype['bf16']!
    expect(ridgeBf16).toBeCloseTo((peak.tflops['bf16']! * 1e12) / (peak.hbmBandwidthGBs * 1e9))
  })
  it('system exposes aggregate rollups', () => {
    const s = SYSTEMS.find(x => x.id === 'hgx-h100-8')!
    const r = skuMetrics(s)
    expect(r.kind).toBe('system')
    if (r.kind === 'system') expect(r.totalHbmGB).toBe(s.aggregate.totalHbmGB)
  })
})

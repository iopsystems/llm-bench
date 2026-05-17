import { describe, it, expect } from 'vitest'
import { modelMetrics, skuMetrics } from '../../src/ui/catalogMetrics'
import { MODELS, ACCELERATORS } from '../../src/data'
import { SYSTEMS } from '../../src/data/systems'
import { SOURCES } from '../../src/data/sources'

describe('modelMetrics', () => {
  it('full-attention model: KV per token per layer = 2·kvHeads·headDim·2 bytes (fp16)', () => {
    const m = MODELS.find(x => x.id === 'llama-3.3-70b')!
    const r = modelMetrics(m)
    expect(r.kvBytesPerTokenPerLayer).toBe(2 * m.numKvHeads * m.headDim * 2)
    expect(r.kvBytesPerToken).toBe(r.kvBytesPerTokenPerLayer * m.layers)
    expect(r.gqaRatio).toBe(m.numHeads / m.numKvHeads)
    expect(r.attentionLabel).toMatch(/grouped-query|full/i)
  })
  it('MoE model exposes active/total ratio', () => {
    const m = MODELS.find(x => x.id === 'deepseek-v3')!
    const r = modelMetrics(m)
    const arch = m.architecture as { type: 'moe'; activeParamCount: number }
    // Concrete ratio so a misrouted paramCount/active field is caught, not
    // masked by both sides moving together.
    expect(r.moeActiveRatio).toBeCloseTo(arch.activeParamCount / m.paramCount, 6)
    expect(r.moeActiveRatio).toBeGreaterThan(0)
    expect(r.moeActiveRatio).toBeLessThan(1)
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

  it('accelerator efficiency = achievable/peak TFLOPS per shared dtype', () => {
    const a = ACCELERATORS.find(x => x.id === 'h100')!
    const r = skuMetrics(a)
    if (r.kind !== 'accelerator') throw new Error('expected accelerator')
    const v = a.variants[0]
    const peak = v.operatingPoints.find(o => o.id === 'peak')!
    const ach = v.operatingPoints.find(o => o.id === 'achievable')!
    const eff = r.variants[0].efficiencyByDtype!
    expect(eff['bf16']).toBeCloseTo(ach.tflops['bf16']! / peak.tflops['bf16']!)
  })

  it('peakTable has one row per variant × peak dtype with correct TFLOPS and ridge', () => {
    const a = ACCELERATORS.find(x => x.id === 'h100')!
    const r = skuMetrics(a)
    if (r.kind !== 'accelerator') throw new Error('expected accelerator')
    // Row count = sum over variants of (# dtypes in that variant's peak op).
    const expectedRows = a.variants.reduce((n, v) => {
      const peak = v.operatingPoints.find(o => o.id === 'peak')!
      return n + Object.values(peak.tflops).filter(t => t !== undefined).length
    }, 0)
    expect(r.peakTable.length).toBe(expectedRows)
    const sxm = a.variants.find(v => v.id === 'sxm-80')!
    const sxmPeak = sxm.operatingPoints.find(o => o.id === 'peak')!
    const row = r.peakTable.find(p => p.variantId === 'sxm-80' && p.dtype === 'bf16')!
    expect(row.variantLabel).toBe(sxm.label)
    expect(row.tflops).toBe(sxmPeak.tflops['bf16'])
    expect(row.ridge).toBeCloseTo((sxmPeak.tflops['bf16']! * 1e12) / (sxmPeak.hbmBandwidthGBs * 1e9))
  })

  it('dtypeSupport classifies native / conversion / software from the tflops set', () => {
    const a = ACCELERATORS.find(x => x.id === 'h100')!
    const r = skuMetrics(a)
    if (r.kind !== 'accelerator') throw new Error('expected accelerator')
    const by = Object.fromEntries(r.dtypeSupport.map(d => [d.dtype, d]))
    // h100 peak tflops cover fp16/bf16/fp8/int8.
    expect(by['bf16'].support).toBe('native')
    expect(by['fp8'].support).toBe('native')
    // fp4 not native; nearest wider supported float is fp8.
    expect(by['fp4'].support).toBe('conversion')
    expect(by['fp4'].via).toBe('fp8')
    // int4 not native; nearest wider supported int is int8.
    expect(by['int4'].support).toBe('conversion')
    expect(by['int4'].via).toBe('int8')
    // fp32 wider than anything supported (max width 16) → software.
    expect(by['fp32'].support).toBe('software')
    expect(by['fp32'].via).toBeUndefined()
  })

  it('accelerator operating point carries resolved provenance', () => {
    const a = ACCELERATORS.find(x => x.id === 'h100')!
    const r = skuMetrics(a)
    if (r.kind !== 'accelerator') throw new Error('expected accelerator')
    const ach = r.variants[0].operatingPoints.find(o => o.id === 'achievable')!
    expect(ach.sources && ach.sources.length).toBeGreaterThan(0)
    // resolved title, not raw key
    expect(ach.sources![0]).toBe(SOURCES['mamf-finder'].title)
  })
})

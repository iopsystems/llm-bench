import { describe, it, expect } from 'vitest'
import { groupedDisaggFabrics, formatFabricLabel } from '../../src/ui/disaggFabrics'

describe('groupedDisaggFabrics', () => {
  it('H100: scale-up shows NVL-256 (compatible), excludes NVL72 (Blackwell-only)', () => {
    const g = groupedDisaggFabrics('h100')
    const ids = g.scaleUp.map(f => f.id)
    expect(ids).toContain('nvlink-4-nvl-256')
    expect(ids).not.toContain('nvlink-5-nvl72')
  })

  it('GB200: scale-up shows NVL72 (compatible), excludes NVL-256 (Hopper-only)', () => {
    const g = groupedDisaggFabrics('gb200')
    const ids = g.scaleUp.map(f => f.id)
    expect(ids).toContain('nvlink-5-nvl72')
    expect(ids).not.toContain('nvlink-4-nvl-256')
  })

  it('non-NVIDIA / non-TPU accelerator: scale-up is empty', () => {
    const g = groupedDisaggFabrics('cerebras-wse3')
    expect(g.scaleUp).toEqual([])
  })

  it('scale-out is the same for any accelerator', () => {
    const h = groupedDisaggFabrics('h100').scaleOut.map(f => f.id)
    const mi = groupedDisaggFabrics('mi300x').scaleOut.map(f => f.id)
    expect(h).toEqual(mi)
    // Should include the new RoCE / Spectrum-X entries.
    expect(h).toContain('roce-400')
    expect(h).toContain('spectrum-x-800')
    expect(h).toContain('ib-ndr')
  })

  it('scale-out sorted by perGpuBandwidthGBs descending', () => {
    const g = groupedDisaggFabrics('h100')
    for (let i = 1; i < g.scaleOut.length; i++) {
      expect(g.scaleOut[i - 1].perGpuBandwidthGBs).toBeGreaterThanOrEqual(g.scaleOut[i].perGpuBandwidthGBs)
    }
  })

  it('does not include intra-node / die-to-die fabrics', () => {
    const g = groupedDisaggFabrics('h100')
    const allIds = [...g.scaleUp, ...g.scaleOut].map(f => f.id)
    expect(allIds).not.toContain('nvlink-4')   // intra-node HGX baseboard
    expect(allIds).not.toContain('pcie-gen5-x16')
    expect(allIds).not.toContain('ultrafusion')
  })
})

describe('formatFabricLabel', () => {
  it('appends " — N GB/s/GPU" to the name', () => {
    const fab = { name: 'RoCEv2 400 GbE', perGpuBandwidthGBs: 100 } as const
    expect(formatFabricLabel(fab as any)).toBe('RoCEv2 400 GbE — 100 GB/s/GPU')
  })
})

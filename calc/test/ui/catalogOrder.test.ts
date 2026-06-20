import { describe, it, expect } from 'vitest'
import { orderModels, orderSkus } from '../../src/ui/catalogOrder'
import type { ModelArch, AcceleratorSpec, MultiAcceleratorSystem } from '../../src/engine/types'

// Minimal model stub — only the fields the comparator touches.
function model(p: Partial<ModelArch> & { id: string; publisher: string; releaseDate: string; paramCount: number }): ModelArch {
  return {
    name: p.id, family: 'fam', layers: 1, hiddenDim: 1, intermediateDim: 1,
    numHeads: 1, numKvHeads: 1, headDim: 1, vocabSize: 1,
    attention: { type: 'full' }, architecture: { type: 'dense' },
    numNextnLayers: 0, maxContext: 1, ...p,
  } as ModelArch
}

function accel(id: string, vendor: string, releaseDate: string, hbms: number[], name?: string): AcceleratorSpec {
  return {
    id, name: name ?? id, vendor, releaseDate, tier: 'datacenter',
    variants: hbms.map((gb, i) => ({
      id: `v${i}`, label: `${gb}GB`, hbmCapacityGB: gb, operatingPoints: [],
    })),
  }
}

function system(id: string, vendor: string, releaseDate: string, count: number, totalHbmGB: number): MultiAcceleratorSystem {
  return {
    id, name: id, vendor, releaseDate, formFactor: 'node',
    accelerator: { id: 'x', variantId: 'v0', count },
    interconnectId: 'ic',
    aggregate: { totalHbmGB, fabricBidirectionalTBs: 1 },
  } as MultiAcceleratorSystem
}

describe('orderModels', () => {
  it('groups by publisher; publisher order = recency of its newest model', () => {
    const models = [
      model({ id: 'old-a', publisher: 'Acme', releaseDate: '2024-01', paramCount: 1e9 }),
      model({ id: 'new-z', publisher: 'Zeta', releaseDate: '2025-06', paramCount: 1e9 }),
      model({ id: 'mid-a', publisher: 'Acme', releaseDate: '2025-01', paramCount: 1e9 }),
    ]
    const groups = orderModels(models)
    // Zeta's newest (2025-06) beats Acme's newest (2025-01) → Zeta first.
    expect(groups.map(g => g.publisher)).toEqual(['Zeta', 'Acme'])
  })

  it('within a publisher: newer first, then larger paramCount first', () => {
    const models = [
      model({ id: 'a-small-new', publisher: 'Acme', releaseDate: '2025-03', paramCount: 7e9 }),
      model({ id: 'a-big-new',   publisher: 'Acme', releaseDate: '2025-03', paramCount: 70e9 }),
      model({ id: 'a-old',       publisher: 'Acme', releaseDate: '2024-01', paramCount: 400e9 }),
    ]
    const [acme] = orderModels(models)
    expect(acme.models.map(m => m.id)).toEqual(['a-big-new', 'a-small-new', 'a-old'])
  })

  it('ties publisher recency by publisher name ascending for determinism', () => {
    const models = [
      model({ id: 'b1', publisher: 'Beta', releaseDate: '2025-01', paramCount: 1 }),
      model({ id: 'a1', publisher: 'Alpha', releaseDate: '2025-01', paramCount: 1 }),
    ]
    expect(orderModels(models).map(g => g.publisher)).toEqual(['Alpha', 'Beta'])
  })
})

describe('orderSkus', () => {
  it('groups by vendor; vendor order = recency of its newest SKU (accel or system)', () => {
    const accelerators = [
      accel('amd-old', 'AMD', '2023-12', [192]),
      accel('nv-new', 'NVIDIA', '2025-03', [180]),
    ]
    const systems = [
      system('amd-sys-new', 'AMD', '2025-06', 8, 1536),
    ]
    const groups = orderSkus(accelerators, systems)
    // AMD newest = system 2025-06, NVIDIA newest = 2025-03 → AMD first.
    expect(groups.map(g => g.publisher)).toEqual(['AMD', 'NVIDIA'])
  })

  it('within a vendor: singles before systems, then newer first, then count then HBM', () => {
    const accelerators = [
      accel('a-old', 'V', '2024-01', [80]),
      accel('a-new', 'V', '2025-01', [141]),
    ]
    const systems = [
      system('s-8',  'V', '2025-01', 8,  1128),
      system('s-72', 'V', '2025-01', 72, 13392),
    ]
    const [v] = orderSkus(accelerators, systems)
    expect(v.entries.map(e => e.id)).toEqual([
      'a-new',  // single, newer
      'a-old',  // single, older
      's-72',   // system, same date, bigger count first
      's-8',
    ])
  })

  it('single-accelerator size tie broken by HBM descending', () => {
    const accelerators = [
      accel('small', 'V', '2025-01', [80]),
      accel('big',   'V', '2025-01', [192]),
    ]
    const [v] = orderSkus(accelerators, [])
    expect(v.entries.map(e => e.id)).toEqual(['big', 'small'])
  })

  it('AMD SKUs group by product line: Instinct → Radeon AI PRO → Radeon PRO → Radeon RX', () => {
    // Deliberately scrambled input dates so pure-recency ordering would differ
    // from product-line ordering — the line rank must win first for AMD.
    const accelerators = [
      accel('rx',       'AMD', '2025-03', [16], 'AMD Radeon RX 9070 XT'),
      accel('instinct', 'AMD', '2023-12', [192], 'AMD Instinct MI300X'),
      accel('pro',      'AMD', '2023-04', [48], 'AMD Radeon PRO W7900'),
      accel('ai-pro',   'AMD', '2025-07', [32], 'AMD Radeon AI PRO R9700'),
    ]
    const [amd] = orderSkus(accelerators, [])
    expect(amd.entries.map(e => e.id)).toEqual(['instinct', 'ai-pro', 'pro', 'rx'])
  })

  it('within an AMD product line: newer first, then HBM descending', () => {
    const accelerators = [
      accel('rx-old',     'AMD', '2022-12', [24], 'AMD Radeon RX 7900 XTX'),
      accel('rx-new',     'AMD', '2025-03', [16], 'AMD Radeon RX 9070 XT'),
      accel('instinct-a', 'AMD', '2024-10', [256], 'AMD Instinct MI325X'),
    ]
    const [amd] = orderSkus(accelerators, [])
    // Instinct line first, then RX line ordered newer-first.
    expect(amd.entries.map(e => e.id)).toEqual(['instinct-a', 'rx-new', 'rx-old'])
  })

  it('non-AMD vendors keep pure recency ordering (no product-line grouping)', () => {
    const accelerators = [
      accel('nv-old', 'NVIDIA', '2022-10', [24], 'NVIDIA RTX 4090'),
      accel('nv-new', 'NVIDIA', '2025-01', [32], 'NVIDIA RTX 5090'),
    ]
    const [nv] = orderSkus(accelerators, [])
    expect(nv.entries.map(e => e.id)).toEqual(['nv-new', 'nv-old'])
  })
})

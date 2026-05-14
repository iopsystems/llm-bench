import { describe, it, expect } from 'vitest'
import { perRankMemoryDivisors, commsBytesPerStep } from '../../src/engine/parallelism'
import type { ModelArch } from '../../src/engine/types'
import { bytesOf } from '../../src/engine/dtypes'

const dense: ModelArch = {
  id: 'd', name: 'D', family: 't',
  layers: 32, hiddenDim: 4096, intermediateDim: 14336,
  numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
  paramCount: 7_000_000_000,
  maxContext: 8192,
  numNextnLayers: 0,
  attention: { type: 'full' },
  architecture: { type: 'dense' }
}
const moe: ModelArch = {
  ...dense,
  paramCount: 47_000_000_000,
  maxContext: 8192,
  architecture: {
    type: 'moe', numExperts: 8, numExpertsActive: 2,
    numSharedExperts: 0, activeParamCount: 13_000_000_000
  }
}

describe('perRankMemoryDivisors', () => {
  it('no parallelism: all divisors = 1', () => {
    const d = perRankMemoryDivisors([], {}, dense)
    expect(d.weights).toBe(1)
    expect(d.kv).toBe(1)
    expect(d.activations).toBe(1)
    expect(d.replicas).toBe(1)
  })

  it('TP=8 dense: weights/8, kv/8 (8 ≤ kvHeads), activations/8', () => {
    const d = perRankMemoryDivisors(['tp'], { tp: 8 }, dense)
    expect(d.weights).toBe(8)
    expect(d.kv).toBe(8)
    expect(d.activations).toBe(8)
    expect(d.replicas).toBe(1)
  })

  it('TP=16 with numKvHeads=8: weights/16, kv/8 (KV sharding capped)', () => {
    const d = perRankMemoryDivisors(['tp'], { tp: 16 }, dense)
    expect(d.weights).toBe(16)
    expect(d.kv).toBe(8)
    expect(d.activations).toBe(16)
  })

  it('DP=2: weights replicated (divisor 1), replicas=2', () => {
    const d = perRankMemoryDivisors(['dp'], { dp: 2 }, dense)
    expect(d.weights).toBe(1)
    expect(d.kv).toBe(1)
    expect(d.activations).toBe(1)
    expect(d.replicas).toBe(2)
  })

  it('TP=8 × DP=2: weights/8 (within replica), replicas=2', () => {
    const d = perRankMemoryDivisors(['tp', 'dp'], { tp: 8, dp: 2 }, dense)
    expect(d.weights).toBe(8)
    expect(d.kv).toBe(8)
    expect(d.activations).toBe(8)
    expect(d.replicas).toBe(2)
  })

  it('PP=4: weights/4, kv/4, activations stay full', () => {
    const d = perRankMemoryDivisors(['pp'], { pp: 4 }, dense)
    expect(d.weights).toBe(4)
    expect(d.kv).toBe(4)
    expect(d.activations).toBe(1)
    expect(d.replicas).toBe(1)
  })

  it('EP=8 MoE: weights/8 (first-cut approximation)', () => {
    const d = perRankMemoryDivisors(['ep'], { ep: 8 }, moe)
    expect(d.weights).toBe(8)
    expect(d.kv).toBe(1)
    expect(d.activations).toBe(1)
    expect(d.replicas).toBe(1)
  })

  it('TP=8 × EP=8 MoE: weights/64 (TP × EP shard expert weights)', () => {
    const d = perRankMemoryDivisors(['tp', 'ep'], { tp: 8, ep: 8 }, moe)
    expect(d.weights).toBe(64)
    expect(d.kv).toBe(8)
    expect(d.activations).toBe(8)
    expect(d.replicas).toBe(1)
  })
})

describe('commsBytesPerStep', () => {
  it('TP all-reduce volume: 2 × layers × 2 × (N-1)/N × B × hidden × bytes', () => {
    const N = 8, B = 1, hidden = 4096, layers = 32
    const bytes = commsBytesPerStep(['tp'], { tp: N }, dense, B, 'fp16')
    const expected = 2 * layers * 2 * ((N - 1) / N) * B * hidden * bytesOf('fp16')
    expect(bytes).toBe(expected)
  })

  it('PP point-to-point volume: (N-1) × B × hidden × bytes', () => {
    const N = 4, B = 1, hidden = 4096
    const bytes = commsBytesPerStep(['pp'], { pp: N }, dense, B, 'fp16')
    const expected = (N - 1) * B * hidden * bytesOf('fp16')
    expect(bytes).toBe(expected)
  })

  it('EP all-to-all volume: 2 × moeLayers × (1 - 1/N) × B × hidden × bytes (MoE only)', () => {
    const N = 8, B = 1, hidden = 4096, layers = 32
    const bytes = commsBytesPerStep(['ep'], { ep: N }, moe, B, 'fp16')
    const expected = 2 * layers * (1 - 1 / N) * B * hidden * bytesOf('fp16')
    expect(bytes).toBe(expected)
  })

  it('EP on dense model: 0 (no MoE layers)', () => {
    const bytes = commsBytesPerStep(['ep'], { ep: 8 }, dense, 1, 'fp16')
    expect(bytes).toBe(0)
  })

  it('DP: 0 in inference', () => {
    const bytes = commsBytesPerStep(['dp'], { dp: 2 }, dense, 1, 'fp16')
    expect(bytes).toBe(0)
  })

  it('TP × EP composed: sum of both volumes', () => {
    const N = 8, B = 1, hidden = 4096, layers = 32
    const bytes = commsBytesPerStep(['tp', 'ep'], { tp: N, ep: N }, moe, B, 'fp16')
    const tpVol = 2 * layers * 2 * ((N - 1) / N) * B * hidden * bytesOf('fp16')
    const epVol = 2 * layers * (1 - 1 / N) * B * hidden * bytesOf('fp16')
    expect(bytes).toBe(tpVol + epVol)
  })

  it('no parallelism: 0', () => {
    const bytes = commsBytesPerStep([], {}, dense, 1, 'fp16')
    expect(bytes).toBe(0)
  })
})

import { defaultParallelism } from '../../src/engine/parallelism'
import type { MultiAcceleratorSystem } from '../../src/engine/types'

const hgxH100 = {
  id: 'hgx-h100-8', name: 'HGX H100', vendor: 'NVIDIA',
  formFactor: 'baseboard' as const,
  accelerator: { id: 'h100', variantId: 'sxm-80', count: 8 },
  interconnectId: 'nvlink-4',
  aggregate: { totalHbmGB: 640, fabricBidirectionalTBs: 7.2 }
} satisfies MultiAcceleratorSystem

const nvl72 = {
  ...hgxH100,
  id: 'gb200-nvl72',
  accelerator: { id: 'gb200', variantId: 'nvl72-186', count: 72 }
} satisfies MultiAcceleratorSystem

describe('defaultParallelism', () => {
  it('dense on HGX (N=8): TP=8', () => {
    const p = defaultParallelism(hgxH100, dense)
    expect(p.parallelism).toEqual(['tp'])
    expect(p.parallelismDegrees).toEqual({ tp: 8 })
  })

  it('MoE on HGX (N=8): TP=8, EP=8', () => {
    const p = defaultParallelism(hgxH100, moe)
    expect(p.parallelism).toContain('tp')
    expect(p.parallelism).toContain('ep')
    expect(p.parallelismDegrees).toEqual({ tp: 8, ep: 8 })
  })

  it('dense on NVL72 (N=72): TP=8 × PP=9', () => {
    const p = defaultParallelism(nvl72, dense)
    expect(p.parallelism).toContain('tp')
    expect(p.parallelism).toContain('pp')
    expect(p.parallelismDegrees).toEqual({ tp: 8, pp: 9 })
  })

  it('MoE on NVL72 (N=72): TP=8 × PP=9 × EP=72', () => {
    const p = defaultParallelism(nvl72, moe)
    expect(p.parallelism.sort()).toEqual(['ep', 'pp', 'tp'].sort())
    expect(p.parallelismDegrees).toEqual({ tp: 8, pp: 9, ep: 72 })
  })

  it('dense single-node MI300X (N=8): TP=8', () => {
    const mi300x8 = {
      ...hgxH100, id: 'mi300x-8', vendor: 'AMD',
      accelerator: { id: 'mi300x', variantId: 'oam-192', count: 8 }
    } satisfies MultiAcceleratorSystem
    const p = defaultParallelism(mi300x8, dense)
    expect(p.parallelismDegrees).toEqual({ tp: 8 })
  })
})

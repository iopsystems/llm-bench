import { describe, it, expect } from 'vitest'
import { perRankMemoryDivisors } from '../../src/engine/parallelism'
import type { ModelArch } from '../../src/engine/types'

const dense: ModelArch = {
  id: 'd', name: 'D', family: 't',
  layers: 32, hiddenDim: 4096, intermediateDim: 14336,
  numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
  paramCount: 7_000_000_000,
  numNextnLayers: 0,
  attention: { type: 'full' },
  architecture: { type: 'dense' }
}
const moe: ModelArch = {
  ...dense,
  paramCount: 47_000_000_000,
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

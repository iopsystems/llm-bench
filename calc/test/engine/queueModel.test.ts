import { describe, it, expect } from 'vitest'
import { computeNMax } from '../../src/engine/queueModel'
import { ACCELERATORS, MODELS } from '../../src/data'
import type { CalcInput } from '../../src/engine/types'

function inputFor(acceleratorId: string, variantId: string, modelId: string): CalcInput {
  const accelerator = ACCELERATORS.find(a => a.id === acceleratorId)!
  const model = MODELS.find(m => m.id === modelId)!
  return {
    accelerator,
    acceleratorVariantId: variantId,
    model,
    quant: { weights: 'bf16', kv: 'fp16', activations: 'bf16' },
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
  }
}

describe('computeNMax', () => {
  it('returns a positive integer for a model that fits with headroom', () => {
    // H100 SXM-80 (80 GB HBM), Llama-3.3-70B at bf16: weights ≈ 140 GB → doesn't fit
    // single-chip; need multi-device or a bigger chip. Use H200 SXM-141 instead.
    const r = computeNMax(inputFor('h200', 'sxm-141', 'llama-3.3-70b'))
    expect(r.boundBy).toBe('kv')
    expect(r.nMax).toBeGreaterThan(0)
    expect(Number.isInteger(r.nMax)).toBe(true)
  })

  it('returns {nMax: 0, boundBy: weights} when weights alone exceed HBM', () => {
    // Llama-3.3-70B bf16 ≈ 140 GB > 80 GB on H100 SXM-80.
    const r = computeNMax(inputFor('h100', 'sxm-80', 'llama-3.3-70b'))
    expect(r.boundBy).toBe('weights')
    expect(r.nMax).toBe(0)
  })
})

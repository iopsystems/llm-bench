import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  acceleratorId, variantId, systemId, modelId, workload, quant,
  concurrencyOverride, nMaxCalc, nMaxDecode, effectiveConcurrency,
  heterogeneous, decodeAcceleratorId, decodeVariantId, decodeSystemId,
  prefillAcceleratorId, prefillVariantId, prefillSystemId,
} from '../../src/ui/stores'

function resetStores() {
  acceleratorId.set('h200')
  variantId.set('sxm-141')
  systemId.set('')
  modelId.set('llama-3.3-70b')
  quant.set({ weights: 'bf16', kv: 'fp16', activations: 'bf16' })
  workload.set({ promptTokens: 2048, outputTokens: 512, concurrency: 1 })
  concurrencyOverride.set(null)
  heterogeneous.set(false)
  prefillAcceleratorId.set('')
  prefillVariantId.set('')
  prefillSystemId.set('')
  decodeAcceleratorId.set('')
  decodeVariantId.set('')
  decodeSystemId.set('')
}

describe('concurrencyOverride + effectiveConcurrency', () => {
  beforeEach(resetStores)

  it('default override is null; effective tracks nMaxCalc', () => {
    expect(get(concurrencyOverride)).toBeNull()
    expect(get(nMaxCalc)).toBeGreaterThan(0)
    expect(get(effectiveConcurrency)).toBe(get(nMaxCalc))
  })

  it('setting override to N makes effective return N', () => {
    concurrencyOverride.set(7)
    expect(get(effectiveConcurrency)).toBe(7)
  })

  it('clearing override (set to null) reverts effective to nMaxCalc', () => {
    concurrencyOverride.set(7)
    concurrencyOverride.set(null)
    expect(get(effectiveConcurrency)).toBe(get(nMaxCalc))
  })

  it('changing hardware re-derives nMaxCalc; effective follows when override is null', () => {
    const before = get(nMaxCalc)
    acceleratorId.set('h100')
    variantId.set('sxm-80')
    // Llama-3.3-70B at bf16 doesn't fit H100-80; nMax should be 0.
    expect(get(nMaxCalc)).toBe(0)
    expect(get(nMaxCalc)).not.toBe(before)
    // Effective floors at 1 even when nMax = 0, so the engine never sees concurrency=0.
    expect(get(effectiveConcurrency)).toBe(1)
  })

  it('override stays sticky when hardware changes', () => {
    concurrencyOverride.set(5)
    acceleratorId.set('h100')
    variantId.set('sxm-80')
    expect(get(effectiveConcurrency)).toBe(5)
  })
})

describe('nMaxDecode vs nMaxCalc under het=on', () => {
  beforeEach(resetStores)

  it('with symmetric hw (het off), nMaxCalc < nMaxDecode (prefill activations are larger)', () => {
    const calc = get(nMaxCalc)
    const decode = get(nMaxDecode)
    // Both positive (H200 + 70B fits)
    expect(calc).toBeGreaterThan(0)
    expect(decode).toBeGreaterThan(0)
    // Prefill activations include promptTokens × hidden — strictly larger than
    // decode activations, so prefill-bound nMax is smaller.
    expect(calc).toBeLessThan(decode)
  })

  it('with het=on + smaller decode hw, nMaxDecode < nMaxCalc', () => {
    // Calc/prefill stays on H200; decode cluster moves to H100-80.
    heterogeneous.set(true)
    prefillAcceleratorId.set('h200')
    prefillVariantId.set('sxm-141')
    decodeAcceleratorId.set('h100')
    decodeVariantId.set('sxm-80')
    // H100-80 can't fit 70B weights → nMaxDecode = 0; nMaxCalc still positive.
    expect(get(nMaxDecode)).toBe(0)
    expect(get(nMaxCalc)).toBeGreaterThan(0)
  })
})

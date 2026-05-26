import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { calcPayloadFromHash, encodeState, decodeState, readUrlIntoStores } from '../../src/ui/share'
import { modelId, quant } from '../../src/ui/stores'
import { MODELS } from '../../src/data'

describe('calcPayloadFromHash', () => {
  it('extracts payload after calc?', () => {
    expect(calcPayloadFromHash('#calc?a=h100&m=x')).toBe('a=h100&m=x')
  })
  it('legacy bare payload (no calc prefix) still works', () => {
    expect(calcPayloadFromHash('#a=h100&m=x')).toBe('a=h100&m=x')
  })
  it('info routes carry no calc payload', () => {
    expect(calcPayloadFromHash('#info/model/deepseek-v3')).toBe('')
    expect(calcPayloadFromHash('#info')).toBe('')
  })
  it('empty hash → empty', () => {
    expect(calcPayloadFromHash('')).toBe('')
    expect(calcPayloadFromHash('#calc')).toBe('')
  })
})

// Behavioral test for the model-implies-native-quant fallback in
// applyToStores. We don't export applyToStores, so exercise it via the
// public readUrlIntoStores → encodeState/decodeState surface indirectly by
// reaching into the stores after a hand-crafted apply. The simpler path:
// import the module under test and call the helpers through their exported
// API. Here decodeState alone is enough to verify the contract that the
// URL `?m=` key by itself does not produce a quant in the partial — the
// store-level fallback is verified by setting modelId via the store and
// inspecting the quant the public surface would land on.
describe('URL with model but no quant → quant seeded from native', () => {
  const fp8Model = MODELS.find(m => m.nativeDtype === 'fp8')!
  const bf16Model = MODELS.find(m => m.nativeDtype === 'bf16')!

  beforeEach(() => {
    // Reset stores to a known prior state so the seed-from-model effect is
    // observable (i.e. quant doesn't already equal the target's nativeDtype).
    modelId.set(bf16Model.id)
    quant.set({ weights: 'fp4', kv: 'int8', activations: 'fp4' })
  })

  it('decodeState returns no quant when URL omits w/kv/ac', () => {
    expect(decodeState(`m=${fp8Model.id}`).quant).toBeUndefined()
  })

  it('explicit quant in URL still decodes verbatim', () => {
    const enc = encodeState({
      acceleratorId: 'h100', variantId: 'sxm-80', systemId: '', modelId: fp8Model.id,
      quant: { weights: 'fp8', kv: 'fp16', activations: 'fp8' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
      parallelismOverride: null, disaggKvTransferFabricId: '', disaggFirstTokenOnPrefill: true,
    })
    expect(enc).not.toContain('ld=')   // no more ld key
    expect(decodeState(enc).quant).toEqual({ weights: 'fp8', kv: 'fp16', activations: 'fp8' })
  })

  it('store-level fallback: URL with model but no quant reseeds weights+activations (kv preserved)', () => {
    // Stub minimal window so readUrlIntoStores' early-return doesn't fire
    // (vitest env is 'node'; no DOM by default).
    const w = globalThis as { window?: { location: { hash: string } } }
    w.window = { location: { hash: `#calc?m=${fp8Model.id}` } }
    try {
      readUrlIntoStores()
      expect(get(modelId)).toBe(fp8Model.id)
      // weights+activations reseeded from native; kv preserved from prior state.
      expect(get(quant)).toEqual({ weights: 'fp8', kv: 'int8', activations: 'fp8' })
    } finally {
      delete w.window
    }
  })
})

import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { tabPayloadFromHash, encodeState, decodeState, readUrlIntoStores } from '../../src/ui/share'
import { modelId, quant } from '../../src/ui/stores'
import { MODELS } from '../../src/data'

describe('tabPayloadFromHash', () => {
  it('extracts payload after calc?', () => {
    expect(tabPayloadFromHash('#calc?a=h100&m=x', 'calc')).toBe('a=h100&m=x')
  })
  it('extracts payload after sim?', () => {
    expect(tabPayloadFromHash('#sim?a=h100&m=x', 'sim')).toBe('a=h100&m=x')
  })
  it('returns empty for mismatched tab', () => {
    expect(tabPayloadFromHash('#calc?a=h100', 'sim')).toBe('')
    expect(tabPayloadFromHash('#sim?a=h100',  'calc')).toBe('')
  })
  it('legacy bare payload counts as calc-tab payload', () => {
    expect(tabPayloadFromHash('#a=h100&m=x', 'calc')).toBe('a=h100&m=x')
    expect(tabPayloadFromHash('#a=h100&m=x', 'sim')).toBe('')
  })
  it('info routes carry no calc/sim payload', () => {
    expect(tabPayloadFromHash('#info/model/deepseek-v3', 'calc')).toBe('')
    expect(tabPayloadFromHash('#info/model/deepseek-v3', 'sim')).toBe('')
    expect(tabPayloadFromHash('#info', 'calc')).toBe('')
  })
  it('empty hash → empty', () => {
    expect(tabPayloadFromHash('', 'calc')).toBe('')
    expect(tabPayloadFromHash('#calc', 'calc')).toBe('')
    expect(tabPayloadFromHash('#sim',  'sim')).toBe('')
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

  it('readUrlIntoStores accepts #sim? prefix too', () => {
    const w = globalThis as { window?: { location: { hash: string } } }
    w.window = { location: { hash: `#sim?m=${fp8Model.id}` } }
    try {
      readUrlIntoStores()
      expect(get(modelId)).toBe(fp8Model.id)
    } finally {
      delete w.window
    }
  })
})

describe('disagg URL encoding (single-chip + scale-out fabric)', () => {
  it('encodes dk/df when no system is selected', () => {
    const state = {
      acceleratorId: 'h100', variantId: 'sxm-80', systemId: '', modelId: 'llama-3.3-70b',
      quant: { weights: 'bf16', kv: 'fp16', activations: 'bf16' } as const,
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
      parallelismOverride: null,
      disaggKvTransferFabricId: 'roce-400',
      disaggFirstTokenOnPrefill: false,
    }
    const enc = encodeState(state)
    expect(enc).toContain('dk=roce-400')
    expect(enc).toContain('df=0')
  })

  it('round-trips single-chip + disagg state through decode', () => {
    const enc = 'a=h100&v=sxm-80&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=1&dk=roce-400'
    const decoded = decodeState(enc)
    expect(decoded.disaggKvTransferFabricId).toBe('roce-400')
    expect(decoded.disaggFirstTokenOnPrefill).toBe(true)   // omitted from URL → default true
  })
})

import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { tabPayloadFromHash, encodeState, decodeState, readUrlIntoStores } from '../../src/ui/share'
import {
  modelId, quant,
  acceleratorId, variantId, systemId,
  heterogeneous,
  prefillAcceleratorId, prefillVariantId, prefillSystemId,
  decodeAcceleratorId, decodeVariantId, decodeSystemId,
} from '../../src/ui/stores'
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
      heterogeneous: false,
      prefillAcceleratorId: '', prefillVariantId: '',
      prefillSystemId: '', prefillParallelismOverride: null,
      decodeAcceleratorId: '', decodeVariantId: '',
      decodeSystemId: '', decodeParallelismOverride: null,
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
      heterogeneous: false,
      prefillAcceleratorId: '', prefillVariantId: '',
      prefillSystemId: '', prefillParallelismOverride: null,
      decodeAcceleratorId: '', decodeVariantId: '',
      decodeSystemId: '', decodeParallelismOverride: null,
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

describe('heterogeneous P/D URL state', () => {
  const base = {
    acceleratorId: 'h100', variantId: 'sxm-80', systemId: '', modelId: 'llama-3.3-70b',
    quant: { weights: 'bf16', kv: 'fp16', activations: 'bf16' } as const,
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
    parallelismOverride: null,
    disaggKvTransferFabricId: 'roce-400',
    disaggFirstTokenOnPrefill: true,
  }

  const emptyOverrides = {
    prefillAcceleratorId: '', prefillVariantId: '',
    prefillSystemId: '', prefillParallelismOverride: null,
    decodeAcceleratorId: '', decodeVariantId: '',
    decodeSystemId: '', decodeParallelismOverride: null,
  }

  it('omits all decode-side keys when heterogeneous=false', () => {
    const enc = encodeState({ ...base, heterogeneous: false,
      ...emptyOverrides,
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141' })
    expect(enc).not.toContain('het=')
    expect(enc).not.toContain('a2=')
    expect(enc).not.toContain('v2=')
  })

  it('emits het=1 + a2/v2 when heterogeneous=true with single-chip decode side', () => {
    const enc = encodeState({ ...base, heterogeneous: true,
      ...emptyOverrides,
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141' })
    expect(enc).toContain('het=1')
    expect(enc).toContain('a2=h200')
    expect(enc).toContain('v2=sxm-141')
    expect(enc).not.toContain('s2=')
  })

  it('emits s2 instead of a2/v2 when decode side is multi-device', () => {
    const enc = encodeState({ ...base, heterogeneous: true,
      ...emptyOverrides,
      decodeSystemId: 'hgx-h200-8' })
    expect(enc).toContain('s2=hgx-h200-8')
    expect(enc).not.toMatch(/(^|&)a2=/)
    expect(enc).not.toMatch(/(^|&)v2=/)
  })

  it('emits a1/v1 when prefill cluster is overridden (single-chip)', () => {
    const enc = encodeState({ ...base, heterogeneous: true,
      ...emptyOverrides,
      prefillAcceleratorId: 'mi300x', prefillVariantId: 'oam-192',
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141' })
    expect(enc).toContain('a1=mi300x')
    expect(enc).toContain('v1=oam-192')
    expect(enc).toContain('a2=h200')
  })

  it('emits s1 when prefill cluster is a multi-device override', () => {
    const enc = encodeState({ ...base, heterogeneous: true,
      ...emptyOverrides,
      prefillSystemId: 'hgx-h100-8',
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141' })
    expect(enc).toContain('s1=hgx-h100-8')
    expect(enc).not.toMatch(/(^|&)a1=/)
    expect(enc).not.toMatch(/(^|&)v1=/)
  })

  it('emits p1 when prefill parallelism is overridden', () => {
    const enc = encodeState({ ...base, heterogeneous: true,
      ...emptyOverrides,
      prefillSystemId: 'hgx-h100-8',
      prefillParallelismOverride: { parallelism: ['tp', 'pp'], parallelismDegrees: { tp: 4, pp: 2 } } })
    expect(enc).toContain('p1=tp4.pp2')
  })

  it('omits prefill-override keys when prefill side is empty (het=1 with only decode override)', () => {
    const enc = encodeState({ ...base, heterogeneous: true,
      ...emptyOverrides,
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141' })
    expect(enc).toContain('het=1')
    expect(enc).not.toMatch(/(^|&)a1=/)
    expect(enc).not.toMatch(/(^|&)v1=/)
    expect(enc).not.toMatch(/(^|&)s1=/)
    expect(enc).not.toMatch(/(^|&)p1=/)
  })

  it('round-trips: prefill + decode overrides preserved', () => {
    const original = { ...base, heterogeneous: true,
      ...emptyOverrides,
      prefillAcceleratorId: 'mi300x', prefillVariantId: 'oam-192',
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141' }
    const round = decodeState(encodeState(original))
    expect(round.heterogeneous).toBe(true)
    expect(round.prefillAcceleratorId).toBe('mi300x')
    expect(round.prefillVariantId).toBe('oam-192')
    expect(round.decodeAcceleratorId).toBe('h200')
    expect(round.decodeVariantId).toBe('sxm-141')
  })

  it('round-trips: encode then decode preserves heterogeneous decode-only state', () => {
    const original = { ...base, heterogeneous: true,
      ...emptyOverrides,
      decodeAcceleratorId: 'h200', decodeVariantId: 'sxm-141' }
    const round = decodeState(encodeState(original))
    expect(round.heterogeneous).toBe(true)
    expect(round.decodeAcceleratorId).toBe('h200')
    expect(round.decodeVariantId).toBe('sxm-141')
  })

  it('URL without het keys decodes to non-heterogeneous (no override state)', () => {
    const round = decodeState('a=h100&v=sxm-80&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=1')
    expect(round.heterogeneous).toBeUndefined()
    expect(round.prefillAcceleratorId).toBeUndefined()
    expect(round.decodeAcceleratorId).toBeUndefined()
  })

  it('backward-compat: old het=1 URL with only decode keys decodes cleanly (prefill empty)', () => {
    const round = decodeState('a=h100&v=sxm-80&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=1&dk=roce-400&het=1&a2=h200&v2=sxm-141')
    expect(round.heterogeneous).toBe(true)
    expect(round.decodeAcceleratorId).toBe('h200')
    expect(round.prefillAcceleratorId).toBeUndefined()
  })
})

describe('readUrlIntoStores seeds prefill+decode overrides when het=1', () => {
  // Without this seeding, an old-format URL like het=1&a2=X (no a1) leaves the
  // prefill-override stores empty; the disagg block then reactively follows
  // the shared (monolithic) hw and the user can no longer change them
  // independently. The bug we're locking out: monolithic edits coupling to
  // disagg prefill display + calculation.
  beforeEach(() => {
    acceleratorId.set('h100')
    variantId.set('sxm-80')
    systemId.set('')
    heterogeneous.set(false)
    prefillAcceleratorId.set('')
    prefillVariantId.set('')
    prefillSystemId.set('')
    decodeAcceleratorId.set('')
    decodeVariantId.set('')
    decodeSystemId.set('')
  })

  it('het=1 URL with only decode keys seeds prefill from shared', () => {
    const w = globalThis as { window?: { location: { hash: string } } }
    w.window = { location: { hash: '#sim?a=h100&v=sxm-80&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=1&dk=roce-400&het=1&a2=h200&v2=sxm-141' } }
    try {
      readUrlIntoStores()
      expect(get(heterogeneous)).toBe(true)
      // Prefill seeded from shared (= a/v from URL).
      expect(get(prefillAcceleratorId)).toBe('h100')
      expect(get(prefillVariantId)).toBe('sxm-80')
      // Decode came from a2/v2.
      expect(get(decodeAcceleratorId)).toBe('h200')
      expect(get(decodeVariantId)).toBe('sxm-141')
    } finally {
      delete w.window
    }
  })

  it('het=1 URL with explicit a1/v1 keeps URL-provided values (no overwrite)', () => {
    const w = globalThis as { window?: { location: { hash: string } } }
    w.window = { location: { hash: '#sim?a=h100&v=sxm-80&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=1&dk=roce-400&het=1&a1=mi300x&v1=oam-192&a2=h200&v2=sxm-141' } }
    try {
      readUrlIntoStores()
      expect(get(prefillAcceleratorId)).toBe('mi300x')
      expect(get(prefillVariantId)).toBe('oam-192')
    } finally {
      delete w.window
    }
  })

  it('het=0 URL leaves overrides untouched (no seeding)', () => {
    const w = globalThis as { window?: { location: { hash: string } } }
    w.window = { location: { hash: '#sim?a=h100&v=sxm-80&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512&c=1' } }
    try {
      readUrlIntoStores()
      expect(get(heterogeneous)).toBe(false)
      expect(get(prefillAcceleratorId)).toBe('')
      expect(get(decodeAcceleratorId)).toBe('')
    } finally {
      delete w.window
    }
  })
})

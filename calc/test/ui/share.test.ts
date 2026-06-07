import { describe, it, expect } from 'vitest'
import { encodeState, decodeState, type ShareableState } from '../../src/ui/share'
import { ACCELERATORS, MODELS } from '../../src/data'
import { SYSTEMS } from '../../src/data/systems'

// Pick a few real ids so round-trips exercise the validation paths against
// the actual registries.
const accel = ACCELERATORS[0]
const variant = accel.variants[0]
const model = MODELS[0]
const system = SYSTEMS[0]

const singleChipState: ShareableState = {
  acceleratorId: accel.id,
  variantId: variant.id,
  systemId: '',
  modelId: model.id,
  quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
  workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
  parallelismOverride: null,
  disaggKvTransferFabricId: '',
  disaggFirstTokenOnPrefill: true,
  heterogeneous: false,
  prefillAcceleratorId: '',
  prefillVariantId: '',
  prefillSystemId: '',
  prefillParallelismOverride: null,
  decodeAcceleratorId: '',
  decodeVariantId: '',
  decodeSystemId: '',
  decodeParallelismOverride: null,
  concurrencyOverride: null,
}

const multiDeviceState: ShareableState = {
  acceleratorId: system.accelerator.id,
  variantId: system.accelerator.variantId,
  systemId: system.id,
  modelId: model.id,
  quant: { weights: 'fp8', kv: 'fp16', activations: 'bf16' },
  workload: { promptTokens: 8192, outputTokens: 1024, concurrency: 16 },
  parallelismOverride: {
    parallelism: ['tp', 'pp'],
    parallelismDegrees: { tp: 8, pp: 2 },
  },
  disaggKvTransferFabricId: '',
  disaggFirstTokenOnPrefill: true,
  heterogeneous: false,
  prefillAcceleratorId: '',
  prefillVariantId: '',
  prefillSystemId: '',
  prefillParallelismOverride: null,
  decodeAcceleratorId: '',
  decodeVariantId: '',
  decodeSystemId: '',
  decodeParallelismOverride: null,
  concurrencyOverride: null,
}

describe('encodeState', () => {
  it('emits a/v for single-chip selection, drops s', () => {
    const s = encodeState(singleChipState)
    expect(s).toContain(`a=${accel.id}`)
    expect(s).toContain(`v=${variant.id}`)
    expect(s).not.toContain('s=')
  })

  it('emits s and skips a/v when a system is selected', () => {
    const s = encodeState(multiDeviceState)
    expect(s).toContain(`s=${system.id}`)
    expect(s).not.toMatch(/(^|&)a=/)
    expect(s).not.toMatch(/(^|&)v=/)
  })

  it('omits parallelism key when override is null', () => {
    const s = encodeState(singleChipState)
    expect(s).not.toMatch(/(^|&)p=/)
  })

  it('encodes parallelism as id+degree pairs joined by dots', () => {
    const s = encodeState(multiDeviceState)
    expect(s).toContain('p=tp8.pp2')
  })

  it('omits df when default (true), emits df=0 when disabled', () => {
    const base: ShareableState = {
      ...multiDeviceState,
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: true,
    }
    expect(encodeState(base)).not.toContain('df=')
    expect(encodeState(base)).toContain('dk=ib-ndr')

    const off = { ...base, disaggFirstTokenOnPrefill: false }
    expect(encodeState(off)).toContain('df=0')
  })

  it('encodes disagg keys for single-chip mode too', () => {
    const s = encodeState({
      ...singleChipState,
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: false,
    })
    expect(s).toContain('dk=ib-ndr')
    expect(s).toContain('df=0')
  })
})

describe('decodeState', () => {
  it('round-trips a single-chip configuration', () => {
    const round = decodeState(encodeState(singleChipState))
    expect(round.acceleratorId).toBe(accel.id)
    expect(round.variantId).toBe(variant.id)
    expect(round.systemId).toBe('')
    expect(round.modelId).toBe(model.id)
    expect(round.quant).toEqual(singleChipState.quant)
    // workload round-trips pt/ot only; concurrency is now owned by concurrencyOverride
    expect(round.workload).toEqual({ promptTokens: 2048, outputTokens: 512 })
    expect(round.concurrencyOverride).toBeUndefined() // null override → omitted from URL
    expect(round.parallelismOverride).toBeUndefined()
  })

  it('round-trips a multi-device configuration with parallelism', () => {
    const round = decodeState(encodeState(multiDeviceState))
    expect(round.systemId).toBe(system.id)
    // System pre-seeds accel + variant from the system definition.
    expect(round.acceleratorId).toBe(system.accelerator.id)
    expect(round.variantId).toBe(system.accelerator.variantId)
    expect(round.parallelismOverride).toEqual(multiDeviceState.parallelismOverride)
  })

  it('drops unknown accelerator id', () => {
    const round = decodeState(`a=not-a-real-gpu&v=x&m=${model.id}`)
    expect(round.acceleratorId).toBeUndefined()
    expect(round.variantId).toBeUndefined()
  })

  it('drops unknown system id', () => {
    const round = decodeState(`s=not-a-real-system&m=${model.id}`)
    expect(round.systemId).toBeUndefined()
    expect(round.acceleratorId).toBeUndefined()
  })

  it('falls back to first variant when accelerator is valid but variant is not', () => {
    const round = decodeState(`a=${accel.id}&v=not-a-real-variant&m=${model.id}`)
    expect(round.acceleratorId).toBe(accel.id)
    expect(round.variantId).toBe(accel.variants[0].id)
  })

  it('drops unknown model id', () => {
    const round = decodeState(`m=not-a-real-model`)
    expect(round.modelId).toBeUndefined()
  })

  it('drops invalid dtype combos', () => {
    const round = decodeState(`w=fp16&kv=fp16&ac=blah`)
    expect(round.quant).toBeUndefined()
  })

  it('ignores non-positive workload values', () => {
    const round = decodeState(`pt=0&ot=-5&c=abc`)
    expect(round.workload).toBeUndefined()
  })

  it('parses partial workload (only some keys)', () => {
    const round = decodeState(`pt=4096`)
    expect(round.workload).toEqual({ promptTokens: 4096 })
  })

  it('rejects malformed parallelism strings', () => {
    expect(decodeState(`p=tp.pp2`).parallelismOverride).toBeNull()
    expect(decodeState(`p=xy8`).parallelismOverride).toBeNull()
    expect(decodeState(`p=tp0`).parallelismOverride).toBeNull()
  })

  it('returns empty object for empty hash', () => {
    expect(decodeState('')).toEqual({})
  })

  it('ignores extraneous keys without throwing', () => {
    const round = decodeState(`m=${model.id}&unknown=foo&bar=baz`)
    expect(round.modelId).toBe(model.id)
  })
})

describe('encodeState then decodeState', () => {
  it('preserves a complex disagg configuration verbatim', () => {
    // Use a known interconnect from the registry so validation passes.
    const state: ShareableState = {
      ...multiDeviceState,
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: false,
    }
    const round = decodeState(encodeState(state))
    expect(round.disaggKvTransferFabricId).toBe('ib-ndr')
    expect(round.disaggFirstTokenOnPrefill).toBe(false)
  })
})

describe('concurrencyOverride URL encoding', () => {
  it('omits c= when override is null', () => {
    const state: ShareableState = {
      ...singleChipState,
      concurrencyOverride: null,
    }
    expect(encodeState(state)).not.toMatch(/(^|&)c=/)
  })

  it('emits c=N when override is set', () => {
    const state: ShareableState = {
      ...singleChipState,
      concurrencyOverride: 7,
    }
    expect(encodeState(state)).toContain('c=7')
  })

  it('decodes c=5 to concurrencyOverride=5', () => {
    expect(decodeState('c=5').concurrencyOverride).toBe(5)
  })

  it('decodes missing c= to concurrencyOverride undefined (recipient default null)', () => {
    expect(decodeState('a=h100&v=sxm-80').concurrencyOverride).toBeUndefined()
  })

  it('backward compat: old URL with c=1 sets override to 1', () => {
    expect(decodeState('c=1').concurrencyOverride).toBe(1)
  })

  it('ignores invalid c= value', () => {
    expect(decodeState('c=abc').concurrencyOverride).toBeUndefined()
    expect(decodeState('c=0').concurrencyOverride).toBeUndefined()
    expect(decodeState('c=-3').concurrencyOverride).toBeUndefined()
  })
})

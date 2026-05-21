import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { modelId, quant, lockDtype, initNativeDtypeSync } from '../../src/ui/stores'
import { MODELS } from '../../src/data'

const fp8Model = MODELS.find(m => m.nativeDtype === 'fp8')!.id
const bf16Model = MODELS.find(m => m.nativeDtype === 'bf16')!.id

describe('native-dtype re-seed', () => {
  let stop: () => void
  beforeEach(() => {
    stop?.()
    lockDtype.set(false)
    quant.set({ weights: 'fp16', kv: 'fp16', activations: 'fp16' })
    modelId.set(bf16Model)
    stop = initNativeDtypeSync()
  })

  it('unlocked: switching model reseeds weights+activations, not kv', () => {
    modelId.set(fp8Model)
    expect(get(quant)).toEqual({ weights: 'fp8', kv: 'fp16', activations: 'fp8' })
  })

  it('locked: switching model leaves quant untouched', () => {
    lockDtype.set(true)
    quant.set({ weights: 'fp4', kv: 'int8', activations: 'fp4' })
    modelId.set(fp8Model)
    expect(get(quant)).toEqual({ weights: 'fp4', kv: 'int8', activations: 'fp4' })
  })

  it('unlock then switch reseeds again', () => {
    lockDtype.set(true)
    modelId.set(fp8Model)
    lockDtype.set(false)
    modelId.set(bf16Model)
    expect(get(quant).weights).toBe('bf16')
    expect(get(quant).activations).toBe('bf16')
  })

  it('initial subscribe reseeds to current model nativeDtype when unlocked', () => {
    // beforeEach has already called initNativeDtypeSync() with modelId=bf16Model
    // and lockDtype=false; the initial fire must reseed weights+activations.
    expect(get(quant).weights).toBe('bf16')
    expect(get(quant).activations).toBe('bf16')
    expect(get(quant).kv).toBe('fp16')   // KV preserved
  })
})

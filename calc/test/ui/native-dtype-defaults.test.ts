import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { modelId, quant, initNativeDtypeSync } from '../../src/ui/stores'
import { MODELS } from '../../src/data'

const fp8Model = MODELS.find(m => m.nativeDtype === 'fp8')!.id
const bf16Model = MODELS.find(m => m.nativeDtype === 'bf16')!.id

describe('native-dtype re-seed', () => {
  let stop: () => void
  beforeEach(() => {
    stop?.()
    modelId.set(bf16Model)
    quant.set({ weights: 'fp16', kv: 'fp16', activations: 'fp16' })
    stop = initNativeDtypeSync()
  })

  it('switching model reseeds weights+activations, not kv', () => {
    modelId.set(fp8Model)
    expect(get(quant)).toEqual({ weights: 'fp8', kv: 'fp16', activations: 'fp8' })
  })

  it('initial subscribe is a no-op (URL-provided quant must survive load)', () => {
    // beforeEach set quant to fp16 across the board and then called
    // initNativeDtypeSync(). The initial subscribe fire must NOT reseed —
    // otherwise an explicit URL `w=/kv=/ac=` would get clobbered at startup.
    expect(get(quant)).toEqual({ weights: 'fp16', kv: 'fp16', activations: 'fp16' })
  })

  it('subsequent model switches still reseed (kv preserved)', () => {
    quant.set({ weights: 'fp4', kv: 'int8', activations: 'fp4' })
    modelId.set(fp8Model)
    expect(get(quant)).toEqual({ weights: 'fp8', kv: 'int8', activations: 'fp8' })
    modelId.set(bf16Model)
    expect(get(quant)).toEqual({ weights: 'bf16', kv: 'int8', activations: 'bf16' })
  })
})

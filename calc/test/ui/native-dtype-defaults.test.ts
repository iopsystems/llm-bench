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

  it('weight-only 4-bit natives (int4/fp4) seed weights only; activations fall back to bf16', () => {
    // Kimi K2.5 ships int4 W4A16-QAT weights; gpt-oss ships mxfp4 MoE weights.
    // Both compute in bf16 — and no accelerator operating point exposes 4-bit
    // matmul rates, so activations=int4 would make every SKU throw.
    modelId.set('kimi-k2.5')
    expect(get(quant)).toEqual({ weights: 'int4', kv: 'fp16', activations: 'bf16' })
    modelId.set('gpt-oss-120b')
    expect(get(quant)).toEqual({ weights: 'fp4', kv: 'fp16', activations: 'bf16' })
  })
})

describe('default quant is computable on a mainstream SKU', () => {
  it('every model calculates cleanly under its own default quant on H100', async () => {
    const { calculate } = await import('../../src/engine')
    const { ACCELERATORS } = await import('../../src/data')
    const { defaultActivationsFor } = await import('../../src/ui/stores')
    const h100 = ACCELERATORS.find(a => a.id === 'h100')!
    for (const m of MODELS) {
      const r = calculate({
        accelerator: h100,
        acceleratorVariantId: 'sxm-80',
        model: m,
        quant: { weights: m.nativeDtype, kv: 'fp16', activations: defaultActivationsFor(m.nativeDtype) },
        workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 }
      })
      for (const p of Object.values(r.perf)) {
        expect(Number.isFinite(p.decode.timePerTokenS), m.id).toBe(true)
      }
    }
  })
})

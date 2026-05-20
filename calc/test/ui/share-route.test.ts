import { describe, it, expect } from 'vitest'
import { calcPayloadFromHash, encodeState, decodeState } from '../../src/ui/share'

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

describe('lockDtype share state', () => {
  const base = {
    acceleratorId: 'h100', variantId: 'sxm-80', systemId: '', modelId: 'llama-3.3-70b',
    quant: { weights: 'bf16', kv: 'fp16', activations: 'bf16' } as const,
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
    parallelismOverride: null, disaggKvTransferFabricId: '', disaggFirstTokenOnPrefill: true,
  }
  it('round-trips lockDtype=true via ld=1', () => {
    const enc = encodeState({ ...base, lockDtype: true })
    expect(enc).toContain('ld=1')
    expect(decodeState(enc).lockDtype).toBe(true)
  })
  it('omits ld when false', () => {
    expect(encodeState({ ...base, lockDtype: false })).not.toContain('ld=')
  })
  it('quant present but no ld → lockDtype true (preserve sharer intent)', () => {
    expect(decodeState('m=llama-3.3-70b&w=fp8&kv=fp16&ac=fp8').lockDtype).toBe(true)
  })
  it('explicit ld=0 with quant is honored', () => {
    expect(decodeState('m=llama-3.3-70b&w=fp8&kv=fp16&ac=fp8&ld=0').lockDtype).toBe(false)
  })
  it('no quant, no ld → lockDtype undefined (caller defaults false)', () => {
    expect(decodeState('m=llama-3.3-70b').lockDtype).toBeUndefined()
  })
})

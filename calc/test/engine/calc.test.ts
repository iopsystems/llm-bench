import { describe, it, expect } from 'vitest'
import { calculate } from '../../src/engine/calc'
import { testInput } from '../fixtures'
import { ACCELERATORS } from '../../src/data/accelerators'
import { MODELS } from '../../src/data/models'
import { SYSTEMS } from '../../src/data/systems'
import type { CalcInput } from '../../src/engine/types'

describe('calculate', () => {
  it('returns memory matching computeMemory', () => {
    const r = calculate(testInput)
    expect(r.memory.weights).toBe(2000)
    expect(r.memory.kvCachePerRequest).toBe(240)
    expect(r.memory.kvCacheTotal).toBe(480)
    expect(r.memory.activationsPeak).toBe(960)
    expect(r.memory.total).toBe(3440)
    expect(r.memory.fits).toBe(true)
  })

  it('produces one perf tier per operating point', () => {
    const r = calculate(testInput)
    expect(Object.keys(r.perf)).toEqual(['peak'])
  })

  it('perf.peak has all the expected fields', () => {
    const r = calculate(testInput)
    const p = r.perf['peak']
    expect(p.prefill.flops).toBe(21600)
    expect(p.decode.flopsPerStep).toBe(4400)
    expect(p.ttftS).toBe(p.prefill.timeS)
    expect(p.outputTokenRate).toBeCloseTo(p.decode.aggregateTokensPerS, 9)
    expect(p.inputTokenRate).toBeCloseTo(testInput.workload.promptTokens / p.prefill.timeS, 6)
  })

  it('derivation is non-empty and ends with the final memory total', () => {
    const r = calculate(testInput)
    expect(r.derivation.length).toBeGreaterThan(0)
    const memoryTotalStep = r.derivation.find(s => s.label === 'memory total')
    expect(memoryTotalStep?.value).toBe(r.memory.total)
  })

  it('throws on unknown variant id', () => {
    expect(() => calculate({ ...testInput, acceleratorVariantId: 'nope' })).toThrow()
  })

  it('exposes kvTransferS=0 in non-disagg config', () => {
    const result = calculate(testInput)
    for (const tier of Object.values(result.perf)) {
      expect(tier.kvTransferS).toBe(0)
    }
  })

  it('exposes a positive kvTransferS when a disagg fabric is configured', () => {
    const h100 = ACCELERATORS.find(a => a.id === 'h100')!
    const llama70b = MODELS.find(m => m.id === 'llama-3.3-70b')!
    const hgxH100 = SYSTEMS.find(s => s.id === 'hgx-h100-8')!
    const inp: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: llama70b,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 },
      multiDevice: {
        system: hgxH100,
        parallelism: ['tp'],
        parallelismDegrees: { tp: 8 },
      },
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: false,
    }
    const result = calculate(inp)
    for (const tier of Object.values(result.perf)) {
      expect(tier.kvTransferS).toBeGreaterThan(0)
      // Sequential handoff: ttftS = prefill + kv transfer.
      expect(tier.ttftS).toBeCloseTo(tier.prefill.timeS + tier.kvTransferS, 9)
    }
  })

  it('disagg works without a multiDevice config (single-chip + scale-out fabric)', () => {
    // Two single-chip nodes connected by a scale-out fabric — no system selected.
    const inp = {
      ...testInput,
      disaggKvTransferFabricId: 'roce-400',
      disaggFirstTokenOnPrefill: false,
    }
    const result = calculate(inp)
    for (const tier of Object.values(result.perf)) {
      expect(tier.kvTransferS).toBeGreaterThan(0)
      expect(tier.ttftS).toBeCloseTo(tier.prefill.timeS + tier.kvTransferS, 9)
    }
  })

  it('heterogeneous P/D: decode perf uses decode-side accelerator', () => {
    // Build a fixture with decodeAccelerator different from prefill side.
    const h100 = ACCELERATORS.find(a => a.id === 'h100')!
    const h200 = ACCELERATORS.find(a => a.id === 'h200')!
    const inp = {
      ...testInput,
      accelerator: h100,
      acceleratorVariantId: h100.variants[0].id,
      decodeAccelerator: h200,
      decodeAcceleratorVariantId: h200.variants[0].id,
    }
    const result = calculate(inp)
    // h200 has higher HBM bandwidth → decode tpot should be lower than the symmetric h100 case.
    const symmetric = calculate({ ...inp, decodeAccelerator: undefined, decodeAcceleratorVariantId: undefined })
    const op = Object.keys(result.perf)[0]
    expect(result.perf[op].decode.timePerTokenS).toBeLessThan(symmetric.perf[op].decode.timePerTokenS)
  })

  it('heterogeneous P/D with firstTokenOnPrefill=true: TTFT uses prefill-cluster decode-step time', () => {
    const h100 = ACCELERATORS.find(a => a.id === 'h100')!
    const h200 = ACCELERATORS.find(a => a.id === 'h200')!
    const inp = {
      ...testInput,
      accelerator: h100,
      acceleratorVariantId: h100.variants[0].id,
      decodeAccelerator: h200,
      decodeAcceleratorVariantId: h200.variants[0].id,
      disaggKvTransferFabricId: 'ib-ndr',
      disaggFirstTokenOnPrefill: true,
    }
    const result = calculate(inp)
    const op = Object.keys(result.perf)[0]
    // TTFT = prefill.timeS + (decode step on prefill cluster's hw, NOT decode cluster's).
    const ttft = result.perf[op].ttftS
    const prefillTime = result.perf[op].prefill.timeS
    expect(ttft).toBeGreaterThan(prefillTime)
    // TTFT must NOT equal prefill + decode-cluster-tpot (which would be h200's number).
    // Relative check: H100 HBM (3.35 TB/s) is slower than H200 (4.8 TB/s) → the
    // first decode step on the prefill cluster should be ~1.43× the h200 tpot.
    const decodeClusterTpot = result.perf[op].decode.timePerTokenS
    const firstStepS = ttft - prefillTime
    expect(firstStepS / decodeClusterTpot).toBeGreaterThan(1.2)
  })

  it('symmetric (no decode fields) still works — backward compat', () => {
    const result = calculate(testInput)
    expect(Object.keys(result.perf).length).toBeGreaterThan(0)
  })
})

describe('calculate — real data integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const llama70b = MODELS.find(m => m.id === 'llama-3.3-70b')!

  const input: CalcInput = {
    accelerator: h100,
    acceleratorVariantId: 'sxm-80',
    model: llama70b,
    quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
  }

  it('Llama 3.3 70B on H100 SXM-80: weights are 141 GB (does not fit single-GPU)', () => {
    const r = calculate(input)
    // 70.55B params × 2 bytes = 141.1 GB
    expect(r.memory.weights / 1e9).toBeCloseTo(141.1, 0)
    expect(r.memory.fits).toBe(false)
  })

  it('Llama 3.3 70B prefill regime is compute-bound for batch=1, prompt=2048', () => {
    const r = calculate(input)
    // Long prefill on dense 70B model — compute term dominates
    expect(r.perf['peak'].prefill.regime).toBe('compute')
  })

  it('Llama 3.3 70B decode at batch=1 is memory-bound', () => {
    const r = calculate(input)
    // Classic single-stream decode: weight-load bandwidth dominates
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — sliding window integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const mistral = MODELS.find(m => m.id === 'mistral-7b-v0.1')!

  it('Mistral 7B at 32k prompt: KV cache bounded by 4k window, not 32k', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: mistral,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // kv per token = 2 × 32 × 8 × 128 × 2 = 131072 bytes
    // bounded at window 4096 → 131072 × 4096 = 536870912 bytes
    expect(r.memory.kvCachePerRequest).toBe(131072 * 4096)
    // Sanity: if it were full attention this would be 131072 × 32768 (8× more)
  })
})

describe('calculate — MoE integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const mixtral = MODELS.find(m => m.id === 'mixtral-8x7b')!

  it('Mixtral 8x7B on H100 SXM-80: weights use total params, decode bytes use active', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: mixtral,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)

    // memory.weights = paramCount (46.7B) × 2 bytes (fp16) ≈ 93.4 GB.
    // 93.4 GB > 80 GB capacity → memory.fits === false.
    expect(r.memory.weights / 1e9).toBeCloseTo(93.4, 0)
    expect(r.memory.fits).toBe(false)

    // decode.bytesPerStep = activeParamCount × 2 bytes + kvCachePerRequest × 1.
    // activeParamCount = 12.879B → 25.76 GB; KV is small at batch=1.
    // Expect decode.bytesPerStep ≈ 25.76 GB, well below paramCount-based 93.4 GB.
    const expectedActiveBytes = 12_879_204_352 * 2  // fp16
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(expectedActiveBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(expectedActiveBytes + 2e9)

    // Decode is memory-bound at batch=1 (active weight reads dominate).
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — MLA integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const dsv2 = MODELS.find(m => m.id === 'deepseek-v2')!

  it('DeepSeek-V2 at 32k prompt: KV cache uses MLA latent formula', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: dsv2,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // MLA KV per token = layers × (kv_lora + rope) × bytes(fp16)
    //                  = 60 × (512 + 64) × 2 = 69_120 bytes per token
    // × 32768 tokens = 2_264_924_160 bytes per request
    expect(r.memory.kvCachePerRequest).toBe(60 * (512 + 64) * 2 * 32768)

    // Sanity vs the GQA equivalent that would apply if attention.type were
    // 'full': 2 × 60 × 128 × 192 × 2 × 32768 ≈ 43× larger.
    // (headDim=192 = qk_nope_head_dim(128) + qk_rope_head_dim(64))
    const gqaEquivalent = 2 * 60 * 128 * 192 * 2 * 32768
    expect(gqaEquivalent / r.memory.kvCachePerRequest).toBeGreaterThan(40)
  })
})

describe('calculate — hybrid attention integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const gemma27b = MODELS.find(m => m.id === 'gemma-3-27b')!

  it('Gemma 3 27B at 8k prompt: KV cache uses hybrid formula (~3.8× smaller than full attention)', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: gemma27b,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Per-layer KV bytes per token = 2 × kvHeads × headDim × bytes(fp16)
    //                              = 2 × 16 × 128 × 2 = 8192
    // attendedSeq = 52 × min(8192, 1024) + 10 × 8192
    //             = 52 × 1024 + 10 × 8192
    //             = 53248 + 81920 = 135168
    // kvCachePerRequest = 8192 × 135168 = 1_107_296_256
    expect(r.memory.kvCachePerRequest).toBe(8192 * 135168)

    // Sanity vs the would-have-been full-attention value:
    //   2 × layers × kvHeads × headDim × bytes × seqlen
    // = 2 × 62 × 16 × 128 × 2 × 8192 = 8192 × (62 × 8192) = 8192 × 507_904
    const fullEquivalent = 8192 * 62 * 8192
    const ratio = fullEquivalent / r.memory.kvCachePerRequest
    expect(ratio).toBeGreaterThan(3.5)        // 3.76 actually
    expect(ratio).toBeLessThan(6.3)           // asymptote layers/numGlobal = 62/10 = 6.2
  })
})

describe('calculate — DeepSeek V3 (MLA + shared-expert MoE) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const dsv3 = MODELS.find(m => m.id === 'deepseek-v3')!

  it('DeepSeek V3 at 32k prompt: MLA KV cache matches V3 geometry', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: dsv3,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // MLA KV per-layer-per-token bytes = (kv_lora + rope) × bytes(fp16) = 576 × 2 = 1152
    // attendedSeq = layers × seq = 61 × 32768 = 1_998_848
    // kvCachePerRequest = 1152 × 1_998_848 = 2_302_672_896
    expect(r.memory.kvCachePerRequest).toBe(61 * (512 + 64) * 2 * 32768)

    // Sanity: GQA-equivalent (same kvHeads=128, headDim=192) would be ~85× larger.
    // 2 × 61 × 128 × 192 × 2 × 32768 vs 61 × 576 × 2 × 32768
    //   = (2 × 128 × 192) / 576 = 49152 / 576 = 85.33
    const gqaEquivalent = 2 * 61 * 128 * 192 * 2 * 32768
    expect(gqaEquivalent / r.memory.kvCachePerRequest).toBeGreaterThan(80)
  })

  it('DeepSeek V3 decode bytes/step use activeParams (37B), not paramCount (671B)', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: dsv3,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // decode.bytesPerStep = activeParams × bytes(fp16) + kvCachePerRequest × concurrency
    //                    = 37e9 × 2 + small KV
    // Lower bound: 37e9 × 2 = 74 GB (small KV is negligible at batch=1, prompt=2048)
    const activeBytes = 37_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 5e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — Mixtral 8x22B integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const mixtral22b = MODELS.find(m => m.id === 'mixtral-8x22b')!

  it('Mixtral 8x22B on H100 SXM-80: weights 282 GB do not fit; decode uses 39B active', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: mixtral22b,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // 141B × 2 bytes = 282 GB → does not fit single H100 (80 GB)
    expect(r.memory.weights / 1e9).toBeCloseTo(282, 0)
    expect(r.memory.fits).toBe(false)
    // decode.bytesPerStep ≈ 39B × 2 = 78 GB (active params, not 141B total)
    const activeBytes = 39_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 2e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — Kimi K2 integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const k2 = MODELS.find(m => m.id === 'kimi-k2')!

  it('Kimi K2 at 32k prompt: MLA KV cache uses identical formula to DeepSeek V3 (same layers + MLA dims)', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: k2,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // MLA: layers × (kv_lora + rope) × bytes × seq = 61 × 576 × 2 × 32768
    expect(r.memory.kvCachePerRequest).toBe(61 * (512 + 64) * 2 * 32768)
    // 1.026T × 2 bytes ≈ 2.05 TB → vastly exceeds single H100 capacity
    expect(r.memory.weights / 1e12).toBeCloseTo(2.05, 1)
    expect(r.memory.fits).toBe(false)
  })

  it('Kimi K2 decode bytes/step use activeParams (32B), not paramCount (1.04T)', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: k2,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    const activeBytes = 32_000_000_000 * 2  // 64 GB at fp16
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 5e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — GLM-4.5-Air integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const glm = MODELS.find(m => m.id === 'glm-4.5-air')!

  it('GLM-4.5-Air at 32k prompt: regular GQA KV cache with 12:1 head ratio', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: glm,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Regular attention KV: 2 × kvHeads × headDim × bytes × layers × seq
    //                    = 2 × 8 × 128 × 2 × 46 × 32768 ≈ 6.17 GB
    expect(r.memory.kvCachePerRequest).toBe(2 * 8 * 128 * 2 * 46 * 32768)
  })

  it('GLM-4.5-Air composes shared-expert MoE with regular GQA attention', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: glm,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // 106B × 2 = 212 GB → doesn't fit single H100
    expect(r.memory.weights / 1e9).toBeCloseTo(212, 0)
    expect(r.memory.fits).toBe(false)
    // decode bytes use activeParams (12B), not paramCount (106B)
    const activeBytes = 12_000_000_000 * 2  // 24 GB
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 2e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — DeepSeek V3.2 (DSA) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const v3 = MODELS.find(m => m.id === 'deepseek-v3')!
  const v32 = MODELS.find(m => m.id === 'deepseek-v3.2')!

  const baseInput: Omit<CalcInput, 'model'> = {
    accelerator: h100,
    acceleratorVariantId: 'sxm-80',
    quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
    workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
  }

  it('V3.2 KV cache at 32k is identical to V3 (DSA does not shrink KV)', () => {
    const r3 = calculate({ ...baseInput, model: v3 })
    const r32 = calculate({ ...baseInput, model: v32 })
    expect(r32.memory.kvCachePerRequest).toBe(r3.memory.kvCachePerRequest)
    expect(r32.memory.kvCachePerRequest).toBe(61 * (512 + 64) * 2 * 32768)
  })

  it('V3.2 prefill attention term shrinks by ratio seqlen/topK ≈ 16× vs V3', () => {
    const r3 = calculate({ ...baseInput, model: v3 })
    const r32 = calculate({ ...baseInput, model: v32 })
    // MLP term identical (same activeParams, same prompt). Attention term differs.
    const mlpTerm = 2 * 37_000_000_000 * 32768
    const v3AttentionTerm = r3.perf['peak'].prefill.flops - mlpTerm
    const v32AttentionTerm = r32.perf['peak'].prefill.flops - mlpTerm
    // V3:   attendedSeq = 61 × 32768 = 1_998_848
    // V3.2: attendedSeq = 61 × 2048  = 124_928 (capped at topK)
    // Ratio: 32768 / 2048 = 16
    expect(v3AttentionTerm / v32AttentionTerm).toBeCloseTo(32768 / 2048, 6)
  })
})

describe('calculate — GLM-5 (MLA + DSA + asymmetric head dims) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const glm5 = MODELS.find(m => m.id === 'glm-5')!

  it('GLM-5 at 32k prompt: MLA KV cache uses 78 layers (vs V3.2 61)', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: glm5,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // MLA KV per-layer-per-token = (kv_lora + rope) × bytes(fp16) = 576 × 2 = 1152
    // attendedSeq (kv side, DSA caches all tokens) = 78 × 32768 = 2_555_904
    // kvCachePerRequest = 1152 × 2_555_904 = 2_944_401_408 ≈ 2.94 GB
    expect(r.memory.kvCachePerRequest).toBe(78 * (512 + 64) * 2 * 32768)
  })

  it('GLM-5 decode bytes/step use activeParams (40B), not paramCount (744B)', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: glm5,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // 744B × 2 bytes = 1.488 TB → way exceeds H100 SXM-80 (80 GB)
    expect(r.memory.fits).toBe(false)
    // decode.bytesPerStep ≈ 40B × 2 = 80 GB (active params, not 744B)
    const activeBytes = 40_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 5e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — Kimi-Linear (linear + MLA hybrid) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const kl = MODELS.find(m => m.id === 'kimi-linear')!

  it('Kimi-Linear at 128k prompt: KV cache uses 7 MLA layers + 20 KDA state', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: kl,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 131072, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // MLA per-layer-per-token = (512+64) × 2 = 1152 bytes
    // attendedSeq (kv) = 7 × 131072 = 917_504
    // MLA KV cache = 1152 × 917_504 = 1_056_964_608 ≈ 1.057 GB
    // KDA state = 20 × 32 × 128² × 2 = 20_971_520 ≈ 20 MB
    // Total kvCachePerRequest = 1_077_936_128 ≈ 1.08 GB
    const mlaKv = 7 * (512 + 64) * 2 * 131072
    const kdaState = 20 * 32 * 128 * 128 * 2
    expect(r.memory.kvCachePerRequest).toBe(mlaKv + kdaState)
  })

  it('Kimi-Linear KV cache at 128k is ~3.78× smaller than hypothetical all-MLA equivalent', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: kl,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 131072, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Hypothetical all-MLA: 27 × 576 × 2 × 131072 = 4_076_863_488
    const allMlaEquivalent = 27 * 576 * 2 * 131072
    const ratio = allMlaEquivalent / r.memory.kvCachePerRequest
    expect(ratio).toBeGreaterThan(3.5)        // actual ≈ 3.78
    expect(ratio).toBeLessThan(3.9)           // asymptote 27/7 ≈ 3.86
  })

  it('Kimi-Linear decode at batch=1 is memory-bound on weight reads (3B active)', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: kl,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // 48B × 2 = 96 GB → exceeds H100 SXM-80 capacity (80 GB)
    expect(r.memory.fits).toBe(false)
    // decode bytes/step ≈ 3B × 2 = 6 GB (active params) + small KV/state
    const activeBytes = 3_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 1e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — DeepSeek V4 integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const v32 = MODELS.find(m => m.id === 'deepseek-v3.2')!
  const v4Flash = MODELS.find(m => m.id === 'deepseek-v4-flash')!
  const v4Pro = MODELS.find(m => m.id === 'deepseek-v4-pro')!

  it('V4-Pro at 1M context: KV cache sums CSA + HCA per-layer-type contributions', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: v4Pro,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 1048576, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Per-compressed-entry bytes = 2 × 1 × 512 × 2 = 2048
    // attendedSeqlen (kv) = 0 + 30 × (1048576/4 + 128) + 31 × (1048576/128 + 128)
    //                    = 30 × 262272 + 31 × 8320 = 7868160 + 257920 = 8126080
    // kvCachePerRequest = 2048 × 8126080 = 16_642_211_840 bytes ≈ 16.64 GB
    const expected = 2048 * (30 * (1048576 / 4 + 128) + 31 * (1048576 / 128 + 128))
    expect(r.memory.kvCachePerRequest).toBe(expected)
  })

  it('V4-Pro at 1M: KV cache is ~4.4× smaller than V3.2 at fp16 apples-to-apples', () => {
    const baseInput: Omit<CalcInput, 'model'> = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 1048576, outputTokens: 0, concurrency: 1 }
    }
    const r4 = calculate({ ...baseInput, model: v4Pro })
    const r32 = calculate({ ...baseInput, model: v32 })
    const ratio = r32.memory.kvCachePerRequest / r4.memory.kvCachePerRequest
    // V4-Pro fp16 / V3.2 fp16: actual ≈ 4.43×
    // Paper's "10×" claim assumes V4 uses fp8 KV — that's a deployment choice, not modeled here.
    expect(ratio).toBeGreaterThan(4)
    expect(ratio).toBeLessThan(5)
  })

  it('V4-Pro decode throughput is 2× the without-MTP equivalent (numNextnLayers=1)', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: v4Pro,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 }
    }
    const rMtp = calculate(input)
    const noMtpInput = { ...input, model: { ...v4Pro, numNextnLayers: 0 } }
    const rNoMtp = calculate(noMtpInput)
    expect(rMtp.perf['peak'].decode.aggregateTokensPerS).toBeCloseTo(
      rNoMtp.perf['peak'].decode.aggregateTokensPerS * 2, 6
    )
    expect(rMtp.perf['peak'].decode.timePerTokenS).toBeCloseTo(
      rNoMtp.perf['peak'].decode.timePerTokenS / 2, 12
    )
    // Per-pass FLOPs and bytes unchanged
    expect(rMtp.perf['peak'].decode.flopsPerStep).toBe(rNoMtp.perf['peak'].decode.flopsPerStep)
    expect(rMtp.perf['peak'].decode.bytesPerStep).toBe(rNoMtp.perf['peak'].decode.bytesPerStep)
  })

  it('V4-Flash at 128k: KV cache uses 2 sliding + 21 CSA + 20 HCA', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: v4Flash,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 131072, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // 2048 × (2 × 128 + 21 × (131072/4 + 128) + 20 × (131072/128 + 128))
    //      = 2048 × (256 + 690816 + 23040)
    //      = 2048 × 714112 = 1_462_501_376 ≈ 1.46 GB
    const expected = 2048 * (
      2 * 128 +
      21 * (131072 / 4 + 128) +
      20 * (131072 / 128 + 128)
    )
    expect(r.memory.kvCachePerRequest).toBe(expected)
  })

  it('V4-Pro 1.6T weights at fp16 do not fit single H100 SXM-80', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: v4Pro,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // 1.6T × 2 bytes = 3.2 TB
    expect(r.memory.weights / 1e12).toBeCloseTo(3.2, 1)
    expect(r.memory.fits).toBe(false)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('calculate — Qwen3.5 delta-hybrid integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!

  it('Qwen3.5-4B at 32k prompt: KV cache uses only 8 Gated Attention layers; DeltaNet state adds ~24 MB', () => {
    const qwen = MODELS.find(m => m.id === 'qwen3.5-4b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: qwen,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    const kv = 4096 * 262144
    const state = 24 * 32 * 128 * 128 * 2
    expect(r.memory.kvCachePerRequest).toBe(kv + state)
    const fullEq = 2 * 32 * 4 * 256 * 2 * 32768
    expect(fullEq / r.memory.kvCachePerRequest).toBeGreaterThan(3.5)
    expect(fullEq / r.memory.kvCachePerRequest).toBeLessThan(4.5)
  })

  it('Qwen3.5-397B-A17B at 32k prompt: large MoE with delta-hybrid attention', () => {
    const qwen = MODELS.find(m => m.id === 'qwen3.5-397b-a17b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: qwen,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    expect(r.memory.weights / 1e9).toBeCloseTo(794, 0)
    expect(r.memory.fits).toBe(false)
    const kv = 2 * 2 * 256 * 2 * 15 * 32768
    const state = 45 * 64 * 128 * 128 * 2
    expect(r.memory.kvCachePerRequest).toBe(kv + state)
  })

  it('Qwen3.5-397B-A17B decode uses activeParams (17B), not paramCount (397B)', () => {
    const qwen = MODELS.find(m => m.id === 'qwen3.5-397b-a17b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: qwen,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    const activeBytes = 17_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 2e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })

  it('Qwen3.5-27B at 32k prompt: MoE with delta-hybrid attention', () => {
    const qwen = MODELS.find(m => m.id === 'qwen3.5-27b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: qwen,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    const kv = 4096 * 16 * 32768
    const state = 48 * 64 * 128 * 128 * 2
    expect(r.memory.kvCachePerRequest).toBe(kv + state)
  })

  it('Qwen3.5-9B at 64k prompt: KV cache ratio vs full attention ≈ 4× (8/32 layers)', () => {
    const qwen = MODELS.find(m => m.id === 'qwen3.5-9b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: qwen,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 65536, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    const kv = 4096 * 8 * 65536
    const state = 24 * 32 * 128 * 128 * 2
    expect(r.memory.kvCachePerRequest).toBe(kv + state)
    const fullEq = 2 * 32 * 4 * 256 * 2 * 65536
    const ratio = fullEq / r.memory.kvCachePerRequest
    expect(ratio).toBeGreaterThan(3.5)
    expect(ratio).toBeLessThan(4.5)
  })

  it('Qwen3.5-4B at 8k is memory-bound on decode (4B dense)', () => {
    const qwen = MODELS.find(m => m.id === 'qwen3.5-4b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: qwen,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    expect(r.memory.weights / 1e9).toBeCloseTo(8.48, 1)
    expect(r.memory.fits).toBe(true)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})

describe('models data — maxContext', () => {
  it('every model has a positive maxContext value', () => {
    for (const m of MODELS) {
      expect(m.maxContext, `${m.id}`).toBeGreaterThan(0)
    }
  })

  it('DeepSeek V4 and Kimi-Linear support 1M context', () => {
    expect(MODELS.find(m => m.id === 'deepseek-v4-pro')!.maxContext).toBe(1048576)
    expect(MODELS.find(m => m.id === 'deepseek-v4-flash')!.maxContext).toBe(1048576)
    expect(MODELS.find(m => m.id === 'kimi-linear')!.maxContext).toBe(1048576)
  })

  it('Llama 3.x has 128k context', () => {
    expect(MODELS.find(m => m.id === 'llama-3.3-70b')!.maxContext).toBe(131072)
    expect(MODELS.find(m => m.id === 'llama-3.1-405b')!.maxContext).toBe(131072)
  })
})

describe('calculate — multi-GPU integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const llama70b = MODELS.find(m => m.id === 'llama-3.3-70b')!
  const v3 = MODELS.find(m => m.id === 'deepseek-v3')!
  const hgxH100 = SYSTEMS.find(s => s.id === 'hgx-h100-8')!
  const nvl72 = SYSTEMS.find(s => s.id === 'gb200-nvl72')!

  it('HGX H100 + Llama 70B + TP=8: weights/8 fits per GPU', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: llama70b,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
      multiDevice: {
        system: hgxH100,
        parallelism: ['tp'],
        parallelismDegrees: { tp: 8 }
      }
    }
    const r = calculate(input)
    expect(r.memory.perRank).toBeDefined()
    expect(r.memory.perRank!.weights / 1e9).toBeCloseTo(17.6, 1)
    expect(r.memory.perRank!.fits).toBe(true)
    expect(r.memory.fits).toBe(false)
  })

  it('HGX H100 + Llama 70B + TP=8: regime is one of compute/memory/comms', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: llama70b,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
      multiDevice: {
        system: hgxH100,
        parallelism: ['tp'],
        parallelismDegrees: { tp: 8 }
      }
    }
    const r = calculate(input)
    expect(['compute', 'memory', 'comms']).toContain(r.perf['peak'].prefill.regime)
    expect(['compute', 'memory', 'comms']).toContain(r.perf['peak'].decode.regime)
  })

  it('GB200 NVL72 + DeepSeek V3 + TP=8 × EP=72: weights / 576 fits per GPU', () => {
    const gb200 = ACCELERATORS.find(a => a.id === 'gb200')!
    const input: CalcInput = {
      accelerator: gb200,
      acceleratorVariantId: 'nvl72-186',
      model: v3,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
      multiDevice: {
        system: nvl72,
        parallelism: ['tp', 'ep'],
        parallelismDegrees: { tp: 8, ep: 72 }
      }
    }
    const r = calculate(input)
    expect(r.memory.perRank).toBeDefined()
    expect(r.memory.perRank!.weights / 1e9).toBeLessThan(5)
    expect(r.memory.perRank!.fits).toBe(true)
  })

  it('single-accelerator path unchanged when multiDevice omitted', () => {
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: llama70b,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    expect(r.memory.perRank).toBeUndefined()
    expect(r.memory.fits).toBe(false)
  })
})

describe('calculate — disaggregated serving (KV transfer TTFT bump)', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const llama70b = MODELS.find(m => m.id === 'llama-3.3-70b')!
  const hgxH100 = SYSTEMS.find(s => s.id === 'hgx-h100-8')!

  function input(disaggFabric?: string, firstTokenOnPrefill?: boolean): CalcInput {
    return {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: llama70b,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 },
      multiDevice: {
        system: hgxH100,
        parallelism: ['tp'],
        parallelismDegrees: { tp: 8 },
      },
      ...(disaggFabric && { disaggKvTransferFabricId: disaggFabric }),
      ...(firstTokenOnPrefill !== undefined && { disaggFirstTokenOnPrefill: firstTokenOnPrefill }),
    }
  }

  it('integrated serving (no disaggKvTransferFabricId): ttftS = prefill.timeS', () => {
    const r = calculate(input())
    expect(r.perf['peak'].ttftS).toBe(r.perf['peak'].prefill.timeS)
  })

  it('disagg default (first-token-on-prefill): ttftS = prefill.timeS + decode.timePerTokenS', () => {
    const r = calculate(input('ib-ndr'))
    expect(r.perf['peak'].ttftS).toBeCloseTo(
      r.perf['peak'].prefill.timeS + r.perf['peak'].decode.timePerTokenS, 12
    )
  })

  it('disagg sequential (first-token-on-prefill=false): ttftS = prefill + kvCachePerRequest / 50e9', () => {
    const r = calculate(input('ib-ndr', false))
    // IB-NDR perDirectionGBs = 50 → 50e9 B/s
    const transferS = r.memory.kvCachePerRequest / (50 * 1e9)
    expect(r.perf['peak'].ttftS).toBeCloseTo(r.perf['peak'].prefill.timeS + transferS, 12)
  })

  it('disagg sequential over slower fabric: TTFT bump scales inversely with fabric BW', () => {
    const rNdr = calculate(input('ib-ndr', false))
    const rHdr = calculate(input('ib-hdr', false))
    // HDR perDirectionGBs = 25, NDR = 50; HDR transfer should be ~2× NDR transfer.
    const ndrBump = rNdr.perf['peak'].ttftS - rNdr.perf['peak'].prefill.timeS
    const hdrBump = rHdr.perf['peak'].ttftS - rHdr.perf['peak'].prefill.timeS
    expect(hdrBump / ndrBump).toBeCloseTo(2, 3)
  })

  it('disagg default: TTFT independent of fabric speed (transfer hidden behind first decode)', () => {
    const rNdr = calculate(input('ib-ndr'))
    const rHdr = calculate(input('ib-hdr'))
    // Both should equal prefill + 1 decode step regardless of fabric.
    expect(rNdr.perf['peak'].ttftS).toBe(rHdr.perf['peak'].ttftS)
  })

  it('disagg does not affect outputTokenRate (decode unchanged)', () => {
    const rIntegrated = calculate(input())
    const rDisagg = calculate(input('ib-ndr'))
    expect(rDisagg.perf['peak'].outputTokenRate).toBe(rIntegrated.perf['peak'].outputTokenRate)
  })

  it('disagg with unknown fabric id falls through silently (no transfer cost)', () => {
    const r = calculate(input('does-not-exist'))
    expect(r.perf['peak'].ttftS).toBe(r.perf['peak'].prefill.timeS)
  })
})

describe('calculate — gpt-oss (alternating sliding/full + MoE) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!

  it('gpt-oss-120b at 8k prompt: KV bounded by 128-token window on half the layers', () => {
    const m = MODELS.find(x => x.id === 'gpt-oss-120b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Per-layer KV bytes/token = 2 × 8 × 64 × 2 = 2048
    // attendedSeq = 18 × min(8192, 128) + 18 × 8192 = 2304 + 147456 = 149760
    expect(r.memory.kvCachePerRequest).toBe(2048 * (18 * 128 + 18 * 8192))
  })

  it('gpt-oss-120b weights 234 GB at fp16 do not fit; decode reads 5.1B active', () => {
    const m = MODELS.find(x => x.id === 'gpt-oss-120b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    expect(r.memory.weights / 1e9).toBeCloseTo(234, 0)
    expect(r.memory.fits).toBe(false)
    const activeBytes = 5_100_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 1e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })

  it('gpt-oss-20b at fp16 fits a single H100 SXM-80', () => {
    const m = MODELS.find(x => x.id === 'gpt-oss-20b')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    // 21B × 2 = 42 GB
    expect(r.memory.weights / 1e9).toBeCloseTo(42, 0)
    expect(r.memory.fits).toBe(true)
  })
})

describe('calculate — Llama 4 (chunked-attention ≈ hybrid + MoE) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!

  it('Llama 4 Maverick at 32k prompt: 36 chunked layers bounded at 8192, 12 global', () => {
    const m = MODELS.find(x => x.id === 'llama-4-maverick')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // Per-layer KV bytes/token = 2 × 8 × 128 × 2 = 4096
    // attendedSeq = 36 × min(32768, 8192) + 12 × 32768 = 294912 + 393216 = 688128
    expect(r.memory.kvCachePerRequest).toBe(4096 * (36 * 8192 + 12 * 32768))
  })

  it('Llama 4 Maverick weights 800 GB at fp16; decode reads 17B active', () => {
    const m = MODELS.find(x => x.id === 'llama-4-maverick')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    expect(r.memory.weights / 1e9).toBeCloseTo(800, 0)
    expect(r.memory.fits).toBe(false)
    const activeBytes = 17_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 2e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })

  it('Llama 4 Scout: 109B total / 17B active, 10M trained context', () => {
    const m = MODELS.find(x => x.id === 'llama-4-scout')!
    expect(m.paramCount).toBe(109_000_000_000)
    expect(m.maxContext).toBe(10485760)
    expect(m.architecture.type).toBe('moe')
    if (m.architecture.type === 'moe') {
      expect(m.architecture.activeParamCount).toBe(17_000_000_000)
    }
  })
})

describe('calculate — MiniMax M2.5/M2.7 (full-attention MoE + MTP-3) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!

  it('MiniMax M2.5 at 32k prompt: full-attention GQA KV across all 62 layers', () => {
    const m = MODELS.find(x => x.id === 'minimax-m2.5')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    // 2 × 8 × 128 × 2 × 62 × 32768
    expect(r.memory.kvCachePerRequest).toBe(2 * 8 * 128 * 2 * 62 * 32768)
  })

  it('MiniMax M2.5 decode: 10B active reads, 4× token rate from MTP depth 3', () => {
    const m = MODELS.find(x => x.id === 'minimax-m2.5')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    }
    const r = calculate(input)
    expect(r.memory.weights / 1e9).toBeCloseTo(460, 0)
    expect(r.memory.fits).toBe(false)
    const activeBytes = 10_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 2e9)
    const rNoMtp = calculate({ ...input, model: { ...m, numNextnLayers: 0 } })
    expect(r.perf['peak'].decode.aggregateTokensPerS).toBeCloseTo(
      rNoMtp.perf['peak'].decode.aggregateTokensPerS * 4, 6
    )
  })

  it('MiniMax M2.7 shares M2.5 geometry with 200k trained context', () => {
    const m25 = MODELS.find(x => x.id === 'minimax-m2.5')!
    const m27 = MODELS.find(x => x.id === 'minimax-m2.7')!
    expect(m27.layers).toBe(m25.layers)
    expect(m27.paramCount).toBe(m25.paramCount)
    expect(m27.maxContext).toBe(204800)
  })
})

describe('calculate — Kimi K2.5 / DeepSeek V3.1 (V3-family MLA) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!

  it('Kimi K2.5 at 32k prompt: MLA KV identical to K2 (same backbone)', () => {
    const m = MODELS.find(x => x.id === 'kimi-k2.5')!
    const input: CalcInput = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r = calculate(input)
    expect(r.memory.kvCachePerRequest).toBe(61 * (512 + 64) * 2 * 32768)
    expect(r.memory.weights / 1e12).toBeCloseTo(2.05, 1)
    expect(m.maxContext).toBe(262144)
    expect(m.nativeDtype).toBe('int4')
  })

  it('DeepSeek V3.1 matches V3 geometry and carries MTP depth 1', () => {
    const v3 = MODELS.find(x => x.id === 'deepseek-v3')!
    const v31 = MODELS.find(x => x.id === 'deepseek-v3.1')!
    const baseInput: Omit<CalcInput, 'model'> = {
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 32768, outputTokens: 0, concurrency: 1 }
    }
    const r3 = calculate({ ...baseInput, model: v3 })
    const r31 = calculate({ ...baseInput, model: v31 })
    expect(r31.memory.kvCachePerRequest).toBe(r3.memory.kvCachePerRequest)
    expect(v31.numNextnLayers).toBe(1)
  })

  it('DeepSeek V3/R1/V3.2 carry MTP depth 1 (num_nextn_predict_layers in config)', () => {
    for (const id of ['deepseek-v3', 'deepseek-r1', 'deepseek-v3.2']) {
      expect(MODELS.find(x => x.id === id)!.numNextnLayers, id).toBe(1)
    }
  })
})

describe('calculate — small dense additions (Qwen3-0.6B, MiMo-7B, Llama 3.1)', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const base = (id: string): CalcInput => ({
    accelerator: h100,
    acceleratorVariantId: 'sxm-80',
    model: MODELS.find(x => x.id === id)!,
    quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
    workload: { promptTokens: 8192, outputTokens: 512, concurrency: 1 }
  })

  it('Qwen3-0.6B: 1.2 GB weights fit; full-attention KV', () => {
    const r = calculate(base('qwen3-0.6b'))
    expect(r.memory.weights / 1e9).toBeCloseTo(1.19, 1)
    expect(r.memory.fits).toBe(true)
    // 2 × 8 × 128 × 2 × 28 × (8192 prompt + 512 output)
    expect(r.memory.kvCachePerRequest).toBe(2 * 8 * 128 * 2 * 28 * (8192 + 512))
  })

  it('MiMo-7B: 15.7 GB weights fit; carries MTP depth 1', () => {
    const r = calculate(base('mimo-7b'))
    expect(r.memory.weights / 1e9).toBeCloseTo(15.7, 0)
    expect(r.memory.fits).toBe(true)
    expect(MODELS.find(x => x.id === 'mimo-7b')!.numNextnLayers).toBe(1)
  })

  it('Llama 3.1 8B: 16.1 GB weights fit; 3.1 70B matches 3.3 70B geometry', () => {
    const r = calculate(base('llama-3.1-8b'))
    expect(r.memory.weights / 1e9).toBeCloseTo(16.1, 0)
    expect(r.memory.fits).toBe(true)
    const l31 = MODELS.find(x => x.id === 'llama-3.1-70b')!
    const l33 = MODELS.find(x => x.id === 'llama-3.3-70b')!
    expect(l31.paramCount).toBe(l33.paramCount)
    expect(l31.layers).toBe(l33.layers)
  })
})

describe('calculate — Nemotron Mamba2 hybrids integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const input = (id: string, promptTokens: number, outputTokens = 0): CalcInput => ({
    accelerator: h100,
    acceleratorVariantId: 'sxm-80',
    model: MODELS.find(x => x.id === id)!,
    quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
    workload: { promptTokens, outputTokens, concurrency: 1 }
  })

  it('Nemotron 3 Nano at 32k: KV from 6 attention blocks + fp32 Mamba2 state', () => {
    const r = calculate(input('nemotron-3-nano-30b-a3b', 32768))
    // Attention KV: 2 × 2 kvHeads × 128 × 2 bytes × (6 attn blocks × 32768)
    const kv = 2 * 2 * 128 * 2 * 6 * 32768
    // SSM state: 23 mamba blocks × 64 heads × 64 headDim × 128 stateSize × 4 (fp32)
    const state = 23 * 64 * 64 * 128 * 4
    expect(r.memory.kvCachePerRequest).toBe(kv + state)
    // 31.6B × 2 bytes ≈ 63 GB fits a single H100 SXM-80
    expect(r.memory.weights / 1e9).toBeCloseTo(63.2, 0)
    expect(r.memory.fits).toBe(true)
  })

  it('Nemotron 3 Nano decode reads 3.5B active params', () => {
    const r = calculate(input('nemotron-3-nano-30b-a3b', 2048, 512))
    const activeBytes = 3_500_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 1e9)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })

  it('Nemotron-H 56B at 8k: SSM state dominates KV; state bytes ignore KV quant (fp32 cache)', () => {
    const rFp16 = calculate(input('nemotron-h-56b', 8192))
    const kv = 2 * 8 * 128 * 2 * 10 * 8192          // 10 attention blocks
    const state = 54 * 256 * 64 * 256 * 4            // 54 mamba blocks, fp32
    expect(rFp16.memory.kvCachePerRequest).toBe(kv + state)
    // Halving KV dtype halves only the attention-KV term, not the SSM state.
    const rFp8 = calculate({
      ...input('nemotron-h-56b', 8192),
      quant: { weights: 'fp16', kv: 'fp8', activations: 'fp16' }
    })
    expect(rFp8.memory.kvCachePerRequest).toBe(kv / 2 + state)
  })

  it('Nemotron 3 Ultra: 1.12 TB weights at fp16; 55B active decode; MTP depth 1', () => {
    const m = MODELS.find(x => x.id === 'nemotron-3-ultra-550b-a55b')!
    const r = calculate(input('nemotron-3-ultra-550b-a55b', 2048, 512))
    expect(r.memory.weights / 1e12).toBeCloseTo(1.12, 1)
    expect(r.memory.fits).toBe(false)
    expect(m.numNextnLayers).toBe(1)
    const activeBytes = 55_000_000_000 * 2
    expect(r.perf['peak'].decode.bytesPerStep).toBeGreaterThan(activeBytes)
    expect(r.perf['peak'].decode.bytesPerStep).toBeLessThan(activeBytes + 5e9)
  })

  it('Nemotron 3 Super: block counts sum to layers; 12B active', () => {
    const m = MODELS.find(x => x.id === 'nemotron-3-super-120b-a12b')!
    expect(m.attention.type).toBe('mamba2-hybrid')
    if (m.attention.type === 'mamba2-hybrid') {
      expect(m.attention.numMambaLayers + m.attention.numFullLayers + m.attention.numFfnLayers)
        .toBe(m.layers)
    }
    if (m.architecture.type === 'moe') {
      expect(m.architecture.activeParamCount).toBe(12_000_000_000)
    }
  })

  it('throws when mamba2-hybrid block counts do not sum to layers', () => {
    const m = MODELS.find(x => x.id === 'nemotron-3-nano-30b-a3b')!
    const broken = { ...m, layers: m.layers + 1 }
    expect(() => calculate({ ...input('nemotron-3-nano-30b-a3b', 2048), model: broken })).toThrow()
  })
})

describe('calculate — Llama-3.3-Nemotron-Super 49B (partial attention) integration', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const input = (promptTokens: number, outputTokens = 0): CalcInput => ({
    accelerator: h100,
    acceleratorVariantId: 'sxm-80',
    model: MODELS.find(x => x.id === 'llama-3.3-nemotron-super-49b')!,
    quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
    workload: { promptTokens, outputTokens, concurrency: 1 }
  })

  it('at 32k prompt: KV cache counts only the 49 blocks with attention', () => {
    const r = calculate(input(32768))
    // 2 × 8 × 128 × 2 × (49 × 32768); the 31 NAS-pruned blocks contribute none
    expect(r.memory.kvCachePerRequest).toBe(2 * 8 * 128 * 2 * 49 * 32768)
  })

  it('weights 99.7 GB at fp16 do not fit a single H100', () => {
    const r = calculate(input(2048, 512))
    expect(r.memory.weights / 1e9).toBeCloseTo(99.7, 0)
    expect(r.memory.fits).toBe(false)
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })

  it('throws when partial numFullLayers exceeds model.layers', () => {
    const m = MODELS.find(x => x.id === 'llama-3.3-nemotron-super-49b')!
    const broken = { ...m, attention: { type: 'partial' as const, numFullLayers: m.layers + 1 } }
    expect(() => calculate({ ...input(2048), model: broken })).toThrow()
  })
})

describe('derivation — formulas match what the engine computed', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!
  const hgxH100 = SYSTEMS.find(s => s.id === 'hgx-h100-8')!

  it('MLA model: kv-per-token expression is the MLA formula, not GQA', () => {
    const k25 = MODELS.find(m => m.id === 'kimi-k2.5')!
    const r = calculate({
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: k25,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 0, concurrency: 1 }
    })
    const row = r.derivation.find(s => s.label === 'kv per token per request')!
    expect(row.expression).toContain('kv_lora_rank')
    expect(row.expression).not.toContain('kv_heads')
    // The value must be reproducible from the printed formula:
    // (512 + 64) × 2 bytes × 61 layers
    expect(row.value).toBe((512 + 64) * 2 * 61)
  })

  it('multi-device: time expressions include the comms term and comms rows exist', () => {
    const v3 = MODELS.find(m => m.id === 'deepseek-v3')!
    const r = calculate({
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: v3,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 64 },
      multiDevice: {
        system: hgxH100,
        parallelism: ['tp', 'ep'],
        parallelismDegrees: { tp: 8, ep: 8 }
      }
    })
    const t = r.derivation.find(s => s.label === 'prefill time @ peak')!
    expect(t.expression).toContain('comms')
    const comms = r.derivation.find(s => s.label === 'prefill comms bytes')!
    expect(comms.value).toBe(r.perf['peak'].prefill.commsBytes)
    expect(comms.value).toBeGreaterThan(0)
  })

  it('single-device: time expressions do NOT mention comms; flops/bytes rows match perf', () => {
    const llama = MODELS.find(m => m.id === 'llama-3.3-70b')!
    const r = calculate({
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: llama,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    })
    const t = r.derivation.find(s => s.label === 'prefill time @ peak')!
    expect(t.expression).not.toContain('comms')
    expect(r.derivation.find(s => s.label === 'prefill flops')!.value)
      .toBe(r.perf['peak'].prefill.flops)
    expect(r.derivation.find(s => s.label === 'prefill bytes (hbm)')!.value)
      .toBe(r.perf['peak'].prefill.bytes)
    expect(r.derivation.find(s => s.label === 'decode flops per step')!.value)
      .toBe(r.perf['peak'].decode.flopsPerStep)
    expect(r.derivation.find(s => s.label === 'decode bytes per step')!.value)
      .toBe(r.perf['peak'].decode.bytesPerStep)
  })

  it('MTP model: decode time expression shows the (1 + mtp_depth) division', () => {
    const m25 = MODELS.find(m => m.id === 'minimax-m2.5')!
    const r = calculate({
      accelerator: h100,
      acceleratorVariantId: 'sxm-80',
      model: m25,
      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
      workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
    })
    const t = r.derivation.find(s => s.label === 'decode time per token @ peak')!
    expect(t.expression).toContain('1 + mtp_depth')
  })
})

describe('models data — every entry produces finite results', () => {
  const h100 = ACCELERATORS.find(a => a.id === 'h100')!

  it('no model yields NaN/0 in memory or perf at 8k/512/c=4', () => {
    for (const m of MODELS) {
      const r = calculate({
        accelerator: h100,
        acceleratorVariantId: 'sxm-80',
        model: m,
        quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
        workload: { promptTokens: 8192, outputTokens: 512, concurrency: 4 }
      })
      expect(Number.isFinite(r.memory.total), `${m.id} memory.total`).toBe(true)
      expect(r.memory.weights, `${m.id} weights`).toBeGreaterThan(0)
      expect(r.memory.kvCachePerRequest, `${m.id} kv`).toBeGreaterThan(0)
      for (const [tier, p] of Object.entries(r.perf)) {
        expect(Number.isFinite(p.prefill.timeS), `${m.id} ${tier} prefill.timeS`).toBe(true)
        expect(p.prefill.timeS, `${m.id} ${tier} prefill.timeS`).toBeGreaterThan(0)
        expect(Number.isFinite(p.decode.timePerTokenS), `${m.id} ${tier} tpot`).toBe(true)
        expect(p.decode.timePerTokenS, `${m.id} ${tier} tpot`).toBeGreaterThan(0)
      }
    }
  })
})

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

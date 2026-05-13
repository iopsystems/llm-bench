import { describe, it, expect } from 'vitest'
import { calculate } from '../../src/engine/calc'
import { testInput } from '../fixtures'
import { GPUS } from '../../src/data/gpus'
import { MODELS } from '../../src/data/models'
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
    expect(() => calculate({ ...testInput, gpuVariantId: 'nope' })).toThrow()
  })
})

describe('calculate — real data integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const llama70b = MODELS.find(m => m.id === 'llama-3.3-70b')!

  const input: CalcInput = {
    gpu: h100,
    gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const mistral = MODELS.find(m => m.id === 'mistral-7b-v0.1')!

  it('Mistral 7B at 32k prompt: KV cache bounded by 4k window, not 32k', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const mixtral = MODELS.find(m => m.id === 'mixtral-8x7b')!

  it('Mixtral 8x7B on H100 SXM-80: weights use total params, decode bytes use active', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const dsv2 = MODELS.find(m => m.id === 'deepseek-v2')!

  it('DeepSeek-V2 at 32k prompt: KV cache uses MLA latent formula', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const gemma27b = MODELS.find(m => m.id === 'gemma-3-27b')!

  it('Gemma 3 27B at 8k prompt: KV cache uses hybrid formula (~3.8× smaller than full attention)', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const dsv3 = MODELS.find(m => m.id === 'deepseek-v3')!

  it('DeepSeek V3 at 32k prompt: MLA KV cache matches V3 geometry', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const mixtral22b = MODELS.find(m => m.id === 'mixtral-8x22b')!

  it('Mixtral 8x22B on H100 SXM-80: weights 282 GB do not fit; decode uses 39B active', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const k2 = MODELS.find(m => m.id === 'kimi-k2')!

  it('Kimi K2 at 32k prompt: MLA KV cache uses identical formula to DeepSeek V3 (same layers + MLA dims)', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const glm = MODELS.find(m => m.id === 'glm-4.5-air')!

  it('GLM-4.5-Air at 32k prompt: regular GQA KV cache with 12:1 head ratio', () => {
    const input: CalcInput = {
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
      gpu: h100,
      gpuVariantId: 'sxm-80',
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
  const h100 = GPUS.find(g => g.id === 'h100')!
  const v3 = MODELS.find(m => m.id === 'deepseek-v3')!
  const v32 = MODELS.find(m => m.id === 'deepseek-v3.2')!

  const baseInput: Omit<CalcInput, 'model'> = {
    gpu: h100,
    gpuVariantId: 'sxm-80',
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

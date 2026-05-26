import { describe, it, expect } from 'vitest'
import {
  activeParams,
  kvBytesPerTokenPerLayer,
  attentionDim,
  attendedSeqlenSummedOverLayers,
  linearAttentionStateBytes,
  linearAttentionFlopsPerToken
} from '../../src/engine/memory'
import type { ModelArch } from '../../src/engine/types'

describe('activeParams', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
    layers: 2, hiddenDim: 4, intermediateDim: 8,
    numHeads: 2, numKvHeads: 1, headDim: 2, vocabSize: 100,
    paramCount: 1000,
    maxContext: 8192,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('returns paramCount for dense models', () => {
    expect(activeParams(base)).toBe(1000)
  })

  it('returns activeParamCount for MoE models', () => {
    const moe: ModelArch = {
      ...base,
      paramCount: 8000,
      maxContext: 8192,
      architecture: {
        type: 'moe',
        numExperts: 8,
        numExpertsActive: 2,
        numSharedExperts: 0,
        activeParamCount: 2000
      }
    }
    expect(activeParams(moe)).toBe(2000)
  })
})

describe('kvBytesPerTokenPerLayer', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    maxContext: 8192,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('GQA / full attention: 2 × kv_heads × head_dim × bytes (no layers factor)', () => {
    // 2 × 2 × 8 × 2 (fp16) = 64
    expect(kvBytesPerTokenPerLayer(base, 'fp16')).toBe(64)
  })

  it('sliding window uses same GQA formula', () => {
    const sliding: ModelArch = {
      ...base,
      attention: { type: 'sliding', window: 50 }
    }
    expect(kvBytesPerTokenPerLayer(sliding, 'fp16')).toBe(64)
  })

  it('MLA: (kv_lora + rope) × bytes (no factor of 2, no layers factor)', () => {
    const mla: ModelArch = {
      ...base,
      attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8, qkNopeHeadDim: 8, vHeadDim: 8 }
    }
    // (32 + 8) × 2 = 80
    expect(kvBytesPerTokenPerLayer(mla, 'fp16')).toBe(80)
  })

  it('mla-dsa: same as MLA (DSA does not change KV size)', () => {
    const dsa: ModelArch = {
      ...base,
      attention: { type: 'mla-dsa', kvLoraRank: 32, qkRopeHeadDim: 8, qkNopeHeadDim: 8, vHeadDim: 8, topK: 16 }
    }
    // (32 + 8) × 2 = 80
    expect(kvBytesPerTokenPerLayer(dsa, 'fp16')).toBe(80)
  })
})

describe('attentionDim', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    maxContext: 8192,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('returns numHeads × headDim for full attention', () => {
    // 8 × 8 = 64 (intentionally different from hiddenDim 16, so the test
    // would fail if the helper accidentally returned hiddenDim — c.f. PR #91)
    expect(attentionDim(base)).toBe(64)
  })

  it('returns numHeads × headDim for sliding window', () => {
    const sliding: ModelArch = {
      ...base,
      attention: { type: 'sliding', window: 50 }
    }
    expect(attentionDim(sliding)).toBe(64)
  })

  it('returns kvLoraRank + qkRopeHeadDim for MLA', () => {
    const mla: ModelArch = {
      ...base,
      attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8, qkNopeHeadDim: 8, vHeadDim: 8 }
    }
    expect(attentionDim(mla)).toBe(40)
  })

  it('mla-dsa: same as MLA (DSA does not change attention representation)', () => {
    const dsa: ModelArch = {
      ...base,
      attention: { type: 'mla-dsa', kvLoraRank: 32, qkRopeHeadDim: 8, qkNopeHeadDim: 8, vHeadDim: 8, topK: 16 }
    }
    expect(attentionDim(dsa)).toBe(40)
  })
})

describe('attendedSeqlenSummedOverLayers', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    maxContext: 8192,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }

  it('full: layers × seqlen', () => {
    expect(attendedSeqlenSummedOverLayers(base, 100)).toBe(400) // 4 × 100
    expect(attendedSeqlenSummedOverLayers(base, 0)).toBe(0)
  })

  it('sliding: layers × min(seqlen, window)', () => {
    const m: ModelArch = { ...base, attention: { type: 'sliding', window: 50 } }
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(200) // 4 × 50
    expect(attendedSeqlenSummedOverLayers(m, 30)).toBe(120)  // 4 × 30
    expect(attendedSeqlenSummedOverLayers(m, 50)).toBe(200)  // 4 × 50
  })

  it('mla: layers × seqlen (dimensional reduction is in attentionDim, not seqlen)', () => {
    const m: ModelArch = { ...base, attention: { type: 'mla', kvLoraRank: 32, qkRopeHeadDim: 8, qkNopeHeadDim: 8, vHeadDim: 8 } }
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(400) // 4 × 100
  })

  it('hybrid: numSliding × min(seqlen, window) + numGlobal × seqlen', () => {
    const m: ModelArch = {
      ...base,
      layers: 6,
      attention: {
        type: 'hybrid', slidingWindow: 50,
        numSlidingLayers: 5, numGlobalLayers: 1
      }
    }
    // seqlen > window: 5 × min(100, 50) + 1 × 100 = 250 + 100 = 350
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(350)
    // seqlen < window: 5 × 30 + 1 × 30 = 180 (sliding cap inactive)
    expect(attendedSeqlenSummedOverLayers(m, 30)).toBe(180)
    // seqlen == window: 5 × 50 + 1 × 50 = 300
    expect(attendedSeqlenSummedOverLayers(m, 50)).toBe(300)
  })

  it('hybrid: throws when numSlidingLayers + numGlobalLayers ≠ model.layers', () => {
    const m: ModelArch = {
      ...base,
      layers: 6,
      attention: {
        type: 'hybrid', slidingWindow: 50,
        numSlidingLayers: 4, numGlobalLayers: 1  // 4+1=5 ≠ 6
      }
    }
    expect(() => attendedSeqlenSummedOverLayers(m, 100)).toThrow(/sum to model\.layers/)
  })

  it('mla-dsa: layers × min(seqlen, topK)', () => {
    const m: ModelArch = {
      ...base,
      attention: { type: 'mla-dsa', kvLoraRank: 32, qkRopeHeadDim: 8, qkNopeHeadDim: 8, vHeadDim: 8, topK: 50 }
    }
    // seqlen > topK: 4 × 50 = 200
    expect(attendedSeqlenSummedOverLayers(m, 100)).toBe(200)
    // seqlen < topK: 4 × 30 = 120
    expect(attendedSeqlenSummedOverLayers(m, 30)).toBe(120)
    // seqlen == topK: 4 × 50 = 200
    expect(attendedSeqlenSummedOverLayers(m, 50)).toBe(200)
  })
})

describe('linear-mla-hybrid branches in existing helpers', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
    layers: 4, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    maxContext: 8192,
    numNextnLayers: 0,
    attention: {
      type: 'linear-mla-hybrid',
      kvLoraRank: 8, qkRopeHeadDim: 2,
      qkNopeHeadDim: 2, vHeadDim: 2,
      numLinearLayers: 3, numFullLayers: 1,
      numLinearHeads: 2, linearHeadDim: 4
    },
    architecture: { type: 'dense' }
  }

  it('kvBytesPerTokenPerLayer: returns the per-full-MLA-layer-per-token bytes', () => {
    // (kvLoraRank + qkRopeHeadDim) × 2 (fp16) = (8 + 2) × 2 = 20
    expect(kvBytesPerTokenPerLayer(base, 'fp16')).toBe(20)
  })

  it('attentionDim: returns the MLA absorbed-form attention dim', () => {
    // kvLoraRank + qkRopeHeadDim = 10
    expect(attentionDim(base)).toBe(10)
  })

  it('attendedSeqlenSummedOverLayers: returns numFullLayers × seqlen', () => {
    expect(attendedSeqlenSummedOverLayers(base, 100)).toBe(100)
    expect(attendedSeqlenSummedOverLayers(base, 30)).toBe(30)
  })

  it('attendedSeqlenSummedOverLayers throws when layer counts do not sum to model.layers', () => {
    const m: ModelArch = {
      ...base,
      attention: {
        type: 'linear-mla-hybrid',
        kvLoraRank: 8, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2,
        numLinearLayers: 2, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 4
      }
    }
    expect(() => attendedSeqlenSummedOverLayers(m, 100)).toThrow(/sum to model\.layers/)
  })
})

describe('linearAttentionStateBytes', () => {
  it('returns 0 for non-linear attention types', () => {
    const m: ModelArch = {
      id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
      layers: 4, hiddenDim: 16, intermediateDim: 64,
      numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
      paramCount: 1000,
      maxContext: 8192,
      numNextnLayers: 0,
      attention: { type: 'full' },
      architecture: { type: 'dense' }
    }
    expect(linearAttentionStateBytes(m, 'fp16')).toBe(0)
  })

  it('returns numLinearLayers × numLinearHeads × linearHeadDim² × bytes for linear-mla-hybrid', () => {
    const m: ModelArch = {
      id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
      layers: 4, hiddenDim: 16, intermediateDim: 64,
      numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
      paramCount: 1000,
      maxContext: 8192,
      numNextnLayers: 0,
      attention: {
        type: 'linear-mla-hybrid',
        kvLoraRank: 8, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2,
        numLinearLayers: 3, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 4
      },
      architecture: { type: 'dense' }
    }
    // 3 × 2 × 4² × 2 (fp16) = 192
    expect(linearAttentionStateBytes(m, 'fp16')).toBe(192)
  })
})

describe('csa-hca-hybrid branches in existing helpers', () => {
  const base: ModelArch = {
    id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
    layers: 3, hiddenDim: 16, intermediateDim: 64,
    numHeads: 8, numKvHeads: 1, headDim: 8, vocabSize: 100,
    paramCount: 1000,
    maxContext: 8192,
    numNextnLayers: 0,
    attention: {
      type: 'csa-hca-hybrid',
      numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 1,
      slidingWindow: 2,
      csaCompressionM: 2, csaTopK: 3,
      csaIndexerHeads: 2, csaIndexerHeadDim: 2,
      hcaCompressionM: 4
    },
    architecture: { type: 'dense' }
  }

  it('kvBytesPerTokenPerLayer: 2 × numKvHeads × headDim × bytes (MQA-style)', () => {
    // 2 × 1 × 8 × 2 (fp16) = 32
    expect(kvBytesPerTokenPerLayer(base, 'fp16')).toBe(32)
  })

  it('attentionDim: numHeads × headDim (full Q-head MQA)', () => {
    // 8 × 8 = 64
    expect(attentionDim(base)).toBe(64)
  })

  it('attendedSeqlenSummedOverLayers (forKv=true): storage formula', () => {
    // seqlen=20:
    // sliding contrib: 1 × min(20, 2) = 2
    // CSA contrib:     1 × (20/2 + 2) = 12
    // HCA contrib:     1 × (20/4 + 2) = 7
    // total = 21
    expect(attendedSeqlenSummedOverLayers(base, 20, true)).toBe(21)
  })

  it('attendedSeqlenSummedOverLayers (forKv=false default): compute formula uses csaTopK for CSA', () => {
    // seqlen=20:
    // sliding contrib: 1 × min(20, 2) = 2
    // CSA contrib:     1 × (csaTopK=3 + 2) = 5
    // HCA contrib:     1 × (20/4 + 2) = 7
    // total = 14
    expect(attendedSeqlenSummedOverLayers(base, 20)).toBe(14)
  })

  it('attendedSeqlenSummedOverLayers throws when layer counts do not sum to model.layers', () => {
    const m: ModelArch = {
      ...base,
      attention: {
        type: 'csa-hca-hybrid',
        numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 2,  // 1+1+2=4 ≠ 3
        slidingWindow: 2,
        csaCompressionM: 2, csaTopK: 3,
        csaIndexerHeads: 2, csaIndexerHeadDim: 2,
        hcaCompressionM: 4
      }
    }
    expect(() => attendedSeqlenSummedOverLayers(m, 20)).toThrow(/sum to model\.layers/)
  })
})

describe('linearAttentionFlopsPerToken', () => {
  it('returns 0 for non-linear attention types', () => {
    const m: ModelArch = {
      id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
      layers: 4, hiddenDim: 16, intermediateDim: 64,
      numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
      paramCount: 1000,
      maxContext: 8192,
      numNextnLayers: 0,
      attention: { type: 'full' },
      architecture: { type: 'dense' }
    }
    expect(linearAttentionFlopsPerToken(m)).toBe(0)
  })

  it('returns 2 × numLinearLayers × numLinearHeads × linearHeadDim² for linear-mla-hybrid', () => {
    const m: ModelArch = {
      id: 't', name: 'Test', family: 'test', publisher: 'test', releaseDate: '2025-01', nativeDtype: 'bf16',
      layers: 4, hiddenDim: 16, intermediateDim: 64,
      numHeads: 8, numKvHeads: 2, headDim: 8, vocabSize: 100,
      paramCount: 1000,
      maxContext: 8192,
      numNextnLayers: 0,
      attention: {
        type: 'linear-mla-hybrid',
        kvLoraRank: 8, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2,
        numLinearLayers: 3, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 4
      },
      architecture: { type: 'dense' }
    }
    // 2 × 3 × 2 × 4² = 192
    expect(linearAttentionFlopsPerToken(m)).toBe(192)
  })
})

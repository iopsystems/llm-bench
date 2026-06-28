import type { ModelArch } from '../engine/types'

// Architecture fields sourced from HuggingFace config.json per model.
// paramCount taken from each model's official card.
export const MODELS: ModelArch[] = [
  // === Qwen3 dense series ===
  {
    id: 'qwen3-0.6b', name: 'Qwen3 0.6B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 28, hiddenDim: 1024, intermediateDim: 3072,
    numHeads: 16, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 596_000_000,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-1.7b', name: 'Qwen3 1.7B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 28, hiddenDim: 2048, intermediateDim: 6144,
    numHeads: 16, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 1_720_000_000,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-4b', name: 'Qwen3 4B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 36, hiddenDim: 2560, intermediateDim: 9728,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 4_020_000_000,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-8b', name: 'Qwen3 8B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 36, hiddenDim: 4096, intermediateDim: 12288,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 8_190_000_000,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-14b', name: 'Qwen3 14B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 40, hiddenDim: 5120, intermediateDim: 17408,
    numHeads: 40, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 14_770_000_000,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-32b', name: 'Qwen3 32B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 64, hiddenDim: 5120, intermediateDim: 25600,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 32_760_000_000,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  // === Qwen3 MoE ===
  {
    id: 'qwen3-30b-a3b', name: 'Qwen3-30B-A3B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 48, hiddenDim: 2048, intermediateDim: 768,
    numHeads: 32, numKvHeads: 4, headDim: 128, vocabSize: 151936,
    paramCount: 30_500_000_000,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 8,
      numSharedExperts: 0,
      activeParamCount: 3_300_000_000
    }
  },
  {
    id: 'qwen3-235b-a22b', name: 'Qwen3-235B-A22B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 94, hiddenDim: 4096, intermediateDim: 1536,
    numHeads: 64, numKvHeads: 4, headDim: 128, vocabSize: 151936,
    paramCount: 235_000_000_000,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 8,
      numSharedExperts: 0,
      activeParamCount: 22_000_000_000
    }
  },
  // === Qwen3 Coder ===
  {
    id: 'qwen3-coder-30b-a3b', name: 'Qwen3-Coder-30B-A3B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-07',
    nativeDtype: 'bf16',
    layers: 48, hiddenDim: 2048, intermediateDim: 768,
    numHeads: 32, numKvHeads: 4, headDim: 128, vocabSize: 151936,
    paramCount: 30_500_000_000,
    maxContext: 262144,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 8,
      numSharedExperts: 0,
      activeParamCount: 3_300_000_000
    }
  },
  {
    id: 'qwen3-coder-480b-a35b', name: 'Qwen3-Coder-480B-A35B', family: 'qwen3',
    publisher: 'Alibaba', releaseDate: '2025-07',
    nativeDtype: 'bf16',
    layers: 62, hiddenDim: 6144, intermediateDim: 2560,
    numHeads: 96, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 480_000_000_000,
    maxContext: 262144,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 160,
      numExpertsActive: 8,
      numSharedExperts: 0,
      activeParamCount: 35_000_000_000
    }
  },
  // === Qwen3.5 dense series (Gated DeltaNet + Gated Attention hybrid) ===
  {
    id: 'qwen3.5-0.8b', name: 'Qwen3.5-0.8B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 24, hiddenDim: 1024, intermediateDim: 3584,
    numHeads: 8, numKvHeads: 2, headDim: 256, vocabSize: 248320,
    paramCount: 830_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 18, numFullLayers: 6,
      numDeltaNetHeads: 16, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3.5-2b', name: 'Qwen3.5-2B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 24, hiddenDim: 2048, intermediateDim: 6144,
    numHeads: 8, numKvHeads: 2, headDim: 256, vocabSize: 248320,
    paramCount: 2_240_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 18, numFullLayers: 6,
      numDeltaNetHeads: 16, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3.5-4b', name: 'Qwen3.5-4B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 32, hiddenDim: 2560, intermediateDim: 9216,
    numHeads: 16, numKvHeads: 4, headDim: 256, vocabSize: 248320,
    paramCount: 4_240_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 24, numFullLayers: 8,
      numDeltaNetHeads: 32, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3.5-8b', name: 'Qwen3.5-8B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 32, hiddenDim: 4096, intermediateDim: 12288,
    numHeads: 16, numKvHeads: 2, headDim: 256, vocabSize: 248320,
    paramCount: 8_460_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 24, numFullLayers: 8,
      numDeltaNetHeads: 32, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3.5-9b', name: 'Qwen3.5-9B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 32, hiddenDim: 4096, intermediateDim: 12288,
    numHeads: 16, numKvHeads: 4, headDim: 256, vocabSize: 248320,
    paramCount: 9_300_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 24, numFullLayers: 8,
      numDeltaNetHeads: 32, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: { type: 'dense' }
  },
  // === Qwen3.5 MoE series (Gated DeltaNet + Gated Attention hybrid) ===
  {
    id: 'qwen3.5-27b', name: 'Qwen3.5-27B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 64, hiddenDim: 5120, intermediateDim: 17408,
    numHeads: 24, numKvHeads: 4, headDim: 256, vocabSize: 248320,
    paramCount: 27_000_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 48, numFullLayers: 16,
      numDeltaNetHeads: 64, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 8,
      numSharedExperts: 0,
      activeParamCount: 2_700_000_000
    }
  },
  {
    id: 'qwen3.5-35b-a3b', name: 'Qwen3.5-35B-A3B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 40, hiddenDim: 2048, intermediateDim: 768,
    numHeads: 16, numKvHeads: 2, headDim: 256, vocabSize: 248320,
    paramCount: 35_000_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 30, numFullLayers: 10,
      numDeltaNetHeads: 64, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 3_300_000_000
    }
  },
  {
    id: 'qwen3.5-122b-a10b', name: 'Qwen3.5-122B-A10B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 48, hiddenDim: 3072, intermediateDim: 1024,
    numHeads: 32, numKvHeads: 2, headDim: 256, vocabSize: 248320,
    paramCount: 122_000_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 36, numFullLayers: 12,
      numDeltaNetHeads: 64, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 10_000_000_000
    }
  },
  {
    id: 'qwen3.5-397b-a17b', name: 'Qwen3.5-397B-A17B', family: 'qwen3.5',
    publisher: 'Alibaba', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 60, hiddenDim: 4096, intermediateDim: 1024,
    numHeads: 32, numKvHeads: 2, headDim: 256, vocabSize: 248320,
    paramCount: 397_000_000_000,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'delta-hybrid',
      numDeltaNetLayers: 45, numFullLayers: 15,
      numDeltaNetHeads: 64, deltaHeadDim: 128,
      ropeDim: 64
    },
    architecture: {
      type: 'moe',
      numExperts: 512,
      numExpertsActive: 10,
      numSharedExperts: 1,
      activeParamCount: 17_000_000_000
    }
  },
  // === Llama ===
  {
    id: 'llama-3.1-8b', name: 'Llama 3.1 8B', family: 'llama-3',
    publisher: 'Meta', releaseDate: '2024-07',
    nativeDtype: 'bf16',
    layers: 32, hiddenDim: 4096, intermediateDim: 14336,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 8_030_261_248,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'llama-3.1-70b', name: 'Llama 3.1 70B', family: 'llama-3',
    publisher: 'Meta', releaseDate: '2024-07',
    nativeDtype: 'bf16',
    layers: 80, hiddenDim: 8192, intermediateDim: 28672,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 70_553_706_496,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'llama-3.3-70b', name: 'Llama 3.3 70B', family: 'llama-3',
    publisher: 'Meta', releaseDate: '2024-12',
    nativeDtype: 'bf16',
    layers: 80, hiddenDim: 8192, intermediateDim: 28672,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 70_553_706_496,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'llama-3.1-405b', name: 'Llama 3.1 405B', family: 'llama-3',
    publisher: 'Meta', releaseDate: '2024-07',
    nativeDtype: 'bf16',
    layers: 126, hiddenDim: 16384, intermediateDim: 53248,
    numHeads: 128, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 405_853_356_032,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  // === Llama 4 ===
  // Llama 4 interleaves chunked local attention (attention_chunk_size 8192,
  // 3 of every 4 layers) with full NoPE attention on every 4th layer. Chunked
  // attention bounds per-layer KV exactly like a sliding window of the same
  // size, so it's modeled as `hybrid` with slidingWindow 8192.
  {
    id: 'llama-4-scout', name: 'Llama 4 Scout 109B-A17B', family: 'llama-4',
    publisher: 'Meta', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 48, hiddenDim: 5120, intermediateDim: 8192,
    numHeads: 40, numKvHeads: 8, headDim: 128, vocabSize: 202048,
    paramCount: 109_000_000_000,
    maxContext: 10485760,
    numNextnLayers: 0,
    attention: { type: 'hybrid', slidingWindow: 8192, numSlidingLayers: 36, numGlobalLayers: 12 },
    architecture: {
      type: 'moe',
      numExperts: 16,
      numExpertsActive: 1,
      numSharedExperts: 1,
      activeParamCount: 17_000_000_000
    }
  },
  // Maverick routes through MoE on every other layer only (interleave step 2);
  // the remaining layers are dense FFN (16384 inner dim). The uniform-MoE
  // schema can't express that split — paramCount/activeParamCount from the
  // model card carry the memory and decode-cost truth.
  {
    id: 'llama-4-maverick', name: 'Llama 4 Maverick 400B-A17B', family: 'llama-4',
    publisher: 'Meta', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 48, hiddenDim: 5120, intermediateDim: 8192,
    numHeads: 40, numKvHeads: 8, headDim: 128, vocabSize: 202048,
    paramCount: 400_000_000_000,
    maxContext: 1048576,
    numNextnLayers: 0,
    attention: { type: 'hybrid', slidingWindow: 8192, numSlidingLayers: 36, numGlobalLayers: 12 },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 1,
      numSharedExperts: 1,
      activeParamCount: 17_000_000_000
    }
  },
  // === Gemma 3 ===
  {
    id: 'gemma-3-12b', name: 'Gemma 3 12B', family: 'gemma-3',
    publisher: 'Google', releaseDate: '2025-03',
    nativeDtype: 'bf16',
    layers: 48, hiddenDim: 3840, intermediateDim: 15360,
    numHeads: 16, numKvHeads: 8, headDim: 256, vocabSize: 262144,
    paramCount: 12_187_000_000,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'hybrid', slidingWindow: 1024, numSlidingLayers: 40, numGlobalLayers: 8 },
    architecture: { type: 'dense' }
  },
  {
    id: 'gemma-3-27b', name: 'Gemma 3 27B', family: 'gemma-3',
    publisher: 'Google', releaseDate: '2025-03',
    nativeDtype: 'bf16',
    layers: 62, hiddenDim: 5376, intermediateDim: 21504,
    numHeads: 32, numKvHeads: 16, headDim: 128, vocabSize: 262144,
    paramCount: 27_009_000_000,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'hybrid', slidingWindow: 1024, numSlidingLayers: 52, numGlobalLayers: 10 },
    architecture: { type: 'dense' }
  },
  // === Mistral ===
  {
    id: 'mistral-7b-v0.1', name: 'Mistral 7B v0.1', family: 'mistral',
    publisher: 'Mistral AI', releaseDate: '2023-09',
    nativeDtype: 'bf16',
    layers: 32, hiddenDim: 4096, intermediateDim: 14336,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
    paramCount: 7_241_732_096,
    maxContext: 32768,
    numNextnLayers: 0,
    attention: { type: 'sliding', window: 4096 },
    architecture: { type: 'dense' }
  },
  {
    id: 'mixtral-8x7b', name: 'Mixtral 8x7B v0.1', family: 'mistral',
    publisher: 'Mistral AI', releaseDate: '2023-12',
    nativeDtype: 'bf16',
    layers: 32, hiddenDim: 4096, intermediateDim: 14336,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
    paramCount: 46_702_792_704,
    maxContext: 32768,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 8,
      numExpertsActive: 2,
      numSharedExperts: 0,
      activeParamCount: 12_879_204_352
    }
  },
  {
    id: 'mixtral-8x22b', name: 'Mixtral 8x22B v0.1', family: 'mistral',
    publisher: 'Mistral AI', releaseDate: '2024-04',
    nativeDtype: 'bf16',
    layers: 56, hiddenDim: 6144, intermediateDim: 16384,
    numHeads: 48, numKvHeads: 8, headDim: 128, vocabSize: 32000,
    paramCount: 141_000_000_000,
    maxContext: 65536,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 8,
      numExpertsActive: 2,
      numSharedExperts: 0,
      activeParamCount: 39_000_000_000
    }
  },
  {
    id: 'mistral-small-3.1-24b', name: 'Mistral Small 3.1 24B', family: 'mistral',
    publisher: 'Mistral AI', releaseDate: '2025-03',
    nativeDtype: 'bf16',
    layers: 40, hiddenDim: 5120, intermediateDim: 32768,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 131072,
    paramCount: 23_572_403_200,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'mistral-small-3.2-24b', name: 'Mistral Small 3.2 24B', family: 'mistral',
    publisher: 'Mistral AI', releaseDate: '2025-06',
    nativeDtype: 'bf16',
    layers: 40, hiddenDim: 5120, intermediateDim: 32768,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 131072,
    paramCount: 23_572_403_200,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'magistral-small', name: 'Magistral Small 24B', family: 'mistral',
    publisher: 'Mistral AI', releaseDate: '2025-06',
    nativeDtype: 'bf16',
    layers: 40, hiddenDim: 5120, intermediateDim: 32768,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 131072,
    paramCount: 23_572_403_200,
    maxContext: 40960,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'mistral-large-2', name: 'Mistral Large 2 123B', family: 'mistral',
    publisher: 'Mistral AI', releaseDate: '2024-07',
    nativeDtype: 'bf16',
    layers: 88, hiddenDim: 12288, intermediateDim: 28672,
    numHeads: 96, numKvHeads: 8, headDim: 128, vocabSize: 32768,
    paramCount: 122_610_524_160,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  // === DeepSeek ===
  {
    id: 'deepseek-v2', name: 'DeepSeek-V2', family: 'deepseek',
    publisher: 'DeepSeek', releaseDate: '2024-05',
    nativeDtype: 'bf16',
    layers: 60, hiddenDim: 5120, intermediateDim: 12288,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 102400,
    paramCount: 236_000_000_000,
    maxContext: 163840,
    numNextnLayers: 0,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128 },
    architecture: {
      type: 'moe',
      numExperts: 160,
      numExpertsActive: 6,
      numSharedExperts: 2,
      activeParamCount: 21_000_000_000
    }
  },
  {
    id: 'deepseek-v3', name: 'DeepSeek-V3', family: 'deepseek',
    publisher: 'DeepSeek', releaseDate: '2024-12',
    nativeDtype: 'fp8',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
    paramCount: 671_000_000_000,
    maxContext: 163840,
    numNextnLayers: 1,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128 },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 37_000_000_000
    }
  },
  {
    id: 'deepseek-r1', name: 'DeepSeek-R1', family: 'deepseek',
    publisher: 'DeepSeek', releaseDate: '2025-01',
    nativeDtype: 'fp8',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
    paramCount: 671_000_000_000,
    maxContext: 163840,
    numNextnLayers: 1,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128 },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 37_000_000_000
    }
  },
  {
    id: 'deepseek-v3.1', name: 'DeepSeek-V3.1', family: 'deepseek',
    publisher: 'DeepSeek', releaseDate: '2025-08',
    nativeDtype: 'fp8',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
    paramCount: 671_000_000_000,
    maxContext: 163840,
    numNextnLayers: 1,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128 },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 37_000_000_000
    }
  },
  {
    id: 'deepseek-v3.2', name: 'DeepSeek-V3.2', family: 'deepseek',
    publisher: 'DeepSeek', releaseDate: '2025-09',
    nativeDtype: 'fp8',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
    paramCount: 671_000_000_000,
    maxContext: 163840,
    numNextnLayers: 1,
    attention: { type: 'mla-dsa', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128, topK: 2048 },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 37_000_000_000
    }
  },
  {
    id: 'deepseek-v4-flash', name: 'DeepSeek-V4-Flash', family: 'deepseek',
    publisher: 'DeepSeek', releaseDate: '2026-03',
    nativeDtype: 'fp8',
    layers: 43, hiddenDim: 4096, intermediateDim: 2048, vocabSize: 129280,
    numHeads: 64, numKvHeads: 1, headDim: 512,
    paramCount: 284_000_000_000,
    maxContext: 1048576,
    numNextnLayers: 1,
    attention: {
      type: 'csa-hca-hybrid',
      numSlidingLayers: 2, numCsaLayers: 21, numHcaLayers: 20,
      slidingWindow: 128,
      csaCompressionM: 4, csaTopK: 512,
      csaIndexerHeads: 64, csaIndexerHeadDim: 128,
      hcaCompressionM: 128
    },
    architecture: {
      type: 'moe',
      numExperts: 256, numExpertsActive: 6,
      numSharedExperts: 1,
      activeParamCount: 13_000_000_000
    }
  },
  {
    id: 'deepseek-v4-pro', name: 'DeepSeek-V4-Pro', family: 'deepseek',
    publisher: 'DeepSeek', releaseDate: '2026-03',
    nativeDtype: 'fp8',
    layers: 61, hiddenDim: 7168, intermediateDim: 3072, vocabSize: 129280,
    numHeads: 128, numKvHeads: 1, headDim: 512,
    paramCount: 1_600_000_000_000,
    maxContext: 1048576,
    numNextnLayers: 1,
    attention: {
      type: 'csa-hca-hybrid',
      numSlidingLayers: 0, numCsaLayers: 30, numHcaLayers: 31,
      slidingWindow: 128,
      csaCompressionM: 4, csaTopK: 1024,
      csaIndexerHeads: 64, csaIndexerHeadDim: 128,
      hcaCompressionM: 128
    },
    architecture: {
      type: 'moe',
      numExperts: 384, numExpertsActive: 6,
      numSharedExperts: 1,
      activeParamCount: 49_000_000_000
    }
  },
  // === Moonshot / Kimi ===
  {
    id: 'kimi-k2', name: 'Kimi K2', family: 'kimi',
    publisher: 'Moonshot AI', releaseDate: '2025-07',
    nativeDtype: 'bf16',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 64, numKvHeads: 64, headDim: 192, vocabSize: 163840,
    paramCount: 1_026_000_000_000,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128 },
    architecture: {
      type: 'moe',
      numExperts: 384,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 32_000_000_000
    }
  },
  // K2.5 reuses the K2 text backbone (same 61-layer MLA MoE) continually
  // pretrained to 256k context, shipping int4 (W4A16) weights. The 400M-param
  // vision encoder is out of scope for this text-decode calc; paramCount is
  // the text tower, matching the card's "1T total / 32B activated".
  {
    id: 'kimi-k2.5', name: 'Kimi K2.5', family: 'kimi',
    publisher: 'Moonshot AI', releaseDate: '2026-01',
    nativeDtype: 'int4',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 64, numKvHeads: 64, headDim: 192, vocabSize: 163840,
    paramCount: 1_026_000_000_000,
    maxContext: 262144,
    numNextnLayers: 0,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128 },
    architecture: {
      type: 'moe',
      numExperts: 384,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 32_000_000_000
    }
  },
  // K2.7-Code (moonshotai/Kimi-K2.7-Code, public, created 2026-06-11): same K2/K2.5
  // text backbone — config text_config is DeepseekV3ForCausalLM, model_type kimi_k25,
  // identical 61-layer MLA MoE (kv_lora 512, qk_rope 64, qk_nope 128, v 128; 384
  // experts / 8 active / 1 shared). Deltas vs K2.5: ships **bf16** weights (config
  // dtype bfloat16) rather than int4, at 256k context (generation max_length 262144).
  // Card states 1T total / 32B activated; paramCount = the text tower (1.026T, same
  // as K2/K2.5), excluding the multimodal vision encoder per the K2.5 convention.
  {
    id: 'kimi-k2.7-code', name: 'Kimi K2.7 Code', family: 'kimi',
    publisher: 'Moonshot AI', releaseDate: '2026-06',
    nativeDtype: 'bf16',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 64, numKvHeads: 64, headDim: 192, vocabSize: 163840,
    paramCount: 1_026_000_000_000,
    maxContext: 262144,
    numNextnLayers: 0,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128 },
    architecture: {
      type: 'moe',
      numExperts: 384,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 32_000_000_000
    }
  },
  {
    id: 'kimi-linear', name: 'Kimi-Linear-48B-A3B', family: 'kimi',
    publisher: 'Moonshot AI', releaseDate: '2026-02',
    nativeDtype: 'bf16',
    layers: 27, hiddenDim: 2304, intermediateDim: 9216,
    numHeads: 32, numKvHeads: 32, headDim: 192, vocabSize: 163840,
    paramCount: 48_000_000_000,
    maxContext: 1048576,
    numNextnLayers: 0,
    attention: {
      type: 'linear-mla-hybrid',
      kvLoraRank: 512, qkRopeHeadDim: 64,
      qkNopeHeadDim: 128, vHeadDim: 128,
      numLinearLayers: 20, numFullLayers: 7,
      numLinearHeads: 32, linearHeadDim: 128
    },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 3_000_000_000
    }
  },
  // === MiniMax ===
  // M2-family: full-attention GQA MoE (no linear/lightning attention — the
  // attn_type_list is all-full). Ships fp8 block-quantized. MTP depth 3
  // (num_mtp_modules 3, one transformer layer each).
  {
    id: 'minimax-m2.5', name: 'MiniMax M2.5', family: 'minimax-m2',
    publisher: 'MiniMax', releaseDate: '2026-02',
    nativeDtype: 'fp8',
    layers: 62, hiddenDim: 3072, intermediateDim: 1536,
    numHeads: 48, numKvHeads: 8, headDim: 128, vocabSize: 200064,
    paramCount: 230_000_000_000,
    maxContext: 196608,
    numNextnLayers: 3,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 0,
      activeParamCount: 10_000_000_000
    }
  },
  {
    id: 'minimax-m2.7', name: 'MiniMax M2.7', family: 'minimax-m2',
    publisher: 'MiniMax', releaseDate: '2026-04',
    nativeDtype: 'fp8',
    layers: 62, hiddenDim: 3072, intermediateDim: 1536,
    numHeads: 48, numKvHeads: 8, headDim: 128, vocabSize: 200064,
    paramCount: 230_000_000_000,
    maxContext: 204800,
    numNextnLayers: 3,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 0,
      activeParamCount: 10_000_000_000
    }
  },
  // === OpenAI gpt-oss ===
  // Alternating sliding(128)/full attention 1:1, MoE with top-4 routing.
  // Ships mxfp4 MoE weights (attention/embeddings stay bf16) → nativeDtype fp4.
  {
    id: 'gpt-oss-20b', name: 'gpt-oss-20b', family: 'gpt-oss',
    publisher: 'OpenAI', releaseDate: '2025-08',
    nativeDtype: 'fp4',
    layers: 24, hiddenDim: 2880, intermediateDim: 2880,
    numHeads: 64, numKvHeads: 8, headDim: 64, vocabSize: 201088,
    paramCount: 21_000_000_000,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'hybrid', slidingWindow: 128, numSlidingLayers: 12, numGlobalLayers: 12 },
    architecture: {
      type: 'moe',
      numExperts: 32,
      numExpertsActive: 4,
      numSharedExperts: 0,
      activeParamCount: 3_600_000_000
    }
  },
  {
    id: 'gpt-oss-120b', name: 'gpt-oss-120b', family: 'gpt-oss',
    publisher: 'OpenAI', releaseDate: '2025-08',
    nativeDtype: 'fp4',
    layers: 36, hiddenDim: 2880, intermediateDim: 2880,
    numHeads: 64, numKvHeads: 8, headDim: 64, vocabSize: 201088,
    paramCount: 117_000_000_000,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'hybrid', slidingWindow: 128, numSlidingLayers: 18, numGlobalLayers: 18 },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 4,
      numSharedExperts: 0,
      activeParamCount: 5_100_000_000
    }
  },
  // === NVIDIA Nemotron ===
  // NemotronH-family block hybrids: num_hidden_layers counts attention, Mamba2,
  // and FFN blocks separately (see mamba2-hybrid in types.ts). Block counts
  // parsed from hybrid_override_pattern / layers_block_type.
  {
    id: 'nemotron-h-56b', name: 'Nemotron-H 56B', family: 'nemotron',
    publisher: 'NVIDIA', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 118, hiddenDim: 8192, intermediateDim: 32768,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 131072,
    paramCount: 56_324_350_464,
    maxContext: 8192,
    numNextnLayers: 0,
    attention: {
      type: 'mamba2-hybrid',
      numMambaLayers: 54, numFullLayers: 10, numFfnLayers: 54,
      numMambaHeads: 256, mambaHeadDim: 64, ssmStateSize: 256
    },
    architecture: { type: 'dense' }
  },
  // Nemotron 3: Mamba2 hybrid + MoE with relu² (2-matrix) experts and a
  // double-width shared expert. activeParamCount from each model card.
  {
    id: 'nemotron-3-nano-30b-a3b', name: 'Nemotron 3 Nano 30B-A3B', family: 'nemotron',
    publisher: 'NVIDIA', releaseDate: '2025-12',
    nativeDtype: 'bf16',
    layers: 52, hiddenDim: 2688, intermediateDim: 1856,
    numHeads: 32, numKvHeads: 2, headDim: 128, vocabSize: 131072,
    paramCount: 31_577_937_344,
    maxContext: 262144,
    numNextnLayers: 0,
    attention: {
      type: 'mamba2-hybrid',
      numMambaLayers: 23, numFullLayers: 6, numFfnLayers: 23,
      numMambaHeads: 64, mambaHeadDim: 64, ssmStateSize: 128
    },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 6,
      numSharedExperts: 1,
      activeParamCount: 3_500_000_000
    }
  },
  {
    id: 'nemotron-3-super-120b-a12b', name: 'Nemotron 3 Super 120B-A12B', family: 'nemotron',
    publisher: 'NVIDIA', releaseDate: '2026-03',
    nativeDtype: 'bf16',
    layers: 88, hiddenDim: 4096, intermediateDim: 2688,
    numHeads: 32, numKvHeads: 2, headDim: 128, vocabSize: 131072,
    paramCount: 123_611_012_096,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'mamba2-hybrid',
      numMambaLayers: 40, numFullLayers: 8, numFfnLayers: 40,
      numMambaHeads: 128, mambaHeadDim: 64, ssmStateSize: 128
    },
    architecture: {
      type: 'moe',
      numExperts: 512,
      numExpertsActive: 22,
      numSharedExperts: 1,
      activeParamCount: 12_000_000_000
    }
  },
  {
    id: 'nemotron-3-ultra-550b-a55b', name: 'Nemotron 3 Ultra 550B-A55B', family: 'nemotron',
    publisher: 'NVIDIA', releaseDate: '2026-06',
    nativeDtype: 'bf16',
    layers: 108, hiddenDim: 8192, intermediateDim: 5120,
    numHeads: 64, numKvHeads: 2, headDim: 128, vocabSize: 131072,
    paramCount: 560_524_578_816,
    maxContext: 262144,
    numNextnLayers: 1,
    attention: {
      type: 'mamba2-hybrid',
      numMambaLayers: 48, numFullLayers: 12, numFfnLayers: 48,
      numMambaHeads: 256, mambaHeadDim: 64, ssmStateSize: 128
    },
    architecture: {
      type: 'moe',
      numExperts: 512,
      numExpertsActive: 22,
      numSharedExperts: 1,
      activeParamCount: 55_000_000_000
    }
  },
  // Puzzle-NAS derivative of Llama 3.3 70B: 31 of 80 blocks had attention
  // removed (attention.no_op in block_configs); the 49 survivors share one
  // GQA geometry (64 heads / 8 KV). FFN widths vary per block (ffn_mult 0.5
  // to 5.25) — intermediateDim is the block average (activations estimate
  // only); paramCount carries the weights truth.
  {
    id: 'llama-3.3-nemotron-super-49b', name: 'Llama-3.3-Nemotron-Super 49B', family: 'nemotron',
    publisher: 'NVIDIA', releaseDate: '2025-03',
    nativeDtype: 'bf16',
    layers: 80, hiddenDim: 8192, intermediateDim: 30720,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 49_867_145_216,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'partial', numFullLayers: 49 },
    architecture: { type: 'dense' }
  },
  // === Xiaomi MiMo ===
  {
    id: 'mimo-7b', name: 'MiMo-7B', family: 'mimo',
    publisher: 'Xiaomi', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 36, hiddenDim: 4096, intermediateDim: 11008,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 151680,
    paramCount: 7_833_409_536,
    maxContext: 32768,
    numNextnLayers: 1,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  // === Z.ai / GLM ===
  // GLM-4.5-Air pairs the new shared-expert MoE schema with regular GQA full
  // attention (no MLA). The 12:1 KV-head reduction (96 attention / 8 KV) is the
  // most aggressive GQA ratio in the seed.
  {
    id: 'glm-4.5-air', name: 'GLM-4.5-Air', family: 'glm',
    publisher: 'Zhipu AI', releaseDate: '2025-07',
    nativeDtype: 'bf16',
    layers: 46, hiddenDim: 4096, intermediateDim: 10944,
    numHeads: 96, numKvHeads: 8, headDim: 128, vocabSize: 151552,
    paramCount: 106_000_000_000,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 12_000_000_000
    }
  },
  // GLM-4.7-Flash introduces MLA to the GLM MoE line: kv_lora_rank 512,
  // qk_nope_head_dim 192, qk_rope_head_dim 64, v_head_dim 256.
  {
    id: 'glm-4.7-flash', name: 'GLM-4.7-Flash', family: 'glm',
    publisher: 'Zhipu AI', releaseDate: '2026-01',
    nativeDtype: 'bf16',
    layers: 47, hiddenDim: 2048, intermediateDim: 1536,
    numHeads: 20, numKvHeads: 20, headDim: 256, vocabSize: 154880,
    paramCount: 30_000_000_000,
    maxContext: 202752,
    numNextnLayers: 0,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 192, vHeadDim: 256 },
    architecture: {
      type: 'moe',
      numExperts: 64,
      numExpertsActive: 4,
      numSharedExperts: 1,
      activeParamCount: 3_000_000_000
    }
  },
  {
    id: 'glm-5', name: 'GLM-5', family: 'glm',
    publisher: 'Zhipu AI', releaseDate: '2026-03',
    nativeDtype: 'bf16',
    layers: 78, hiddenDim: 6144, intermediateDim: 12288,
    numHeads: 64, numKvHeads: 64, headDim: 256, vocabSize: 154880,
    paramCount: 744_000_000_000,
    maxContext: 202752,
    attention: {
      type: 'mla-dsa',
      kvLoraRank: 512, qkRopeHeadDim: 64,
      qkNopeHeadDim: 192, vHeadDim: 256,
      topK: 2048
    },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 40_000_000_000
    },
    // config num_nextn_predict_layers: 1 (one MTP layer). Engine models this as
    // mtpFactor = 1 + depth = 2× decode throughput (100%-acceptance ceiling),
    // matching the DeepSeek V3+ entries' treatment of MTP.
    numNextnLayers: 1
  },
  // GLM-5.2 (zai-org/GLM-5.2, config model_type glm_moe_dsa, created 2026-06-16):
  // same MLA-DSA backbone as GLM-5 — identical 78 layers / 6144 hidden / 256-expert
  // (8 active + 1 shared) MoE and identical attention geometry (kv_lora 512,
  // qk_rope 64, qk_nope 192, v 256, index_topk 2048). The shipped change is context:
  // max_position_embeddings 1,048,576 (1M) vs GLM-5's 202,752. paramCount 753B from
  // the safetensors index (753,329,940,480 params, BF16); activeParamCount inherited
  // from GLM-5's identical active path (card doesn't break it out). headDim 256
  // follows the GLM-5 entry (inert for MLA — KV/attn key off kv_lora + rope, not
  // headDim). numNextnLayers 1 matches config num_nextn_predict_layers and GLM-5.
  {
    id: 'glm-5.2', name: 'GLM-5.2', family: 'glm',
    publisher: 'Zhipu AI', releaseDate: '2026-06',
    nativeDtype: 'bf16',
    layers: 78, hiddenDim: 6144, intermediateDim: 12288,
    numHeads: 64, numKvHeads: 64, headDim: 256, vocabSize: 154880,
    paramCount: 753_000_000_000,
    maxContext: 1048576,
    attention: {
      type: 'mla-dsa',
      kvLoraRank: 512, qkRopeHeadDim: 64,
      qkNopeHeadDim: 192, vHeadDim: 256,
      topK: 2048
    },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 40_000_000_000
    },
    numNextnLayers: 1
  },
  // === Phi ===
  {
    id: 'phi-4', name: 'Phi-4 14B', family: 'phi',
    publisher: 'Microsoft', releaseDate: '2024-12',
    nativeDtype: 'bf16',
    layers: 40, hiddenDim: 5120, intermediateDim: 17920,
    numHeads: 40, numKvHeads: 10, headDim: 128, vocabSize: 100352,
    paramCount: 14_659_507_200,
    maxContext: 16384,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'phi-4-mini', name: 'Phi-4-mini 3.8B', family: 'phi',
    publisher: 'Microsoft', releaseDate: '2025-02',
    nativeDtype: 'bf16',
    layers: 32, hiddenDim: 3072, intermediateDim: 8192,
    numHeads: 24, numKvHeads: 8, headDim: 128, vocabSize: 200064,
    paramCount: 3_840_000_000,
    maxContext: 131072,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'phi-4-reasoning', name: 'Phi-4-reasoning 14B', family: 'phi',
    publisher: 'Microsoft', releaseDate: '2025-04',
    nativeDtype: 'bf16',
    layers: 40, hiddenDim: 5120, intermediateDim: 17920,
    numHeads: 40, numKvHeads: 10, headDim: 128, vocabSize: 100352,
    paramCount: 14_659_507_200,
    maxContext: 32768,
    numNextnLayers: 0,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }
]

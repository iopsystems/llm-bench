import type { ModelArch } from '../engine/types'

// Architecture fields sourced from HuggingFace config.json per model.
// paramCount taken from each model's official card.
export const MODELS: ModelArch[] = [
  // === Qwen3 dense series ===
  {
    id: 'qwen3-1.7b', name: 'Qwen3 1.7B', family: 'qwen3',
    layers: 28, hiddenDim: 2048, intermediateDim: 6144,
    numHeads: 16, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 1_720_000_000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-4b', name: 'Qwen3 4B', family: 'qwen3',
    layers: 36, hiddenDim: 2560, intermediateDim: 9728,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 4_020_000_000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-8b', name: 'Qwen3 8B', family: 'qwen3',
    layers: 36, hiddenDim: 4096, intermediateDim: 12288,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 8_190_000_000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-14b', name: 'Qwen3 14B', family: 'qwen3',
    layers: 40, hiddenDim: 5120, intermediateDim: 17408,
    numHeads: 40, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 14_770_000_000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'qwen3-32b', name: 'Qwen3 32B', family: 'qwen3',
    layers: 64, hiddenDim: 5120, intermediateDim: 25600,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 32_760_000_000,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  // === Llama ===
  {
    id: 'llama-3.3-70b', name: 'Llama 3.3 70B', family: 'llama-3',
    layers: 80, hiddenDim: 8192, intermediateDim: 28672,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 70_553_706_496,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'llama-3.1-405b', name: 'Llama 3.1 405B', family: 'llama-3',
    layers: 126, hiddenDim: 16384, intermediateDim: 53248,
    numHeads: 128, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 405_853_356_032,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  // === Gemma 3 ===
  {
    id: 'gemma-3-12b', name: 'Gemma 3 12B', family: 'gemma-3',
    layers: 48, hiddenDim: 3840, intermediateDim: 15360,
    numHeads: 16, numKvHeads: 8, headDim: 256, vocabSize: 262144,
    paramCount: 12_187_000_000,
    attention: { type: 'hybrid', slidingWindow: 1024, numSlidingLayers: 40, numGlobalLayers: 8 },
    architecture: { type: 'dense' }
  },
  {
    id: 'gemma-3-27b', name: 'Gemma 3 27B', family: 'gemma-3',
    layers: 62, hiddenDim: 5376, intermediateDim: 21504,
    numHeads: 32, numKvHeads: 16, headDim: 128, vocabSize: 262144,
    paramCount: 27_009_000_000,
    attention: { type: 'hybrid', slidingWindow: 1024, numSlidingLayers: 52, numGlobalLayers: 10 },
    architecture: { type: 'dense' }
  },
  // === Mistral ===
  {
    id: 'mistral-7b-v0.1', name: 'Mistral 7B v0.1', family: 'mistral',
    layers: 32, hiddenDim: 4096, intermediateDim: 14336,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
    paramCount: 7_241_732_096,
    attention: { type: 'sliding', window: 4096 },
    architecture: { type: 'dense' }
  },
  {
    id: 'mixtral-8x7b', name: 'Mixtral 8x7B v0.1', family: 'mistral',
    layers: 32, hiddenDim: 4096, intermediateDim: 14336,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 32000,
    paramCount: 46_702_792_704,
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
    layers: 56, hiddenDim: 6144, intermediateDim: 16384,
    numHeads: 48, numKvHeads: 8, headDim: 128, vocabSize: 32000,
    paramCount: 141_000_000_000,
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
    layers: 40, hiddenDim: 5120, intermediateDim: 32768,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 131072,
    paramCount: 23_572_403_200,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  {
    id: 'mistral-large-2', name: 'Mistral Large 2 123B', family: 'mistral',
    layers: 88, hiddenDim: 12288, intermediateDim: 28672,
    numHeads: 96, numKvHeads: 8, headDim: 128, vocabSize: 32768,
    paramCount: 122_610_524_160,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  },
  // === DeepSeek ===
  {
    id: 'deepseek-v2', name: 'DeepSeek-V2', family: 'deepseek',
    layers: 60, hiddenDim: 5120, intermediateDim: 12288,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 102400,
    paramCount: 236_000_000_000,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 },
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
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
    paramCount: 671_000_000_000,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 },
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
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 128, numKvHeads: 128, headDim: 192, vocabSize: 129280,
    paramCount: 671_000_000_000,
    attention: { type: 'mla-dsa', kvLoraRank: 512, qkRopeHeadDim: 64, topK: 2048 },
    architecture: {
      type: 'moe',
      numExperts: 256,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 37_000_000_000
    }
  },
  // === Moonshot / Kimi ===
  {
    id: 'kimi-k2', name: 'Kimi K2', family: 'kimi',
    layers: 61, hiddenDim: 7168, intermediateDim: 18432,
    numHeads: 64, numKvHeads: 64, headDim: 192, vocabSize: 163840,
    paramCount: 1_026_000_000_000,
    attention: { type: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 },
    architecture: {
      type: 'moe',
      numExperts: 384,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 32_000_000_000
    }
  },
  // === Z.ai / GLM ===
  // GLM-4.5-Air pairs the new shared-expert MoE schema with regular GQA full
  // attention (no MLA). The 12:1 KV-head reduction (96 attention / 8 KV) is the
  // most aggressive GQA ratio in the seed.
  {
    id: 'glm-4.5-air', name: 'GLM-4.5-Air', family: 'glm',
    layers: 46, hiddenDim: 4096, intermediateDim: 10944,
    numHeads: 96, numKvHeads: 8, headDim: 128, vocabSize: 151552,
    paramCount: 106_000_000_000,
    attention: { type: 'full' },
    architecture: {
      type: 'moe',
      numExperts: 128,
      numExpertsActive: 8,
      numSharedExperts: 1,
      activeParamCount: 12_000_000_000
    }
  },
  // === Phi ===
  {
    id: 'phi-4', name: 'Phi-4 14B', family: 'phi',
    layers: 40, hiddenDim: 5120, intermediateDim: 17920,
    numHeads: 40, numKvHeads: 10, headDim: 128, vocabSize: 100352,
    paramCount: 14_659_507_200,
    attention: { type: 'full' },
    architecture: { type: 'dense' }
  }
]

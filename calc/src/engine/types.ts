export type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4'

export interface GpuOperatingPoint {
  id: string
  label: string
  tflops: Partial<Record<Dtype, number>>
  hbmBandwidthGBs: number
  // Provenance — primarily for non-peak tiers. Keys reference src/data/sources.ts.
  // Per-axis arrays: list the same key in both when one source covers both axes.
  tflopsSources?: string[]
  bandwidthSources?: string[]
  asOf?: string
  notes?: string
}

export interface GpuVariant {
  id: string
  label: string
  hbmCapacityGB: number
  operatingPoints: GpuOperatingPoint[]
}

export interface GpuSpec {
  id: string
  name: string
  vendor: string
  family?: string
  variants: GpuVariant[]
}

export type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla'; kvLoraRank: number; qkRopeHeadDim: number }
  | { type: 'hybrid'; slidingWindow: number; numSlidingLayers: number; numGlobalLayers: number }
  | { type: 'mla-dsa'; kvLoraRank: number; qkRopeHeadDim: number; topK: number }

export type ArchitectureConfig =
  | { type: 'dense' }
  | { type: 'moe';
      numExperts: number;          // routed-only
      numExpertsActive: number;    // top-K routed per token
      numSharedExperts: number;    // always-active expert count (separate from routed pool)
      activeParamCount: number;    // aggregate routed-active + shared (from model card)
    }

export interface ModelArch {
  id: string
  name: string
  family: string
  layers: number
  hiddenDim: number
  intermediateDim: number
  numHeads: number
  numKvHeads: number
  headDim: number
  vocabSize: number
  paramCount: number
  attention: AttentionConfig
  architecture: ArchitectureConfig
}

export interface Quantization {
  weights: Dtype
  kv: Dtype
  activations: Dtype
}

export interface Workload {
  promptTokens: number
  outputTokens: number
  concurrency: number
}

export interface CalcInput {
  gpu: GpuSpec
  gpuVariantId: string
  model: ModelArch
  quant: Quantization
  workload: Workload
}

export interface MemoryResult {
  weights: number
  kvCachePerRequest: number
  kvCacheTotal: number
  activationsPeak: number
  total: number
  hbmCapacityGB: number
  headroom: number
  fits: boolean
}

export interface PerfTier {
  prefill: { flops: number; bytes: number; timeS: number; regime: 'compute' | 'memory' }
  decode:  { flopsPerStep: number; bytesPerStep: number; timePerTokenS: number;
             regime: 'compute' | 'memory'; aggregateTokensPerS: number }
  ttftS: number
  inputTokenRate: number
  outputTokenRate: number
  // Echoed from the source operating point so consumers can show provenance.
  tflopsSources?: string[]
  bandwidthSources?: string[]
  asOf?: string
  notes?: string
}

export interface DerivationStep {
  label: string
  expression: string
  value: number
  unit: string
}

export interface CalcResult {
  memory: MemoryResult
  perf: Record<string, PerfTier>
  derivation: DerivationStep[]
}

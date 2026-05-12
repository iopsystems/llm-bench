export type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4'

export interface GpuOperatingPoint {
  id: string
  label: string
  tflops: Partial<Record<Dtype, number>>
  hbmBandwidthGBs: number
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

export type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'fp4' | 'int8' | 'int4'

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

// === Multi-GPU types ===
// These describe interconnects and parallelism modes for multi-GPU systems.
// The roofline engine in this directory is still single-GPU; these types back
// the data registries in src/data/interconnects.ts and src/data/parallelism.ts
// so callers and a future multi-GPU engine have a stable schema to consume.

export type InterconnectTopology =
  | 'point-to-point'   // direct GPU↔GPU links (small meshes; HGX cube-mesh pre-NVSwitch)
  | 'switched'         // NVSwitch / NVL Switch — non-blocking among the connected set
  | '2d-torus'         // TPU v6e
  | '3d-torus'         // TPU v4 / v5p
  | 'fat-tree'         // typical InfiniBand multi-rail scale-out
  | 'ring'             // pure ring (legacy)

export type InterconnectScale =
  | 'die-to-die'       // intra-package (Apple UltraFusion, MI300 IF on-package)
  | 'intra-node'       // HGX baseboard, OAM tray
  | 'scale-up'         // NVL72 fabric, TPU pod
  | 'scale-out'        // multi-node IB/RoCE

export interface InterconnectSpec {
  id: string
  name: string
  vendor: string
  generation?: string

  // Bandwidth conventions:
  //   - perGpuBandwidthGBs is bidirectional aggregate per GPU/chip — the headline
  //     number vendors quote. For ring all-reduce math, callers need per-direction
  //     bandwidth, which is half of bidirectional unless perDirectionGBs is set.
  //   - perLinkGBs is per direction unless otherwise noted in `notes`.
  perGpuBandwidthGBs: number
  perDirectionGBs?: number
  linksPerGpu?: number
  perLinkGBs?: number

  topology: InterconnectTopology
  scale: InterconnectScale
  maxScaleUpGpus?: number      // size of the largest non-blocking domain

  // Round-trip latency for a single hop, ns. Optional; many vendors don't disclose.
  hopLatencyNs?: number

  sources?: string[]
  asOf?: string
  notes?: string
}

export type CollectivePrimitive =
  | 'all-reduce'
  | 'all-gather'
  | 'reduce-scatter'
  | 'all-to-all'
  | 'point-to-point'
  | 'broadcast'

export type ParallelismApplicability = 'dense' | 'moe'

export interface ParallelismMode {
  id: 'tp' | 'pp' | 'ep' | 'sp' | 'cp' | 'dp'
  name: string
  shortLabel: string
  collective: CollectivePrimitive

  // Frequency of the collective. Human-readable; the multi-GPU engine will
  // encode the actual per-step formula when implemented.
  collectiveFrequency: string

  // Volume formula in words for documentation. Expressed in terms of:
  //   N = ranks in the parallelism group
  //   B = batch tokens (prefill) or batch (decode)
  //   d = hidden dim
  //   L = number of layers
  //   E = active experts, E_total = total experts (for EP)
  // Engine consumers should re-derive formulas from a parallelism math module;
  // this string is a contract spec, not parser input.
  volumeFormulaText: string

  shardingDim: string
  applicableTo: ParallelismApplicability[]

  // Practical scale ceiling — beyond this, comms dominates compute and the
  // mode stops being useful. Empirical, not a hard limit.
  typicalScaleLimit?: { ranks: number; reason: string }

  // Modes that compose multiplicatively with this one. E.g. TP × PP × DP is
  // common; TP × EP coexists in MoE deployments.
  composesWith?: ParallelismMode['id'][]

  notes?: string
}


export type AttentionConfig =
  | { type: 'full' }
  | { type: 'sliding'; window: number }
  | { type: 'mla';
      kvLoraRank: number;
      qkRopeHeadDim: number;
      qkNopeHeadDim: number;
      vHeadDim: number
    }
  | { type: 'hybrid'; slidingWindow: number; numSlidingLayers: number; numGlobalLayers: number }
  | { type: 'mla-dsa';
      kvLoraRank: number;
      qkRopeHeadDim: number;
      qkNopeHeadDim: number;
      vHeadDim: number;
      topK: number
    }
  | { type: 'linear-mla-hybrid';
      // Inner MLA configuration (for the full-attention layers)
      kvLoraRank: number;
      qkRopeHeadDim: number;
      qkNopeHeadDim: number;
      vHeadDim: number;
      // Per-layer counts (must sum to model.layers)
      numLinearLayers: number;
      numFullLayers: number;
      // Linear-attention geometry (state size = numLinearHeads × linearHeadDim² per layer)
      numLinearHeads: number;
      linearHeadDim: number
    }
  | { type: 'csa-hca-hybrid';
      // Layer counts (must sum to model.layers)
      numSlidingLayers: number;
      numCsaLayers: number;
      numHcaLayers: number;
      // Sliding-window size (applies to dedicated sliding layers AND
      // to the per-layer side-branch on CSA/HCA layers)
      slidingWindow: number;
      // CSA params
      csaCompressionM: number;
      csaTopK: number;
      csaIndexerHeads: number;
      csaIndexerHeadDim: number;
      // HCA params
      hcaCompressionM: number
    }

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
  numNextnLayers: number  // Multi-Token Prediction depth; 0 for non-MTP models
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

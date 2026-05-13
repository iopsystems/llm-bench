export type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'fp4' | 'int8' | 'int4'

export interface AcceleratorOperatingPoint {
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

export interface AcceleratorVariant {
  id: string
  label: string
  hbmCapacityGB: number
  operatingPoints: AcceleratorOperatingPoint[]
}

export interface AcceleratorSpec {
  id: string
  name: string
  vendor: string
  family?: string
  variants: AcceleratorVariant[]
}

// === Multi-accelerator types ===
// These describe interconnects and parallelism modes for multi-accelerator
// systems. The roofline engine in this directory is still single-accelerator;
// these types back the data registries in src/data/interconnects.ts and
// src/data/parallelism.ts so callers and a future multi-accelerator engine
// have a stable schema to consume.
//
// Field-naming note: lower-level fields like `perGpuBandwidthGBs` and
// `linksPerGpu` retain "Gpu" in their names. The vocabulary mirrors how
// vendors describe interconnect specs, and renaming them touches every
// consumer. Treat them as "per accelerator" for non-GPU products
// (TPU/Trainium/Gaudi/etc.).

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

  // Achievable-BW model. Two layers; either or both may be omitted.
  //
  //   `contention`  — analytical, derives effective per-GPU BW for any
  //                   (collective, N) from a small set of fabric-level
  //                   parameters. Generalizes to configurations not measured.
  //   `tiers`       — empirical measurements at specific (collective, N,
  //                   message size) points. Override the analytical model
  //                   when present, the way GPU `achievable` operating points
  //                   override `peak` TFLOPS.
  //
  // CORE ASSUMPTION: "one workload owns the fabric." Both layers assume the
  // entire interconnect is dedicated to the single LLM serving workload being
  // modeled — no cross-tenant contention, no other jobs sharing the NVSwitch
  // or torus. Real shared clusters degrade further; the calc does not model
  // that. Stated once here so per-entry notes don't have to repeat it.
  contention?: FabricContention
  tiers?: InterconnectAchievableTier[]

  sources?: string[]
  asOf?: string
  notes?: string
}

export type HopCostModel =
  | 'flat'            // Switched fabrics (NVSwitch, NVL72, fat-tree): any pair = 1 hop, no distance penalty.
  | 'mesh-2d'         // 2D grid w/o wrap. Mean hop count ≈ (2/3) · sqrt(N).
  | 'torus-2d'        // 2D grid with wrap (TPU v6e). Mean hop ≈ sqrt(N) / 4.
  | 'torus-3d'        // 3D torus (TPU v5p). Mean hop ≈ cbrt(N) · 3/8.
  | 'fat-tree-2level' // Hierarchical: leaf switch + spine. Hop count is fixed (1 or 3) but oversubscription kicks in.

export interface FabricContention {
  // What fraction of `perGpuBandwidthGBs` survives at the maximum non-blocking
  // scale, assuming the workload occupies the whole fabric. 1.0 = full
  // bisection (NVSwitch HGX/NVL72); ~0.5 = HGX cube-mesh pre-NVSwitch; ~0.2 =
  // PCIe routed through a CPU root complex. Used by the engine as the upper
  // ceiling at N = maxScaleUpGpus.
  bisectionFactor: number

  // Leaf:spine oversubscription ratio for hierarchical fabrics. 1.0 means
  // non-blocking; 2.0 means spine carries half the leaf bandwidth (typical
  // datacenter IB). Only consulted when traffic crosses the hierarchy — i.e.
  // collectives that span more than one leaf switch.
  oversubscription?: number

  // How hop count grows with rank count. Engine looks up the table per class
  // (see HopCostModel docstrings for the formulas) and computes effective BW
  // as roughly `perGpuBandwidthGBs · singleHopUtilization · bisectionFactor /
  // hops(N)` for distance-sensitive topologies.
  hopCostModel: HopCostModel

  // Fraction of peak that a single isolated GPU-pair can drive on this fabric
  // — distinct from contention. Captures the protocol/stack overhead before
  // any sharing. NCCL on NVLink lands around 0.45 (despite the fabric being
  // non-blocking) because the algorithmic ring on top doesn't drive the link
  // at line rate. Raw RDMA on IB lands closer to 0.9.
  singleHopUtilization: number
}

export interface InterconnectAchievableTier {
  id: string                    // e.g. 'nccl-allreduce-tp8'
  label: string                 // e.g. 'NCCL all-reduce, TP=8, 16MiB msg'

  // Context for the measurement. The engine matches a (collective, ranks)
  // tuple against tiers; messageSizeBytes is informational.
  collective: CollectivePrimitive
  ranks: number
  messageSizeBytes?: number

  // Achieved per-GPU bidirectional bandwidth, same units as
  // InterconnectSpec.perGpuBandwidthGBs.
  perGpuBandwidthGBs: number

  // Optional: framework / driver / NCCL version, since these meaningfully
  // shift the curve.
  software?: string             // e.g. 'NCCL 2.21 + CUDA 12.4'

  sources?: string[]
  asOf?: string
  notes?: string
}

// === Multi-accelerator systems ===
// A MultiAcceleratorSystem composes a specific accelerator (GPU, TPU, Trainium,
// Gaudi, etc., referenced by id + variant) with a specific scale-up
// interconnect and a count, capturing concrete products users can actually buy
// or rent today. The registry lives in src/data/systems.ts.

export type SystemFormFactor =
  | 'baseboard'        // 8-accelerator baseboard sold to OEMs (HGX H100/H200/B200)
  | 'node'             // ready-to-deploy server (DGX H100, MI300X 8-OAM)
  | 'rack'             // pre-integrated rack (GB200 NVL72)
  | 'pod-slice'        // TPU pod slice
  | 'cloud-instance'   // cloud-only SKU with no on-prem equivalent (AWS Trn2)
  | 'wafer'            // wafer-scale (Cerebras CS-3)

export type CloudProvider =
  | 'aws'
  | 'azure'
  | 'gcp'
  | 'oci'
  | 'coreweave'
  | 'lambda'
  | 'crusoe'
  | 'intel-tiber'
  | 'cerebras-cloud'

export interface MultiAcceleratorSystem {
  id: string
  name: string
  vendor: string
  generation?: string
  formFactor: SystemFormFactor

  // Composition by reference to other registries. The data file should set
  // these to ids that exist in ACCELERATORS / INTERCONNECTS; not enforced at
  // the type level (would require const-keyed lookup), but reviewers can check.
  accelerator: {
    id: string                 // AcceleratorSpec.id
    variantId: string          // AcceleratorVariant.id (which capacity / SKU)
    count: number              // accelerators in the system
  }

  // Primary scale-up fabric among the accelerators. References InterconnectSpec.id.
  interconnectId: string

  // Optional scale-out NIC layer — DGX-class nodes ship with multiple
  // ConnectX/EFA NICs for multi-node training. Single-instance serving
  // rarely crosses this, but useful for fleet sizing.
  scaleOutInterconnectId?: string
  scaleOutNicsPerNode?: number

  // Denormalized aggregates for UI display / cross-system comparison.
  // Consumers MUST NOT recompute a roofline from these — they should look up
  // the accelerator and interconnect entries directly. Stored here so
  // reviewers can sanity-check the composition at a glance.
  aggregate: {
    totalHbmGB: number              // ∑ hbmCapacityGB across accelerators
    fabricBidirectionalTBs: number  // total per-accelerator bidirectional BW × count, in TB/s
  }

  availability?: {
    onPrem?: boolean                  // sold through OEM channel
    clouds?: CloudProvider[]
  }

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
  | { type: 'delta-hybrid';
      // Qwen3.5: Gated DeltaNet (linear/state-space) + Gated Attention (RoPE)
      // Per-layer counts (must sum to model.layers)
      numDeltaNetLayers: number;
      numFullLayers: number;
      // DeltaNet linear-attention geometry (state = numDeltaNetHeads × deltaHeadDim² per layer)
      numDeltaNetHeads: number;   // V heads (value projection heads for the recurrent state)
      deltaHeadDim: number;       // state matrix inner dim (same for Q and K)
      // Gated Attention geometry (standard RoPE attention on a subset of layers)
      // Uses model.numKvHeads and model.headDim for KV cache computation
      ropeDim: number;            // RoPE embedding dimension
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

export interface MultiDeviceConfig {
  system: MultiAcceleratorSystem
  parallelism: ParallelismMode['id'][]
  parallelismDegrees: Partial<Record<ParallelismMode['id'], number>>
}

export interface CalcInput {
  accelerator: AcceleratorSpec
  acceleratorVariantId: string
  model: ModelArch
  quant: Quantization
  workload: Workload
  multiDevice?: MultiDeviceConfig
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
  perRank?: {
    weights: number
    kvCachePerRequest: number
    activationsPeak: number
    total: number
    headroom: number
    fits: boolean
  }
}

export interface PerfTier {
  prefill: { flops: number; bytes: number; timeS: number; regime: 'compute' | 'memory' | 'comms' }
  decode:  { flopsPerStep: number; bytesPerStep: number; timePerTokenS: number;
             regime: 'compute' | 'memory' | 'comms'; aggregateTokensPerS: number }
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

import type { ParallelismMode } from '../engine/types'

// Parallelism mode registry — describes the modes of operation available when
// multiple accelerators serve one model. Used by the multi-GPU tab to:
//   1. Render mode descriptions / explainers.
//   2. Look up which collective and volume formula a mode uses, so the engine
//      can add the correct comms cost to its roofline.
//
// Volume formulas use these symbols:
//   N        = ranks in the parallelism group (TP world size for TP, etc.)
//   B        = batch tokens this collective participates in
//   d        = model hidden dim
//   d_ff     = MLP intermediate dim (≈ 4d for vanilla transformer, varies)
//   L        = number of transformer layers
//   dtype    = activation/comm element size in bytes
//   E_active = active experts per token (MoE only)
//
// "Per layer", "per microbatch", etc. describe the collective's invocation
// frequency, not its volume.

export const PARALLELISM_MODES: ParallelismMode[] = [
  {
    id: 'tp',
    name: 'Tensor Parallelism',
    shortLabel: 'TP',
    collective: 'all-reduce',
    collectiveFrequency: '2 per transformer layer (after self-attention output projection, after MLP down-projection)',
    volumeFormulaText:
      'Per all-reduce: 2 · (N-1)/N · B · d · dtype bytes (ring algorithm). ' +
      'Total per layer = 2× that. The MLP/attention input projections do NOT need an ' +
      'all-reduce because they consume the already-replicated input; only the ' +
      'output projections collapse the sharded partial sums.',
    shardingDim: 'hidden / head dimensions of weight matrices (column-parallel on QKV/up, row-parallel on out/down)',
    applicableTo: ['dense', 'moe'],
    typicalScaleLimit: {
      ranks: 8,
      reason: 'Comms volume grows linearly with B·d while compute/rank scales as 1/N; beyond ~8 ranks on NVLink-class fabric, the all-reduce term starts to dominate decode and even prefill at moderate batch.'
    },
    composesWith: ['pp', 'ep', 'dp', 'sp', 'cp'],
    notes: 'The dominant intra-node parallelism for serving. Megatron-LM convention. With sequence parallelism (SP), the all-reduce decomposes into all-gather + reduce-scatter, same total volume but enables overlap with compute.'
  },
  {
    id: 'pp',
    name: 'Pipeline Parallelism',
    shortLabel: 'PP',
    collective: 'point-to-point',
    collectiveFrequency: 'one send/recv per pipeline stage boundary per microbatch (so L/N times per forward pass across the pipeline)',
    volumeFormulaText:
      'Per stage boundary: B · d · dtype bytes (single send/recv of the activation tensor between adjacent stages). ' +
      'Per token, the boundary is crossed N-1 times across N stages.',
    shardingDim: 'layers (each stage owns L/N contiguous transformer blocks)',
    applicableTo: ['dense', 'moe'],
    typicalScaleLimit: {
      ranks: 16,
      reason: 'Decode steady-state throughput is independent of N once the pipeline fills, but the prefill bubble = (N-1)/microbatches grows; long contexts can amortize it but interactive serving rarely uses PP > 4-8.'
    },
    composesWith: ['tp', 'ep', 'dp'],
    notes: 'Memory-saving and bandwidth-cheap (point-to-point only) but adds first-token latency via the pipeline bubble. 1F1B and interleaved schedules reduce but do not eliminate the bubble.'
  },
  {
    id: 'ep',
    name: 'Expert Parallelism',
    shortLabel: 'EP',
    collective: 'all-to-all',
    collectiveFrequency: '2 per MoE layer (dispatch tokens to expert ranks, then combine results back)',
    volumeFormulaText:
      'Per all-to-all: (N-1)/N · B · d · dtype bytes (each rank sends (N-1)/N of its tokens to other ranks, on average). ' +
      'For routed-only MoE, only tokens assigned to non-local experts are sent. Cost scales with E_active and the imbalance of routing.',
    shardingDim: 'experts (each rank owns E_total/N experts)',
    applicableTo: ['moe'],
    typicalScaleLimit: {
      ranks: 64,
      reason: 'All-to-all volume is communication-pattern-heavy and benefits from a flat, high-bisection fabric (NVSwitch, TPU torus). Effective scale ceiling depends on routing skew and fabric topology more than on theoretical limits.'
    },
    composesWith: ['tp', 'pp', 'dp'],
    notes: 'MoE-only. DeepSeek-V3 and Mixtral-class models route to top-K experts; the all-to-all dominates serving cost on consumer interconnects but is fine on NVL72 / TPU torus / Gaudi RoCE mesh.'
  },
  {
    id: 'sp',
    name: 'Sequence Parallelism',
    shortLabel: 'SP',
    collective: 'all-gather',
    collectiveFrequency: 'paired with TP — replaces each TP all-reduce with an all-gather + reduce-scatter of the same total volume',
    volumeFormulaText:
      'Same total volume as the TP all-reduce it replaces: 2 · (N-1)/N · B · d · dtype bytes per layer. ' +
      'But each half can overlap with the other tensor-parallel rank\'s compute, hiding ~50% of the comms cost in steady-state.',
    shardingDim: 'sequence length within the layer-norm and dropout regions where TP would otherwise replicate activations',
    applicableTo: ['dense', 'moe'],
    typicalScaleLimit: {
      ranks: 8,
      reason: 'SP\'s scale is determined by TP\'s — it\'s an optimization on top of TP, not a standalone axis.'
    },
    composesWith: ['tp', 'pp', 'dp'],
    notes: 'Standard in Megatron-LM/Nemo. Not a separate parallelism axis from TP — it\'s an overlap optimization.'
  },
  {
    id: 'cp',
    name: 'Context Parallelism',
    shortLabel: 'CP',
    collective: 'all-gather',
    collectiveFrequency: 'attention layer only — all-gather KV (or partial attention outputs) across ranks',
    volumeFormulaText:
      'Per attention layer: O(B · d_kv · dtype) all-gather, where B is the sequence chunk per rank. ' +
      'Volume grows linearly with sequence length and rank count; subtler than TP because only attention touches the comms.',
    shardingDim: 'sequence length (each rank owns a contiguous chunk of the prompt/context)',
    applicableTo: ['dense', 'moe'],
    typicalScaleLimit: {
      ranks: 8,
      reason: 'Used for long contexts (32K+). At small contexts the all-gather overhead dwarfs the compute savings.'
    },
    composesWith: ['tp', 'pp', 'dp', 'ep'],
    notes: 'Sometimes called "Ulysses" (DeepSpeed) or "Ring Attention" (variants). Targets the attention quadratic — for typical serving with shorter contexts, less common than TP.'
  },
  {
    id: 'dp',
    name: 'Data Parallelism',
    shortLabel: 'DP',
    collective: 'all-reduce',
    collectiveFrequency: 'training only (gradient sync per step); for serving, DP just means model replicas serving independent requests',
    volumeFormulaText:
      'Training: 2 · (N-1)/N · params · dtype bytes per step. ' +
      'Serving: zero comms cost — DP is "request-level replication", not collective.',
    shardingDim: 'batch (each replica processes a different request set)',
    applicableTo: ['dense', 'moe'],
    composesWith: ['tp', 'pp', 'ep'],
    notes: 'For inference, DP scales throughput linearly with replica count and adds no per-request latency. The calc currently models per-replica throughput; system throughput = replica count × per-replica.'
  }
]

import type { CalcInput, Dtype, AcceleratorVariant, MemoryResult, ModelArch } from './types'
import { bytesOf } from './dtypes'
import { perRankMemoryDivisors } from './parallelism'

const BYTES_PER_GB = 1024 ** 3

export function activeParams(model: ModelArch): number {
  return model.architecture.type === 'moe'
    ? model.architecture.activeParamCount
    : model.paramCount
}

export function kvBytesPerTokenPerLayer(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  if (att.type === 'linear-mla-hybrid') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  if (att.type === 'delta-hybrid') {
    // DeltaNet layers: no KV cache. Only Gated Attention layers store KV.
    return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
  }
  if (att.type === 'csa-hca-hybrid') {
    return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
  }
  return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

export function attentionDim(model: ModelArch): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') return att.kvLoraRank + att.qkRopeHeadDim
  if (att.type === 'linear-mla-hybrid') return att.kvLoraRank + att.qkRopeHeadDim
  if (att.type === 'delta-hybrid') return model.numHeads * model.headDim
  if (att.type === 'csa-hca-hybrid') return model.numHeads * model.headDim
  return model.numHeads * model.headDim
}

// forKv=true: KV storage (mla-dsa caches all tokens, topK only limits compute attention).
export function attendedSeqlenSummedOverLayers(model: ModelArch, seqlen: number, forKv = false): number {
  const att = model.attention
  if (att.type === 'hybrid') {
    if (att.numSlidingLayers + att.numGlobalLayers !== model.layers) {
      throw new Error(
        `hybrid layer counts must sum to model.layers: ` +
        `${att.numSlidingLayers} + ${att.numGlobalLayers} ≠ ${model.layers}`
      )
    }
    return att.numSlidingLayers * Math.min(seqlen, att.slidingWindow)
         + att.numGlobalLayers * seqlen
  }
  if (att.type === 'linear-mla-hybrid') {
    if (att.numLinearLayers + att.numFullLayers !== model.layers) {
      throw new Error(
        `linear-mla-hybrid layer counts must sum to model.layers: ` +
        `${att.numLinearLayers} + ${att.numFullLayers} ≠ ${model.layers}`
      )
    }
    return att.numFullLayers * seqlen
  }
  if (att.type === 'delta-hybrid') {
    if (att.numDeltaNetLayers + att.numFullLayers !== model.layers) {
      throw new Error(
        `delta-hybrid layer counts must sum to model.layers: ` +
        `${att.numDeltaNetLayers} + ${att.numFullLayers} ≠ ${model.layers}`
      )
    }
    // DeltaNet layers: no attention over sequence (constant-time state update).
    // Gated Attention layers: full sequence attention.
    return att.numFullLayers * seqlen
  }
  if (att.type === 'csa-hca-hybrid') {
    if (att.numSlidingLayers + att.numCsaLayers + att.numHcaLayers !== model.layers) {
      throw new Error(
        `csa-hca-hybrid layer counts must sum to model.layers: ` +
        `${att.numSlidingLayers} + ${att.numCsaLayers} + ${att.numHcaLayers} ≠ ${model.layers}`
      )
    }
    const csaCount = forKv ? (seqlen / att.csaCompressionM) : att.csaTopK
    return att.numSlidingLayers * Math.min(seqlen, att.slidingWindow)
         + att.numCsaLayers * (csaCount + att.slidingWindow)
         + att.numHcaLayers * (seqlen / att.hcaCompressionM + att.slidingWindow)
  }
  if (att.type === 'mla-dsa') return model.layers * (forKv ? seqlen : Math.min(seqlen, att.topK))
  const perLayer = att.type === 'sliding' ? Math.min(seqlen, att.window) : seqlen
  return model.layers * perLayer
}

// Constant per-request state bytes from linear-attention layers. Zero for non-linear models.
export function linearAttentionStateBytes(model: ModelArch, kvDtype: Dtype): number {
  if (model.attention.type !== 'linear-mla-hybrid') return 0
  const a = model.attention
  return a.numLinearLayers * a.numLinearHeads * a.linearHeadDim * a.linearHeadDim * bytesOf(kvDtype)
}

// FLOPs per token from linear-attention layers (constant in seqlen). Zero for non-linear models.
export function linearAttentionFlopsPerToken(model: ModelArch): number {
  if (model.attention.type !== 'linear-mla-hybrid') return 0
  const a = model.attention
  return 2 * a.numLinearLayers * a.numLinearHeads * a.linearHeadDim * a.linearHeadDim
}

// Constant per-request state bytes from DeltaNet (Gated DeltaNet) layers. Zero for non-DeltaNet models.
export function deltaStateBytes(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type !== 'delta-hybrid') return 0
  // State matrix per DeltaNet layer: numDeltaNetHeads × deltaHeadDim²
  return att.numDeltaNetLayers * att.numDeltaNetHeads * att.deltaHeadDim * att.deltaHeadDim * bytesOf(kvDtype)
}

// FLOPs per token from DeltaNet layers (constant in seqlen). Zero for non-DeltaNet models.
export function deltaAttentionFlopsPerToken(model: ModelArch): number {
  const att = model.attention
  if (att.type !== 'delta-hybrid') return 0
  // Per-layer DeltaNet FLOPs ≈ 2 × numDeltaNetHeads × deltaHeadDim²
  return 2 * att.numDeltaNetLayers * att.numDeltaNetHeads * att.deltaHeadDim * att.deltaHeadDim
}

function findVariant(input: CalcInput): AcceleratorVariant {
  const v = input.accelerator.variants.find(v => v.id === input.acceleratorVariantId)
  if (!v) throw new Error(`Variant ${input.acceleratorVariantId} not in ${input.accelerator.id}`)
  return v
}

export function computeMemory(input: CalcInput): MemoryResult {
  const { model, quant, workload } = input
  const variant = findVariant(input)
  const seqlen = workload.promptTokens + workload.outputTokens

  const weights = model.paramCount * bytesOf(quant.weights)
  const kvPerLayerPerToken = kvBytesPerTokenPerLayer(model, quant.kv)
  const attendedSeqlen = attendedSeqlenSummedOverLayers(model, seqlen, true)
  const kvCachePerRequest =
    kvPerLayerPerToken * attendedSeqlen
    + linearAttentionStateBytes(model, quant.kv)
    + deltaStateBytes(model, quant.kv)
  const kvCacheTotal = kvCachePerRequest * workload.concurrency

  // Coarse: one layer's attention + FFN buffer × small constant.
  // Assumes FlashAttention-style kernels (no materialized S×S matrix).
  const activationsPeak =
    workload.concurrency * workload.promptTokens *
    (model.hiddenDim + model.intermediateDim) * bytesOf(quant.activations) * 2

  const total = weights + kvCacheTotal + activationsPeak
  const hbmCapacityBytes = variant.hbmCapacityGB * BYTES_PER_GB
  const headroom = hbmCapacityBytes - total
  const fits = headroom >= 0

  let perRank: MemoryResult['perRank'] = undefined
  if (input.multiDevice) {
    const divisors = perRankMemoryDivisors(
      input.multiDevice.parallelism,
      input.multiDevice.parallelismDegrees,
      model
    )
    const rankWeights = weights / divisors.weights
    const perReplicaConcurrency = workload.concurrency / divisors.replicas
    const rankKvPerRequest = kvCachePerRequest / divisors.kv
    const rankKvTotal = rankKvPerRequest * perReplicaConcurrency
    const rankActivations = activationsPeak / divisors.activations
    const rankTotal = rankWeights + rankKvTotal + rankActivations
    const rankHeadroom = hbmCapacityBytes - rankTotal
    perRank = {
      weights: rankWeights,
      kvCachePerRequest: rankKvPerRequest,
      activationsPeak: rankActivations,
      total: rankTotal,
      headroom: rankHeadroom,
      fits: rankHeadroom >= 0
    }
  }

  return {
    weights,
    kvCachePerRequest,
    kvCacheTotal,
    activationsPeak,
    total,
    hbmCapacityGB: variant.hbmCapacityGB,
    headroom,
    fits,
    ...(perRank && { perRank })
  }
}

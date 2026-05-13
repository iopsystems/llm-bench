import type { CalcInput, Dtype, GpuVariant, MemoryResult, ModelArch } from './types'
import { bytesOf } from './dtypes'

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
  return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

export function attentionDim(model: ModelArch): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') return att.kvLoraRank + att.qkRopeHeadDim
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
  if (att.type === 'mla-dsa') return model.layers * (forKv ? seqlen : Math.min(seqlen, att.topK))
  const perLayer = att.type === 'sliding' ? Math.min(seqlen, att.window) : seqlen
  return model.layers * perLayer
}

function findVariant(input: CalcInput): GpuVariant {
  const v = input.gpu.variants.find(v => v.id === input.gpuVariantId)
  if (!v) throw new Error(`Variant ${input.gpuVariantId} not in ${input.gpu.id}`)
  return v
}

export function computeMemory(input: CalcInput): MemoryResult {
  const { model, quant, workload } = input
  const variant = findVariant(input)
  const seqlen = workload.promptTokens + workload.outputTokens

  const weights = model.paramCount * bytesOf(quant.weights)
  const kvPerLayerPerToken = kvBytesPerTokenPerLayer(model, quant.kv)
  const attendedSeqlen = attendedSeqlenSummedOverLayers(model, seqlen, true)
  const kvCachePerRequest = kvPerLayerPerToken * attendedSeqlen
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

  return {
    weights,
    kvCachePerRequest,
    kvCacheTotal,
    activationsPeak,
    total,
    hbmCapacityGB: variant.hbmCapacityGB,
    headroom,
    fits
  }
}

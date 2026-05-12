import type { AttentionConfig, CalcInput, Dtype, GpuVariant, MemoryResult, ModelArch } from './types'
import { bytesOf } from './dtypes'

const BYTES_PER_GB = 1024 ** 3

export function effectiveAttentionLength(rawSeqlen: number, attention: AttentionConfig): number {
  if (attention.type === 'sliding') return Math.min(rawSeqlen, attention.window)
  return rawSeqlen
}

export function activeParams(model: ModelArch): number {
  return model.architecture.type === 'moe'
    ? model.architecture.activeParamCount
    : model.paramCount
}

export function kvBytesPerToken(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla') {
    return model.layers * (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  return 2 * model.layers * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

export function attentionDim(model: ModelArch): number {
  const att = model.attention
  if (att.type === 'mla') return att.kvLoraRank + att.qkRopeHeadDim
  return model.numHeads * model.headDim
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
  const kvPerTokenPerRequest = kvBytesPerToken(model, quant.kv)
  const effSeqlen = effectiveAttentionLength(seqlen, model.attention)
  const kvCachePerRequest = kvPerTokenPerRequest * effSeqlen
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

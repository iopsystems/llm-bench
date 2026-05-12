import type { CalcInput, MemoryResult } from './types'
import { bytesOf } from './dtypes'

export function computeMemory(input: CalcInput): MemoryResult {
  const { model, quant, workload } = input
  const seqlen = workload.promptTokens + workload.outputTokens

  const weights = model.paramCount * bytesOf(quant.weights)
  const kvPerTokenPerRequest =
    2 * model.layers * model.numKvHeads * model.headDim * bytesOf(quant.kv)
  const kvCachePerRequest = kvPerTokenPerRequest * seqlen
  const kvCacheTotal = kvCachePerRequest * workload.concurrency

  return {
    weights,
    kvCachePerRequest,
    kvCacheTotal,
    activationsPeak: 0,
    total: 0,
    hbmCapacityGB: 0,
    headroom: 0,
    fits: false
  }
}

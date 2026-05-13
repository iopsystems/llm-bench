import type { CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'
import {
  attendedSeqlenSummedOverLayers,
  activeParams,
  attentionDim,
  linearAttentionFlopsPerToken,
  linearAttentionStateBytes
} from './memory'
import { bytesOf } from './dtypes'

export function computeDecode(
  input: CalcInput,
  opPoint: AcceleratorOperatingPoint,
  memory: MemoryResult
): PerfTier['decode'] {
  const { model, quant, workload } = input
  const avgSeqlen = workload.promptTokens + workload.outputTokens / 2

  const flopsPerStep =
    (2 * activeParams(model)
     + 2 * attendedSeqlenSummedOverLayers(model, avgSeqlen) * attentionDim(model)
     + linearAttentionFlopsPerToken(model)) *
    workload.concurrency
  const bytesPerStep =
    activeParams(model) * bytesOf(quant.weights) +
    memory.kvCachePerRequest * workload.concurrency +
    linearAttentionStateBytes(model, quant.kv) * workload.concurrency  // KDA state write-back

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  const { timeS, regime } = roofline({
    flops: flopsPerStep, bytes: bytesPerStep,
    tflops, bwGBs: opPoint.hbmBandwidthGBs
  })

  return {
    flopsPerStep,
    bytesPerStep,
    timePerTokenS: timeS,
    regime,
    aggregateTokensPerS: workload.concurrency / timeS
  }
}

import type { CalcInput, GpuOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'
import { effectiveAttentionLength, activeParams } from './memory'
import { bytesOf } from './dtypes'

export function computeDecode(
  input: CalcInput,
  opPoint: GpuOperatingPoint,
  memory: MemoryResult
): PerfTier['decode'] {
  const { model, quant, workload } = input
  const avgSeqlen = workload.promptTokens + workload.outputTokens / 2

  const effAvg = effectiveAttentionLength(avgSeqlen, model.attention)
  const flopsPerStep =
    (2 * activeParams(model) + 2 * model.layers * effAvg * model.hiddenDim) *
    workload.concurrency
  const bytesPerStep =
    activeParams(model) * bytesOf(quant.weights) +
    memory.kvCachePerRequest * workload.concurrency

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

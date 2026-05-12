import type { CalcInput, GpuOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'
import { effectiveAttentionLength } from './memory'

export function computeDecode(
  input: CalcInput,
  opPoint: GpuOperatingPoint,
  memory: MemoryResult
): PerfTier['decode'] {
  const { model, quant, workload } = input
  const avgSeqlen = workload.promptTokens + workload.outputTokens / 2

  const effAvg = effectiveAttentionLength(avgSeqlen, model.attention)
  const flopsPerStep =
    (2 * model.paramCount + 2 * model.layers * effAvg * model.hiddenDim) *
    workload.concurrency
  const bytesPerStep = memory.weights + memory.kvCachePerRequest * workload.concurrency

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

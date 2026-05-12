import type { CalcInput, GpuOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'
import { effectiveAttentionLength, activeParams, attentionDim } from './memory'

export function computePrefill(
  input: CalcInput,
  opPoint: GpuOperatingPoint,
  memory: MemoryResult
): PerfTier['prefill'] {
  const { model, quant, workload } = input
  const p = workload.promptTokens

  const effP = effectiveAttentionLength(p, model.attention)
  const flops =
    2 * activeParams(model) * p +
    2 * model.layers * p * effP * attentionDim(model)
  const bytes = memory.weights + memory.activationsPeak

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  const { timeS, regime } = roofline({
    flops, bytes, tflops, bwGBs: opPoint.hbmBandwidthGBs
  })
  return { flops, bytes, timeS, regime }
}

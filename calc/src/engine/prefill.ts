import type { CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'
import { attendedSeqlenSummedOverLayers, activeParams, attentionDim, linearAttentionFlopsPerToken } from './memory'

export function computePrefill(
  input: CalcInput,
  opPoint: AcceleratorOperatingPoint,
  memory: MemoryResult
): PerfTier['prefill'] {
  const { model, quant, workload } = input
  const p = workload.promptTokens

  const flops =
    2 * activeParams(model) * p +
    2 * p * attendedSeqlenSummedOverLayers(model, p) * attentionDim(model) +
    p * linearAttentionFlopsPerToken(model)
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

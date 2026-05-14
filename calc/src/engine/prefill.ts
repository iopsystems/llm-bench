import type { CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'
import { attendedSeqlenSummedOverLayers, activeParams, attentionDim, linearAttentionFlopsPerToken, deltaAttentionFlopsPerToken } from './memory'
import { commsBytesPerStep } from './parallelism'
import { INTERCONNECTS } from '../data/interconnects'

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
    p * linearAttentionFlopsPerToken(model) +
    p * deltaAttentionFlopsPerToken(model)
  const bytes = memory.weights + memory.activationsPeak

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  let commsBytes: number | undefined = undefined
  let interconnectBwGBs: number | undefined = undefined
  if (input.multiDevice) {
    const B = workload.promptTokens * workload.concurrency
    commsBytes = commsBytesPerStep(
      input.multiDevice.parallelism,
      input.multiDevice.parallelismDegrees,
      model,
      B,
      quant.activations
    )
    const ic = INTERCONNECTS.find(i => i.id === input.multiDevice!.system.interconnectId)
    if (ic) interconnectBwGBs = ic.perDirectionGBs ?? ic.perGpuBandwidthGBs / 2
  }

  const { timeS, regime } = roofline({
    flops, bytes, tflops, bwGBs: opPoint.hbmBandwidthGBs,
    commsBytes, interconnectBwGBs
  })
  return { flops, bytes, timeS, regime }
}

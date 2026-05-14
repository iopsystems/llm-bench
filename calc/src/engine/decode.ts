import type { CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'
import {
  attendedSeqlenSummedOverLayers,
  activeParams,
  attentionDim,
  linearAttentionFlopsPerToken,
  linearAttentionStateBytes,
  deltaStateBytes,
  deltaAttentionFlopsPerToken
} from './memory'
import { bytesOf } from './dtypes'
import { commsBytesPerStep } from './parallelism'
import { INTERCONNECTS } from '../data/interconnects'

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
     + linearAttentionFlopsPerToken(model)
     + deltaAttentionFlopsPerToken(model)) *
    workload.concurrency
  const bytesPerStep =
    activeParams(model) * bytesOf(quant.weights) +
    memory.kvCachePerRequest * workload.concurrency +
    linearAttentionStateBytes(model, quant.kv) * workload.concurrency +  // KDA state write-back
    deltaStateBytes(model, quant.kv) * workload.concurrency  // DeltaNet state write-back

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  let commsBytes: number | undefined = undefined
  let interconnectBwGBs: number | undefined = undefined
  if (input.multiDevice) {
    const B = workload.concurrency  // decode: one token per request per pass
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
    flops: flopsPerStep, bytes: bytesPerStep,
    tflops, bwGBs: opPoint.hbmBandwidthGBs,
    commsBytes, interconnectBwGBs
  })

  const mtpFactor = 1 + model.numNextnLayers
  return {
    flopsPerStep,
    bytesPerStep,
    timePerTokenS: timeS / mtpFactor,
    regime,
    aggregateTokensPerS: workload.concurrency * mtpFactor / timeS
  }
}

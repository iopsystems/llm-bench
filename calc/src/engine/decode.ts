import type { CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier, MultiDeviceConfig } from './types'
import { roofline } from './roofline'
import {
  attendedSeqlenSummedOverLayers,
  activeParams,
  attentionDim,
  linearAttentionFlopsPerToken,
  linearAttentionStateBytes,
  deltaStateBytes,
  deltaAttentionFlopsPerToken,
  mambaStateBytes,
  mambaFlopsPerToken
} from './memory'
import { bytesOf } from './dtypes'
import { commsBytesPerStep } from './parallelism'
import { INTERCONNECTS } from '../data/interconnects'

export function computeDecode(
  input: CalcInput,
  opPoint: AcceleratorOperatingPoint,
  memory: MemoryResult,
  multiDeviceOverride?: MultiDeviceConfig,
): PerfTier['decode'] {
  const { model, quant, workload } = input
  const multiDevice = multiDeviceOverride ?? input.decodeMultiDevice ?? input.multiDevice
  const avgSeqlen = workload.promptTokens + workload.outputTokens / 2

  const flopsPerStep =
    (2 * activeParams(model)
     + 2 * attendedSeqlenSummedOverLayers(model, avgSeqlen) * attentionDim(model)
     + linearAttentionFlopsPerToken(model)
     + deltaAttentionFlopsPerToken(model)
     + mambaFlopsPerToken(model)) *
    workload.concurrency
  const bytesPerStep =
    activeParams(model) * bytesOf(quant.weights) +
    memory.kvCachePerRequest * workload.concurrency +
    linearAttentionStateBytes(model, quant.kv) * workload.concurrency +  // KDA state write-back
    deltaStateBytes(model, quant.kv) * workload.concurrency +  // DeltaNet state write-back
    mambaStateBytes(model) * workload.concurrency  // Mamba2 SSM state write-back (fp32)

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  let commsBytes: number | undefined = undefined
  let interconnectBwGBs: number | undefined = undefined
  if (multiDevice) {
    const B = workload.concurrency  // decode: one token per request per pass
    commsBytes = commsBytesPerStep(
      multiDevice.parallelism,
      multiDevice.parallelismDegrees,
      model,
      B,
      quant.activations
    )
    const ic = INTERCONNECTS.find(i => i.id === multiDevice.system.interconnectId)
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
    aggregateTokensPerS: workload.concurrency * mtpFactor / timeS,
    ...(commsBytes !== undefined && { commsBytes })
  }
}

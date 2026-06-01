import type { CalcInput, CalcResult, PerfTier } from './types'
import { bytesOf } from './dtypes'
import { computeMemory } from './memory'
import { computePrefill } from './prefill'
import { computeDecode } from './decode'
import { DerivationBuilder } from './derivation'
import { INTERCONNECTS } from '../data/interconnects'

export function calculate(input: CalcInput): CalcResult {
  const variant = input.accelerator.variants.find(v => v.id === input.acceleratorVariantId)
  if (!variant) {
    throw new Error(`Variant ${input.acceleratorVariantId} not in ${input.accelerator.id}`)
  }

  // Validate activations dtype against each operating point up front, so the
  // error message names the actual accelerator and the supported alternatives
  // instead of leaking engine vocabulary.
  for (const op of variant.operatingPoints) {
    if (op.tflops[input.quant.activations] === undefined) {
      const supported = Object.keys(op.tflops).join(', ')
      throw new Error(
        `${input.accelerator.name} ${variant.label} has no ${input.quant.activations} ` +
        `compute throughput. Try: ${supported}.`
      )
    }
  }

  const memory = computeMemory(input)
  const d = new DerivationBuilder()

  d.add('weights', 'paramCount × bytes(weight_dtype)', memory.weights, 'bytes')
  d.add(
    'kv per token per request',
    '2 × layers × kv_heads × head_dim × bytes(kv_dtype)',
    memory.kvCachePerRequest / (input.workload.promptTokens + input.workload.outputTokens),
    'bytes'
  )
  d.add('kv per request', 'kv_per_token × (prompt + output)', memory.kvCachePerRequest, 'bytes')
  d.add('kv total', 'kv_per_request × concurrency', memory.kvCacheTotal, 'bytes')
  d.add(
    'activations peak (coarse)',
    'concurrency × prompt × (hidden + intermediate) × bytes(act_dtype) × 2',
    memory.activationsPeak, 'bytes'
  )
  d.add('memory total', 'weights + kv_total + activations_peak', memory.total, 'bytes')

  // Disaggregated serving: KV cache ships from prefill cluster to decode
  // cluster over a separate fabric. Adds a one-shot transfer time to TTFT.
  // For integrated serving (single cluster) this is 0.
  let kvTransferS = 0
  if (input.disaggKvTransferFabricId) {
    const fab = INTERCONNECTS.find(i => i.id === input.disaggKvTransferFabricId)
    if (fab) {
      const bw = fab.perDirectionGBs ?? fab.perGpuBandwidthGBs / 2
      kvTransferS = memory.kvCachePerRequest / (bw * 1e9)
    }
  }
  // Production-standard: prefill node emits the first decoded token locally
  // while KV transfer streams in parallel. Defaults true when disagg is on.
  const firstTokenOnPrefill =
    input.disaggFirstTokenOnPrefill ?? true

  const perf: Record<string, PerfTier> = {}
  for (const op of variant.operatingPoints) {
    const prefill = computePrefill(input, op, memory)
    const decode = computeDecode(input, op, memory)
    // TTFT composition under disagg:
    //   firstTokenOnPrefill=true:  ttft = prefill + first decode step (transfer
    //                               hidden in parallel with that decode step).
    //   firstTokenOnPrefill=false: ttft = prefill + full kv transfer (worst case).
    const ttftS = kvTransferS > 0 && firstTokenOnPrefill
      ? prefill.timeS + decode.timePerTokenS
      : prefill.timeS + kvTransferS
    perf[op.id] = {
      prefill, decode,
      ttftS,
      kvTransferS,
      // inputTokenRate stays on prefill (cluster throughput); ttftS includes
      // disagg overhead (user-facing latency to first decoded token).
      inputTokenRate: input.workload.promptTokens / prefill.timeS,
      outputTokenRate: decode.aggregateTokensPerS,
      ...(op.tflopsSources && { tflopsSources: op.tflopsSources }),
      ...(op.bandwidthSources && { bandwidthSources: op.bandwidthSources }),
      ...(op.asOf && { asOf: op.asOf }),
      ...(op.notes && { notes: op.notes })
    }
    d.add(
      `prefill time @ ${op.id}`,
      'max(prefill_flops / tflops, prefill_bytes / bw)',
      prefill.timeS, 's'
    )
    if (kvTransferS > 0) {
      d.add(
        `kv transfer time @ ${op.id}`,
        firstTokenOnPrefill
          ? 'kv_cache_per_request / disagg_fabric_bw (overlapped with first decode)'
          : 'kv_cache_per_request / disagg_fabric_bw',
        kvTransferS, 's'
      )
    }
    d.add(
      `decode time per token @ ${op.id}`,
      'max(decode_flops / tflops, decode_bytes / bw)',
      decode.timePerTokenS, 's'
    )
  }

  // bytesOf re-exported so consumers can read the same dtype table.
  void bytesOf

  return { memory, perf, derivation: d.steps() }
}

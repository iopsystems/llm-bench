import type { CalcInput, CalcResult, PerfTier } from './types'
import { bytesOf } from './dtypes'
import { computeMemory } from './memory'
import { computePrefill } from './prefill'
import { computeDecode } from './decode'
import { DerivationBuilder } from './derivation'

export function calculate(input: CalcInput): CalcResult {
  const variant = input.gpu.variants.find(v => v.id === input.gpuVariantId)
  if (!variant) {
    throw new Error(`Variant ${input.gpuVariantId} not in GPU ${input.gpu.id}`)
  }

  // Validate activations dtype against each operating point up front, so the
  // error message names the actual GPU and the supported alternatives instead
  // of leaking engine vocabulary.
  for (const op of variant.operatingPoints) {
    if (op.tflops[input.quant.activations] === undefined) {
      const supported = Object.keys(op.tflops).join(', ')
      throw new Error(
        `${input.gpu.name} ${variant.label} has no ${input.quant.activations} ` +
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

  const perf: Record<string, PerfTier> = {}
  for (const op of variant.operatingPoints) {
    const prefill = computePrefill(input, op, memory)
    const decode = computeDecode(input, op, memory)
    perf[op.id] = {
      prefill, decode,
      ttftS: prefill.timeS,
      inputTokenRate: input.workload.promptTokens / prefill.timeS,
      outputTokenRate: decode.aggregateTokensPerS
    }
    d.add(
      `prefill time @ ${op.id}`,
      'max(prefill_flops / tflops, prefill_bytes / bw)',
      prefill.timeS, 's'
    )
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

import { writable, derived, type Readable } from 'svelte/store'
import { GPUS, MODELS } from '../data'
import { calculate } from '../engine'
import type { CalcInput, CalcResult, Quantization, Workload } from '../engine/types'

const defaultGpu = GPUS[0]
const defaultModel = MODELS[0]

export const gpuId = writable(defaultGpu.id)
export const variantId = writable(defaultGpu.variants[0].id)
export const modelId = writable(defaultModel.id)

export const quant = writable<Quantization>({
  weights: 'fp16', kv: 'fp16', activations: 'fp16'
})
export const workload = writable<Workload>({
  promptTokens: 2048, outputTokens: 512, concurrency: 1
})

export const input: Readable<CalcInput | null> = derived(
  [gpuId, variantId, modelId, quant, workload],
  ([$gpuId, $variantId, $modelId, $quant, $workload]) => {
    const gpu = GPUS.find(g => g.id === $gpuId)
    const model = MODELS.find(m => m.id === $modelId)
    if (!gpu || !model) return null
    if (!gpu.variants.find(v => v.id === $variantId)) return null
    return { gpu, gpuVariantId: $variantId, model, quant: $quant, workload: $workload }
  }
)

interface Computed {
  result: CalcResult | null
  error: string | null
}

const computed: Readable<Computed> = derived(input, $input => {
  if (!$input) return { result: null, error: null }
  try { return { result: calculate($input), error: null } }
  catch (err) { return { result: null, error: (err as Error).message } }
})

export const result: Readable<CalcResult | null> = derived(computed, $c => $c.result)
export const error: Readable<string | null> = derived(computed, $c => $c.error)

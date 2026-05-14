import { writable, derived, type Readable } from 'svelte/store'
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import { calculate } from '../engine'
import { defaultParallelism, type ParallelismConfig } from '../engine/parallelism'
import type { CalcInput, CalcResult, MultiDeviceConfig, Quantization, Workload } from '../engine/types'

const defaultAccelerator = ACCELERATORS[0]
const defaultModel = MODELS[0]

// `systemId` is empty string when user picks a single accelerator (no multi-device).
// Non-empty when a MultiAcceleratorSystem is selected.
export const acceleratorId = writable(defaultAccelerator.id)
export const variantId = writable(defaultAccelerator.variants[0].id)
export const systemId = writable<string>('')
export const modelId = writable(defaultModel.id)

// User overrides for parallelism. null means "use defaultParallelism".
export const parallelismOverride = writable<ParallelismConfig | null>(null)

// Disaggregated serving: id of the inter-cluster fabric used to ship KV cache
// from prefill to decode. Empty string means integrated (no disagg).
export const disaggKvTransferFabricId = writable<string>('')
// Production-standard optimization: prefill node emits the first token while
// KV streams. Defaults true; uncheck to model the worst-case sequential handoff.
export const disaggFirstTokenOnPrefill = writable<boolean>(true)

export const quant = writable<Quantization>({
  weights: 'fp16', kv: 'fp16', activations: 'fp16'
})
export const workload = writable<Workload>({
  promptTokens: 2048, outputTokens: 512, concurrency: 1
})

export const multiDevice: Readable<MultiDeviceConfig | undefined> = derived(
  [systemId, modelId, parallelismOverride, disaggKvTransferFabricId, disaggFirstTokenOnPrefill],
  ([$systemId, $modelId, $override, $disagg, $firstTokenOnPrefill]) => {
    if (!$systemId) return undefined
    const system = SYSTEMS.find(s => s.id === $systemId)
    const model = MODELS.find(m => m.id === $modelId)
    if (!system || !model) return undefined
    const pc = $override ?? defaultParallelism(system, model)
    return {
      system,
      parallelism: pc.parallelism,
      parallelismDegrees: pc.parallelismDegrees,
      ...($disagg && {
        disaggKvTransferFabricId: $disagg,
        disaggFirstTokenOnPrefill: $firstTokenOnPrefill,
      })
    }
  }
)

export const input: Readable<CalcInput | null> = derived(
  [acceleratorId, variantId, modelId, quant, workload, multiDevice, systemId],
  ([$acceleratorId, $variantId, $modelId, $quant, $workload, $multiDevice, $systemId]) => {
    // When a system is selected, resolve accelerator from the system's chip ref.
    let accelerator
    let resolvedVariantId: string
    if ($systemId && $multiDevice) {
      accelerator = ACCELERATORS.find(a => a.id === $multiDevice.system.accelerator.id)
      resolvedVariantId = $multiDevice.system.accelerator.variantId
    } else {
      accelerator = ACCELERATORS.find(a => a.id === $acceleratorId)
      resolvedVariantId = $variantId
    }
    const model = MODELS.find(m => m.id === $modelId)
    if (!accelerator || !model) return null
    if (!accelerator.variants.find(v => v.id === resolvedVariantId)) return null
    return {
      accelerator,
      acceleratorVariantId: resolvedVariantId,
      model,
      quant: $quant,
      workload: $workload,
      ...($multiDevice && { multiDevice: $multiDevice })
    }
  }
)

interface Computed { result: CalcResult | null; error: string | null }

const computed: Readable<Computed> = derived(input, $input => {
  if (!$input) return { result: null, error: null }
  try { return { result: calculate($input), error: null } }
  catch (err) { return { result: null, error: (err as Error).message } }
})

export const result: Readable<CalcResult | null> = derived(computed, $c => $c.result)
export const error: Readable<string | null> = derived(computed, $c => $c.error)

import { writable, derived, type Readable } from 'svelte/store'
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import { calculate } from '../engine'
import { defaultParallelism, type ParallelismConfig } from '../engine/parallelism'
import type { CalcInput, CalcResult, MultiDeviceConfig, Quantization, Workload } from '../engine/types'

const defaultAccelerator = ACCELERATORS[0]
const defaultModel = MODELS[0]

// Derivation drawer open state — shared so App can reflow the main content
// out from under the fixed drawer instead of letting it overlap.
export const showMath = writable(false)

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

// Heterogeneous PD-disagg — separate hw + parallelism per cluster, fully
// decoupled from the monolithic (shared) stores. When `heterogeneous` is
// false these are ignored and the disagg block reuses the shared hw (symmetric
// comparison). When true, both clusters get their own overrides; the
// DisaggInputPanel seeds them from shared on toggle-on, so the user starts
// symmetric and changes one side at a time. Empty/null fields fall back to
// the shared (monolithic) values — that's the path old het=1 URLs without
// a1/v1 follow, and what we render before the toggle-on seed lands.
export const prefillAcceleratorId       = writable<string>('')
export const prefillVariantId           = writable<string>('')
export const prefillSystemId            = writable<string>('')
export const prefillParallelismOverride = writable<ParallelismConfig | null>(null)
export const decodeAcceleratorId        = writable<string>('')
export const decodeVariantId            = writable<string>('')
export const decodeSystemId             = writable<string>('')
export const decodeParallelismOverride  = writable<ParallelismConfig | null>(null)
export const heterogeneous              = writable<boolean>(false)

// Initial quant follows the default model's native precision; KV defaults to
// fp16 because cache quant is an independent serving-side axis, not a
// property of how the weights ship.
export const quant = writable<Quantization>({
  weights: defaultModel.nativeDtype, kv: 'fp16', activations: defaultModel.nativeDtype
})

// Wire the model→quant coupling: switching models reseeds weights+activations
// to the new model's nativeDtype (KV untouched). Call once at startup AFTER
// readUrlIntoStores(); the initial subscribe fire is skipped so URL-provided
// quant survives load. Fresh-load defaults are handled by the store's initial
// value above (and applyToStores for URL-with-model-but-no-quant).
export function initNativeDtypeSync(): () => void {
  let first = true
  return modelId.subscribe($modelId => {
    if (first) { first = false; return }
    const m = MODELS.find(x => x.id === $modelId)
    if (!m) return
    quant.update(q => ({ ...q, weights: m.nativeDtype, activations: m.nativeDtype }))
  })
}

export const workload = writable<Workload>({
  promptTokens: 2048, outputTokens: 512, concurrency: 1
})

export const multiDevice: Readable<MultiDeviceConfig | undefined> = derived(
  [systemId, modelId, parallelismOverride],
  ([$systemId, $modelId, $override]) => {
    if (!$systemId) return undefined
    const system = SYSTEMS.find(s => s.id === $systemId)
    const model = MODELS.find(m => m.id === $modelId)
    if (!system || !model) return undefined
    const pc = $override ?? defaultParallelism(system, model)
    return {
      system,
      parallelism: pc.parallelism,
      parallelismDegrees: pc.parallelismDegrees,
    }
  }
)

export const prefillMultiDevice: Readable<MultiDeviceConfig | undefined> = derived(
  [prefillSystemId, modelId, prefillParallelismOverride],
  ([$prefillSystemId, $modelId, $override]) => {
    if (!$prefillSystemId) return undefined
    const system = SYSTEMS.find(s => s.id === $prefillSystemId)
    const model = MODELS.find(m => m.id === $modelId)
    if (!system || !model) return undefined
    const pc = $override ?? defaultParallelism(system, model)
    return {
      system,
      parallelism: pc.parallelism,
      parallelismDegrees: pc.parallelismDegrees,
    }
  }
)

export const decodeMultiDevice: Readable<MultiDeviceConfig | undefined> = derived(
  [decodeSystemId, modelId, decodeParallelismOverride],
  ([$decodeSystemId, $modelId, $override]) => {
    if (!$decodeSystemId) return undefined
    const system = SYSTEMS.find(s => s.id === $decodeSystemId)
    const model = MODELS.find(m => m.id === $modelId)
    if (!system || !model) return undefined
    const pc = $override ?? defaultParallelism(system, model)
    return {
      system,
      parallelism: pc.parallelism,
      parallelismDegrees: pc.parallelismDegrees,
    }
  }
)

export const input: Readable<CalcInput | null> = derived(
  [acceleratorId, variantId, modelId, quant, workload, multiDevice, systemId,
   disaggKvTransferFabricId, disaggFirstTokenOnPrefill],
  ([$acceleratorId, $variantId, $modelId, $quant, $workload, $multiDevice, $systemId,
    $disagg, $firstTokenOnPrefill]) => {
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
      ...($multiDevice && { multiDevice: $multiDevice }),
      ...($disagg && {
        disaggKvTransferFabricId: $disagg,
        disaggFirstTokenOnPrefill: $firstTokenOnPrefill,
      }),
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

// --- Single-request simulator ---
// The simulator tab renders two stacked configurations (monolithic + disagg)
// from the same shared inputs. Each block gets its own derived CalcInput +
// CalcResult; the monolithic side nulls the disagg fields, the disagg side
// passes them through. Concurrency is clamped to 1 in both (sim is by
// definition single-request, regardless of what the shared workload carries).
export const simInputMonolithic: Readable<CalcInput | null> = derived(input, $input => {
  if (!$input) return null
  return {
    ...$input,
    workload: { ...$input.workload, concurrency: 1 },
    disaggKvTransferFabricId: undefined,
    disaggFirstTokenOnPrefill: undefined,
  }
})

export const simInputDisagg: Readable<CalcInput | null> = derived(
  [input, heterogeneous,
   prefillAcceleratorId, prefillVariantId, prefillSystemId, prefillMultiDevice,
   decodeAcceleratorId, decodeVariantId, decodeSystemId, decodeMultiDevice],
  ([$input, $het,
    $prefillAcceleratorId, $prefillVariantId, $prefillSystemId, $prefillMultiDevice,
    $decodeAcceleratorId, $decodeVariantId, $decodeSystemId, $decodeMultiDevice]) => {
    if (!$input) return null
    const base: CalcInput = {
      ...$input,
      workload: { ...$input.workload, concurrency: 1 },
    }
    if (!$het) return base
    // Heterogeneous: both clusters can override the shared (monolithic) hw.
    // Empty override stores fall back to shared, so old het=1 URLs (which
    // only carried decode-side keys) keep working — disagg prefill =
    // monolithic in that case, identical to pre-decoupling behavior.
    //
    // "Override active" is signaled by *either* system or accelerator being
    // set; we need this distinction so that picking a single-chip override
    // on a side that originally had a multiDevice doesn't accidentally
    // inherit the shared multiDevice.
    const prefillOverride = !!$prefillSystemId || !!$prefillAcceleratorId
    let prefillAccelerator = $input.accelerator
    let prefillVariantIdResolved = $input.acceleratorVariantId
    let prefillMD = $input.multiDevice
    if (prefillOverride) {
      if ($prefillSystemId && $prefillMultiDevice) {
        prefillAccelerator = ACCELERATORS.find(a => a.id === $prefillMultiDevice.system.accelerator.id) ?? $input.accelerator
        prefillVariantIdResolved = $prefillMultiDevice.system.accelerator.variantId
        prefillMD = $prefillMultiDevice
      } else if ($prefillAcceleratorId) {
        const found = ACCELERATORS.find(a => a.id === $prefillAcceleratorId)
        if (found) {
          prefillAccelerator = found
          prefillVariantIdResolved = $prefillVariantId || found.variants[0].id
        }
        prefillMD = undefined
      }
    }

    // Decode side falls back to the (possibly-overridden) prefill cluster,
    // not the shared monolithic — so changing only the prefill cluster
    // propagates to decode until decode is itself explicitly overridden.
    const decodeOverride = !!$decodeSystemId || !!$decodeAcceleratorId
    let decodeAccelerator = prefillAccelerator
    let decodeAcceleratorVariantId = prefillVariantIdResolved
    let decodeMD = prefillMD
    if (decodeOverride) {
      if ($decodeSystemId && $decodeMultiDevice) {
        decodeAccelerator = ACCELERATORS.find(a => a.id === $decodeMultiDevice.system.accelerator.id) ?? prefillAccelerator
        decodeAcceleratorVariantId = $decodeMultiDevice.system.accelerator.variantId
        decodeMD = $decodeMultiDevice
      } else if ($decodeAcceleratorId) {
        const found = ACCELERATORS.find(a => a.id === $decodeAcceleratorId)
        if (found) {
          decodeAccelerator = found
          decodeAcceleratorVariantId = $decodeVariantId || found.variants[0].id
        }
        decodeMD = undefined
      }
    }

    return {
      ...base,
      accelerator: prefillAccelerator,
      acceleratorVariantId: prefillVariantIdResolved,
      multiDevice: prefillMD,
      decodeAccelerator,
      decodeAcceleratorVariantId,
      ...(decodeMD && { decodeMultiDevice: decodeMD }),
    }
  }
)

interface SimComputed { result: CalcResult | null; error: string | null }
function safeCalc($input: CalcInput | null): SimComputed {
  if (!$input) return { result: null, error: null }
  try { return { result: calculate($input), error: null } }
  catch (err) { return { result: null, error: (err as Error).message } }
}

const simComputedMonolithic: Readable<SimComputed> = derived(simInputMonolithic, safeCalc)
const simComputedDisagg:     Readable<SimComputed> = derived(simInputDisagg,     safeCalc)

export const simResultMonolithic: Readable<CalcResult | null> = derived(simComputedMonolithic, $c => $c.result)
export const simResultDisagg:     Readable<CalcResult | null> = derived(simComputedDisagg,     $c => $c.result)
export const simError:       Readable<string | null> = derived(simComputedMonolithic, $c => $c.error)
// Disagg can error independently of monolithic — e.g. heterogeneous decode hw
// lacks the workload's activations dtype. Surface separately so the disagg
// block can render an inline error instead of silently disappearing.
export const simErrorDisagg: Readable<string | null> = derived(simComputedDisagg,     $c => $c.error)

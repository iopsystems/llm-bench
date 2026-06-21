import { writable, derived, type Readable } from 'svelte/store'
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import { calculate } from '../engine'
import { defaultParallelism, type ParallelismConfig } from '../engine/parallelism'
import type { CalcInput, CalcResult, Dtype, MultiDeviceConfig, Quantization, Workload } from '../engine/types'
import { computeNMax } from '../engine/queueModel'
import { groupedDisaggFabrics } from './disaggFabrics'

const defaultAccelerator = ACCELERATORS[0]
const defaultModel = MODELS[0]

// Fastest fabric eligible for PD-disagg KV transfer on the default accelerator.
// `groupedDisaggFabrics` sorts each group by BW descending; pick the highest-BW
// candidate across both groups so the Sim tab opens with disagg pre-armed on
// the most aggressive realistic interconnect. Falls back to '' (= disagg off)
// only if neither group has any entries — defensive; practically never fires.
function pickFastestDisaggFabric(): string {
  const groups = groupedDisaggFabrics(defaultAccelerator.id)
  const candidates = [...groups.scaleUp, ...groups.scaleOut]
  if (candidates.length === 0) return ''
  candidates.sort((a, b) => b.perGpuBandwidthGBs - a.perGpuBandwidthGBs)
  return candidates[0].id
}

// Derivation drawer open state — shared so App can reflow the main content
// out from under the fixed drawer instead of letting it overlap.
export const showMath = writable(false)

// In-memory toggle: when true, consumer-tier accelerators (RTX 4090/5090,
// Apple M-series, Radeon RX, etc.) appear in the accelerator pickers and
// the Info-tab catalog. Default false — the calc app is serving-focused so
// datacenter SKUs are the primary surface. URL state isn't needed; the
// auto-show-current-selection rule in `filterByTier` handles shared links
// that point to a consumer SKU.
export const showConsumerSkus = writable<boolean>(false)

// `systemId` is empty string when user picks a single accelerator (no multi-device).
// Non-empty when a MultiAcceleratorSystem is selected.
export const acceleratorId = writable(defaultAccelerator.id)
export const variantId = writable(defaultAccelerator.variants[0].id)
export const systemId = writable<string>('')
export const modelId = writable(defaultModel.id)

// User overrides for parallelism. null means "use defaultParallelism".
export const parallelismOverride = writable<ParallelismConfig | null>(null)

// User override for in-flight count. null ⇒ "use computed nMax". The Calc-tab
// concurrency input and the Sim-tab LoadSection slider both bind to this
// store; their displayed defaults differ (nMaxCalc vs nMaxDecode) but the
// override is shared.
export const concurrencyOverride = writable<number | null>(null)

// Disaggregated serving (Calc tab): id of the inter-cluster fabric used to
// ship KV cache from prefill to decode. Empty string means integrated (no
// disagg). Calc-tab default is OFF — the Calc tab is a sizing tool that opens
// on monolithic by default; user opts into disagg by picking a fabric.
export const disaggKvTransferFabricId = writable<string>('')
// Production-standard optimization: prefill node emits the first token while
// KV streams. Defaults true; uncheck to model the worst-case sequential handoff.
export const disaggFirstTokenOnPrefill = writable<boolean>(true)

// Disaggregated serving (Sim tab): separate from the Calc-tab store because
// the Sim tab opens with disagg pre-armed on the fastest eligible fabric —
// it's a deployment-architecture exploration tool, so the richer (disagg)
// view is the default first impression. Calc and Sim are independent disagg
// configs; URL state encodes them with separate keys (dk for Calc, sdk for
// Sim).
export const simDisaggKvTransferFabricId = writable<string>(pickFastestDisaggFabric())
export const simDisaggFirstTokenOnPrefill = writable<boolean>(true)

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

// Activations default that pairs with a model's nativeDtype. 4-bit natives
// (Kimi K2.5 int4 W4A16-QAT, gpt-oss mxfp4) are weight-only ship formats:
// the matmuls run in bf16 after in-kernel dequant — no current datacenter
// chip exposes int4 tensor cores (Hopper dropped them; Blackwell's 4-bit
// path is fp4) and no accelerator entry lists 4-bit TFLOPS, so seeding
// activations to the ship format would make every SKU throw.
export function defaultActivationsFor(native: Dtype): Dtype {
  return native === 'int4' || native === 'fp4' ? 'bf16' : native
}

// Initial quant follows the default model's native precision; KV defaults to
// fp16 because cache quant is an independent serving-side axis, not a
// property of how the weights ship.
export const quant = writable<Quantization>({
  weights: defaultModel.nativeDtype, kv: 'fp16',
  activations: defaultActivationsFor(defaultModel.nativeDtype)
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
    quant.update(q => ({ ...q, weights: m.nativeDtype, activations: defaultActivationsFor(m.nativeDtype) }))
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

// nMaxCalc: KV-cap ceiling computed against the Calc-tab (shared) input.
// Drives the Calc-tab concurrency default. Derived from the raw stores (not
// from `input`) to break the circular dep — `input` will consume
// `effectiveConcurrency`, which derives from `nMaxCalc`.
export const nMaxCalc: Readable<number> = derived(
  [acceleratorId, variantId, modelId, quant, workload, multiDevice, systemId],
  ([$acceleratorId, $variantId, $modelId, $quant, $workload, $multiDevice, $systemId]) => {
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
    if (!accelerator || !model) return 0
    if (!accelerator.variants.find(v => v.id === resolvedVariantId)) return 0
    const probe: CalcInput = {
      accelerator, acceleratorVariantId: resolvedVariantId, model,
      quant: $quant, workload: { ...$workload, concurrency: 1 },
      ...($multiDevice && { multiDevice: $multiDevice }),
    }
    return computeNMax(probe, 'prefill').nMax
  }
)

// Effective concurrency for Calc-tab consumers. Floor at 1 so the engine
// never sees concurrency=0 (would zero out tokens-per-step math).
export const effectiveConcurrency: Readable<number> = derived(
  [concurrencyOverride, nMaxCalc],
  ([$override, $nMax]) => $override ?? Math.max(1, $nMax)
)

export const input: Readable<CalcInput | null> = derived(
  [acceleratorId, variantId, modelId, quant, workload, multiDevice, systemId,
   disaggKvTransferFabricId, disaggFirstTokenOnPrefill, effectiveConcurrency],
  ([$acceleratorId, $variantId, $modelId, $quant, $workload, $multiDevice, $systemId,
    $disagg, $firstTokenOnPrefill, $effectiveConcurrency]) => {
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
      workload: { ...$workload, concurrency: $effectiveConcurrency },
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
   decodeAcceleratorId, decodeVariantId, decodeSystemId, decodeMultiDevice,
   simDisaggKvTransferFabricId, simDisaggFirstTokenOnPrefill],
  ([$input, $het,
    $prefillAcceleratorId, $prefillVariantId, $prefillSystemId, $prefillMultiDevice,
    $decodeAcceleratorId, $decodeVariantId, $decodeSystemId, $decodeMultiDevice,
    $simFabric, $simFirstToken]) => {
    if (!$input) return null
    // Sim-tab disagg config is independent of Calc-tab's — overlay the Sim
    // stores onto the base input so the Sim view's "disagg on" doesn't
    // depend on Calc's disagg picker (and vice versa).
    const base: CalcInput = {
      ...$input,
      workload: { ...$input.workload, concurrency: 1 },
      ...($simFabric
        ? { disaggKvTransferFabricId: $simFabric, disaggFirstTokenOnPrefill: $simFirstToken }
        : { disaggKvTransferFabricId: undefined, disaggFirstTokenOnPrefill: undefined }),
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

// nMaxDecode: KV-cap ceiling for the disagg decode cluster (heterogeneity
// aware via simInputDisagg, which already clamps concurrency=1 — no circular
// dep). Drives the LoadSection slider default and clamp.
export const nMaxDecode: Readable<number> = derived(
  [simInputDisagg],
  ([$d]) => $d ? computeNMax($d).nMax : 0
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

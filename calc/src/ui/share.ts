// Shareable-URL state encoding.
//
// The calculator's entire input set lives in the Svelte stores in stores.ts.
// To make a configuration shareable, we mirror those stores into the URL hash
// (e.g. `#a=h100&v=sxm-80&m=llama-3.3-70b&w=fp16&kv=fp16&ac=fp16&pt=2048&ot=512&c=1`).
// The hash auto-updates as the user adjusts inputs, so they can copy the
// browser URL at any time and paste it to someone else — recipient sees the
// same configuration on load.
//
// Hash (vs query string) keeps SPA routing trivial: the asset server never
// sees it, and `history.replaceState` updates the bar without reloading.

import { get } from 'svelte/store'
import {
  acceleratorId, variantId, systemId, modelId,
  parallelismOverride, disaggKvTransferFabricId, disaggFirstTokenOnPrefill,
  quant, workload, concurrencyOverride, defaultActivationsFor,
  heterogeneous,
  prefillAcceleratorId, prefillVariantId, prefillSystemId, prefillParallelismOverride,
  decodeAcceleratorId, decodeVariantId, decodeSystemId, decodeParallelismOverride,
} from './stores'
import { parseRoute } from './route'
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import { INTERCONNECTS } from '../data/interconnects'
import type { Dtype, Quantization, Workload } from '../engine/types'
import type { ParallelismConfig } from '../engine/parallelism'

type ParallelismId = 'tp' | 'pp' | 'ep' | 'sp' | 'cp' | 'dp'

const DTYPES: readonly Dtype[] = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4']
const PARALLELISM_IDS: readonly ParallelismId[] = ['tp', 'pp', 'ep', 'sp', 'cp', 'dp']

function isDtype(s: string): s is Dtype {
  return (DTYPES as readonly string[]).includes(s)
}

function isParallelismId(s: string): s is ParallelismId {
  return (PARALLELISM_IDS as readonly string[]).includes(s)
}

export interface ShareableState {
  acceleratorId: string
  variantId: string
  systemId: string
  modelId: string
  quant: Quantization
  workload: Workload
  parallelismOverride: ParallelismConfig | null
  disaggKvTransferFabricId: string
  disaggFirstTokenOnPrefill: boolean

  // Heterogeneous PD-disagg — only encoded when heterogeneous is true.
  // Prefill-side overrides (a1/v1/s1/p1) are emitted only when explicitly
  // set; empty fields fall back to the shared (monolithic) hw on decode.
  heterogeneous: boolean
  prefillAcceleratorId: string
  prefillVariantId: string
  prefillSystemId: string
  prefillParallelismOverride: ParallelismConfig | null
  decodeAcceleratorId: string
  decodeVariantId: string
  decodeSystemId: string
  decodeParallelismOverride: ParallelismConfig | null

  // Top-level override — null means "use auto/default". Decoupled from
  // workload so the workload object stays stable for in-memory math.
  concurrencyOverride: number | null
}

// Encode state to a URL-search-style string (no leading `#`).
// When a system is selected, the underlying accelerator id is implied by the
// system definition, so we emit `s=` and skip `a=`/`v=`.
export function encodeState(state: ShareableState): string {
  const p = new URLSearchParams()
  if (state.systemId) {
    p.set('s', state.systemId)
  } else {
    p.set('a', state.acceleratorId)
    p.set('v', state.variantId)
  }
  p.set('m', state.modelId)
  p.set('w', state.quant.weights)
  p.set('kv', state.quant.kv)
  p.set('ac', state.quant.activations)
  p.set('pt', String(state.workload.promptTokens))
  p.set('ot', String(state.workload.outputTokens))
  if (state.concurrencyOverride !== null) {
    p.set('c', String(state.concurrencyOverride))
  }
  if (state.parallelismOverride) {
    p.set('p', encodeParallelism(state.parallelismOverride))
  }
  if (state.disaggKvTransferFabricId) {
    p.set('dk', state.disaggKvTransferFabricId)
    // `df=1` is the default — only emit when the user opted into the
    // worst-case sequential handoff.
    if (!state.disaggFirstTokenOnPrefill) p.set('df', '0')
  }
  if (state.heterogeneous) {
    p.set('het', '1')
    if (state.prefillSystemId) {
      p.set('s1', state.prefillSystemId)
    } else if (state.prefillAcceleratorId) {
      p.set('a1', state.prefillAcceleratorId)
      if (state.prefillVariantId) p.set('v1', state.prefillVariantId)
    }
    if (state.prefillParallelismOverride) {
      p.set('p1', encodeParallelism(state.prefillParallelismOverride))
    }
    if (state.decodeSystemId) {
      p.set('s2', state.decodeSystemId)
    } else if (state.decodeAcceleratorId) {
      p.set('a2', state.decodeAcceleratorId)
      if (state.decodeVariantId) p.set('v2', state.decodeVariantId)
    }
    if (state.decodeParallelismOverride) {
      p.set('p2', encodeParallelism(state.decodeParallelismOverride))
    }
  }
  return p.toString()
}

// Decode a URL-search-style string into a partial state. Only keys present in
// the input appear in the output. Invalid values (unknown ids, malformed
// numbers, unknown dtypes) are silently dropped — a partial restore is better
// than a thrown error when someone shares a URL across versions.
export function decodeState(hash: string): Partial<ShareableState> {
  const params = new URLSearchParams(hash)
  const out: Partial<ShareableState> = {}

  // System takes precedence; ignore a/v if s is present and valid.
  const s = params.get('s')
  if (s !== null) {
    const sys = SYSTEMS.find(x => x.id === s)
    if (sys) {
      out.systemId = s
      // Pre-seed accelerator + variant from the system so toggling back to
      // single-chip mode lands on a sensible default rather than an unrelated
      // recipient default.
      out.acceleratorId = sys.accelerator.id
      out.variantId = sys.accelerator.variantId
    }
  } else if (params.has('a')) {
    out.systemId = ''
    const a = params.get('a')!
    const accel = ACCELERATORS.find(x => x.id === a)
    if (accel) {
      out.acceleratorId = a
      const v = params.get('v')
      if (v && accel.variants.find(x => x.id === v)) {
        out.variantId = v
      } else {
        out.variantId = accel.variants[0].id
      }
    }
  }

  const m = params.get('m')
  if (m && MODELS.find(x => x.id === m)) out.modelId = m

  const w = params.get('w')
  const kv = params.get('kv')
  const ac = params.get('ac')
  if (w && kv && ac && isDtype(w) && isDtype(kv) && isDtype(ac)) {
    out.quant = { weights: w, kv, activations: ac }
  }

  const pt = params.get('pt')
  const ot = params.get('ot')
  if (pt !== null || ot !== null) {
    const wl: Partial<Workload> = {}
    if (pt !== null) {
      const n = parseInt(pt, 10)
      if (Number.isFinite(n) && n > 0) wl.promptTokens = n
    }
    if (ot !== null) {
      const n = parseInt(ot, 10)
      if (Number.isFinite(n) && n > 0) wl.outputTokens = n
    }
    if (Object.keys(wl).length > 0) out.workload = wl as Workload
  }

  // concurrencyOverride: standalone top-level key (was nested in workload pre-decoupling).
  const c = params.get('c')
  if (c !== null) {
    const n = parseInt(c, 10)
    if (Number.isFinite(n) && n > 0) out.concurrencyOverride = n
  }

  if (params.has('p')) {
    const pc = decodeParallelism(params.get('p')!)
    if (pc) out.parallelismOverride = pc
    else out.parallelismOverride = null
  }

  if (params.has('dk')) {
    const dk = params.get('dk')!
    if (dk && INTERCONNECTS.find(i => i.id === dk)) {
      out.disaggKvTransferFabricId = dk
      out.disaggFirstTokenOnPrefill = params.get('df') !== '0'
    } else if (dk === '') {
      out.disaggKvTransferFabricId = ''
    }
  }

  if (params.get('het') === '1') {
    out.heterogeneous = true
    const s1 = params.get('s1')
    if (s1 !== null) {
      const sys = SYSTEMS.find(x => x.id === s1)
      if (sys) {
        out.prefillSystemId = s1
        out.prefillAcceleratorId = sys.accelerator.id
        out.prefillVariantId = sys.accelerator.variantId
      }
    } else if (params.has('a1')) {
      const a1 = params.get('a1')!
      const accel = ACCELERATORS.find(x => x.id === a1)
      if (accel) {
        out.prefillSystemId = ''
        out.prefillAcceleratorId = a1
        const v1 = params.get('v1')
        out.prefillVariantId = v1 && accel.variants.find(x => x.id === v1)
          ? v1 : accel.variants[0].id
      }
    }
    if (params.has('p1')) {
      const pc = decodeParallelism(params.get('p1')!)
      out.prefillParallelismOverride = pc ?? null
    }
    const s2 = params.get('s2')
    if (s2 !== null) {
      const sys = SYSTEMS.find(x => x.id === s2)
      if (sys) {
        out.decodeSystemId = s2
        out.decodeAcceleratorId = sys.accelerator.id
        out.decodeVariantId = sys.accelerator.variantId
      }
    } else if (params.has('a2')) {
      const a2 = params.get('a2')!
      const accel = ACCELERATORS.find(x => x.id === a2)
      if (accel) {
        out.decodeSystemId = ''
        out.decodeAcceleratorId = a2
        const v2 = params.get('v2')
        out.decodeVariantId = v2 && accel.variants.find(x => x.id === v2)
          ? v2 : accel.variants[0].id
      }
    }
    if (params.has('p2')) {
      const pc = decodeParallelism(params.get('p2')!)
      out.decodeParallelismOverride = pc ?? null
    }
  }

  return out
}

function encodeParallelism(p: ParallelismConfig): string {
  return p.parallelism
    .map(id => `${id}${p.parallelismDegrees[id] ?? ''}`)
    .join('.')
}

function decodeParallelism(s: string): ParallelismConfig | null {
  const parts = s.split('.').filter(Boolean)
  if (parts.length === 0) return null
  const ids: ParallelismId[] = []
  const degrees: Partial<Record<ParallelismId, number>> = {}
  for (const part of parts) {
    const m = part.match(/^([a-z]{2})(\d+)$/)
    if (!m) return null
    const id = m[1]
    if (!isParallelismId(id)) return null
    const deg = parseInt(m[2], 10)
    if (!Number.isFinite(deg) || deg < 1) return null
    ids.push(id)
    degrees[id] = deg
  }
  return { parallelism: ids, parallelismDegrees: degrees }
}

function readStoreState(): ShareableState {
  return {
    acceleratorId: get(acceleratorId),
    variantId: get(variantId),
    systemId: get(systemId),
    modelId: get(modelId),
    quant: get(quant),
    workload: get(workload),
    parallelismOverride: get(parallelismOverride),
    disaggKvTransferFabricId: get(disaggKvTransferFabricId),
    disaggFirstTokenOnPrefill: get(disaggFirstTokenOnPrefill),
    heterogeneous: get(heterogeneous),
    prefillAcceleratorId: get(prefillAcceleratorId),
    prefillVariantId: get(prefillVariantId),
    prefillSystemId: get(prefillSystemId),
    prefillParallelismOverride: get(prefillParallelismOverride),
    decodeAcceleratorId: get(decodeAcceleratorId),
    decodeVariantId: get(decodeVariantId),
    decodeSystemId: get(decodeSystemId),
    decodeParallelismOverride: get(decodeParallelismOverride),
    concurrencyOverride: get(concurrencyOverride),
  }
}

function applyToStores(partial: Partial<ShareableState>): void {
  // Order matters: set systemId AFTER acceleratorId/variantId so that, if the
  // URL specifies both (system path pre-seeds accel+variant), the systemId
  // write doesn't get clobbered by a later accel write.
  if (partial.acceleratorId !== undefined) acceleratorId.set(partial.acceleratorId)
  if (partial.variantId !== undefined) variantId.set(partial.variantId)
  if (partial.systemId !== undefined) systemId.set(partial.systemId)
  if (partial.modelId !== undefined) modelId.set(partial.modelId)
  if (partial.quant !== undefined) {
    quant.set(partial.quant)
  } else if (partial.modelId !== undefined) {
    // URL specified a model without explicit quant: seed weights+activations
    // from the model's native precision so sharing `?m=X` lands on X's
    // defaults rather than whatever the recipient last had loaded.
    const m = MODELS.find(x => x.id === partial.modelId)
    if (m) quant.update(q => ({ ...q, weights: m.nativeDtype, activations: defaultActivationsFor(m.nativeDtype) }))
  }
  if (partial.workload !== undefined) workload.set(partial.workload)
  if (partial.concurrencyOverride !== undefined) concurrencyOverride.set(partial.concurrencyOverride)
  if (partial.parallelismOverride !== undefined) parallelismOverride.set(partial.parallelismOverride)
  if (partial.disaggKvTransferFabricId !== undefined) disaggKvTransferFabricId.set(partial.disaggKvTransferFabricId)
  if (partial.disaggFirstTokenOnPrefill !== undefined) disaggFirstTokenOnPrefill.set(partial.disaggFirstTokenOnPrefill)

  // Heterogeneous fields.
  if (partial.heterogeneous !== undefined) heterogeneous.set(partial.heterogeneous)
  if (partial.prefillAcceleratorId !== undefined) prefillAcceleratorId.set(partial.prefillAcceleratorId)
  if (partial.prefillVariantId !== undefined) prefillVariantId.set(partial.prefillVariantId)
  if (partial.prefillSystemId !== undefined) prefillSystemId.set(partial.prefillSystemId)
  if (partial.prefillParallelismOverride !== undefined) prefillParallelismOverride.set(partial.prefillParallelismOverride)
  if (partial.decodeAcceleratorId !== undefined) decodeAcceleratorId.set(partial.decodeAcceleratorId)
  if (partial.decodeVariantId !== undefined) decodeVariantId.set(partial.decodeVariantId)
  if (partial.decodeSystemId !== undefined) decodeSystemId.set(partial.decodeSystemId)
  if (partial.decodeParallelismOverride !== undefined) decodeParallelismOverride.set(partial.decodeParallelismOverride)

  // Invariant: when het=on, both cluster overrides must be non-empty —
  // otherwise the disagg block reactively follows the shared (monolithic)
  // stores and the user can no longer change them independently. Old URLs
  // with het=1 but missing a1/v1 land here (decode side only was emitted
  // pre-decoupling). Seed from shared so subsequent edits decouple cleanly.
  if (get(heterogeneous)) {
    if (!get(prefillAcceleratorId) && !get(prefillSystemId)) {
      prefillAcceleratorId.set(get(acceleratorId))
      prefillVariantId.set(get(variantId))
      prefillSystemId.set(get(systemId))
      if (get(prefillParallelismOverride) === null) {
        prefillParallelismOverride.set(get(parallelismOverride))
      }
    }
    if (!get(decodeAcceleratorId) && !get(decodeSystemId)) {
      decodeAcceleratorId.set(get(acceleratorId))
      decodeVariantId.set(get(variantId))
      decodeSystemId.set(get(systemId))
      if (get(decodeParallelismOverride) === null) {
        decodeParallelismOverride.set(get(parallelismOverride))
      }
    }
  }
}

// Extract the per-tab payload from a raw location.hash. Supports the current
// `#calc?<payload>` / `#sim?<payload>` forms and the legacy bare `#<payload>`
// form (treated as calc-tab payload for backwards compatibility with old
// shared links). Info routes carry no payload regardless of tab argument.
export function tabPayloadFromHash(hash: string, tab: 'calc' | 'sim'): string {
  const h = hash.replace(/^#/, '')
  if (h === '' || h === tab) return ''
  if (h.startsWith(`${tab}?`)) return h.slice(tab.length + 1)
  if (h.startsWith('calc') || h.startsWith('sim') || h.startsWith('info')) return ''
  // Legacy bare payload: only honor it for the calc tab.
  return tab === 'calc' ? h : ''
}

// Read `window.location.hash` and apply any encoded state to the stores.
// Call once at startup, before mounting the app.
export function readUrlIntoStores(): void {
  if (typeof window === 'undefined') return
  // Try both tab prefixes — share URLs can be either #calc?... or #sim?...
  // and the recipient just lands on the corresponding tab. The payload itself
  // is identical (shared state), so either tab can decode the other's URL.
  const payload =
    tabPayloadFromHash(window.location.hash, 'calc') ||
    tabPayloadFromHash(window.location.hash, 'sim')
  if (!payload) return
  applyToStores(decodeState(payload))
}

// Subscribe to the input stores and mirror state back to the URL hash on any
// change. Returns an unsubscribe handle for cleanup (the app never unmounts in
// production, but the handle keeps tests / HMR honest).
export function startUrlSync(): () => void {
  if (typeof window === 'undefined') return () => {}

  let ready = false
  const write = () => {
    if (!ready) return
    const tab = parseRoute(window.location.hash).tab
    // Info tab carries no calc payload; never overwrite it.
    if (tab !== 'calc' && tab !== 'sim') return
    const encoded = encodeState(readStoreState())
    const next = `${window.location.pathname}${window.location.search}#${tab}?${encoded}`
    window.history.replaceState(window.history.state, '', next)
  }

  // Each .subscribe fires synchronously with the current value, so registering
  // the writer would emit one URL write per store before we're done. Hold off
  // until all subscriptions are wired, then flip ready and write once.
  const unsubs = [
    acceleratorId.subscribe(write),
    variantId.subscribe(write),
    systemId.subscribe(write),
    modelId.subscribe(write),
    parallelismOverride.subscribe(write),
    disaggKvTransferFabricId.subscribe(write),
    disaggFirstTokenOnPrefill.subscribe(write),
    quant.subscribe(write),
    workload.subscribe(write),
    concurrencyOverride.subscribe(write),
    heterogeneous.subscribe(write),
    prefillAcceleratorId.subscribe(write),
    prefillVariantId.subscribe(write),
    prefillSystemId.subscribe(write),
    prefillParallelismOverride.subscribe(write),
    decodeAcceleratorId.subscribe(write),
    decodeVariantId.subscribe(write),
    decodeSystemId.subscribe(write),
    decodeParallelismOverride.subscribe(write),
  ]
  ready = true
  write()

  return () => unsubs.forEach(u => u())
}

// Build a shareable absolute URL for the current store state. Used by the
// "Copy link" button. Hash prefix follows the current tab (calc/sim); the
// info tab falls back to calc since info has no shareable payload anyway.
export function buildShareUrl(): string {
  const encoded = encodeState(readStoreState())
  if (typeof window === 'undefined') return `#calc?${encoded}`
  const tab = parseRoute(window.location.hash).tab
  const prefix = tab === 'sim' ? 'sim' : 'calc'
  const { origin, pathname, search } = window.location
  return `${origin}${pathname}${search}#${prefix}?${encoded}`
}

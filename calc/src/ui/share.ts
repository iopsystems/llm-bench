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
  quant, workload
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
  p.set('c', String(state.workload.concurrency))
  if (state.parallelismOverride) {
    p.set('p', encodeParallelism(state.parallelismOverride))
  }
  if (state.systemId && state.disaggKvTransferFabricId) {
    p.set('dk', state.disaggKvTransferFabricId)
    // `df=1` is the default — only emit when the user opted into the
    // worst-case sequential handoff.
    if (!state.disaggFirstTokenOnPrefill) p.set('df', '0')
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
  const c = params.get('c')
  if (pt !== null || ot !== null || c !== null) {
    const wl: Partial<Workload> = {}
    if (pt !== null) {
      const n = parseInt(pt, 10)
      if (Number.isFinite(n) && n > 0) wl.promptTokens = n
    }
    if (ot !== null) {
      const n = parseInt(ot, 10)
      if (Number.isFinite(n) && n > 0) wl.outputTokens = n
    }
    if (c !== null) {
      const n = parseInt(c, 10)
      if (Number.isFinite(n) && n > 0) wl.concurrency = n
    }
    if (Object.keys(wl).length > 0) out.workload = wl as Workload
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
  if (partial.quant !== undefined) quant.set(partial.quant)
  if (partial.workload !== undefined) workload.set(partial.workload)
  if (partial.parallelismOverride !== undefined) parallelismOverride.set(partial.parallelismOverride)
  if (partial.disaggKvTransferFabricId !== undefined) disaggKvTransferFabricId.set(partial.disaggKvTransferFabricId)
  if (partial.disaggFirstTokenOnPrefill !== undefined) disaggFirstTokenOnPrefill.set(partial.disaggFirstTokenOnPrefill)
}

// Extract the calculator payload from a raw location.hash. Supports the
// current `#calc?<payload>` form and the legacy bare `#<payload>` form so
// old shared links keep working. Info routes carry no payload.
export function calcPayloadFromHash(hash: string): string {
  const h = hash.replace(/^#/, '')
  if (h === '' || h === 'calc') return ''
  if (h.startsWith('calc?')) return h.slice('calc?'.length)
  if (h.startsWith('info')) return ''
  return h // legacy: bare payload directly after '#'
}

// Read `window.location.hash` and apply any encoded state to the stores.
// Call once at startup, before mounting the app.
export function readUrlIntoStores(): void {
  if (typeof window === 'undefined') return
  const payload = calcPayloadFromHash(window.location.hash)
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
    // Don't clobber info deep-links with calc hash — only sync when on calc.
    if (parseRoute(window.location.hash).tab !== 'calc') return
    const encoded = encodeState(readStoreState())
    const next = `${window.location.pathname}${window.location.search}#calc?${encoded}`
    // replaceState keeps the back button uncluttered; the URL still updates.
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
  ]
  ready = true
  write()

  return () => unsubs.forEach(u => u())
}

// Build a shareable absolute URL for the current store state. Used by the
// "Copy link" button.
export function buildShareUrl(): string {
  const encoded = encodeState(readStoreState())
  if (typeof window === 'undefined') return `#calc?${encoded}`
  const { origin, pathname, search } = window.location
  return `${origin}${pathname}${search}#calc?${encoded}`
}

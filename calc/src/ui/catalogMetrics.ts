// Derived static metrics for spec sheets. Pure; reuses engine helpers so the
// numbers can't drift from the calculator. fp16 is the fixed KV reference.
import type {
  ModelArch, AcceleratorSpec, MultiAcceleratorSystem,
} from '../engine/types'
import { kvBytesPerTokenPerLayer, activeParams } from '../engine/memory'
import { SOURCES } from '../data/sources'

const KV_REF_DTYPE = 'fp16' as const

function attentionLabel(m: ModelArch): string {
  switch (m.attention.type) {
    case 'full':
      return m.numKvHeads < m.numHeads ? 'Grouped-query attention (GQA)' : 'Full multi-head attention'
    case 'sliding': return `Sliding-window attention (window ${m.attention.window})`
    case 'hybrid': return 'Hybrid sliding/global attention'
    case 'mla': return 'Multi-head latent attention (MLA)'
    case 'mla-dsa': return 'MLA + decoupled sparse attention'
    case 'linear-mla-hybrid': return 'Linear-attention / MLA hybrid'
    case 'csa-hca-hybrid': return 'Compressed sparse + heavily-compressed attention'
    case 'delta-hybrid': return 'Gated DeltaNet + gated attention hybrid'
    case 'mamba2-hybrid': return 'Mamba2 SSM + attention hybrid'
    case 'partial': return `Partial attention (${m.attention.numFullLayers}/${m.layers} blocks, NAS-pruned)`
    default: {
      const _exhaustive: never = m.attention
      return (_exhaustive as { type: string }).type
    }
  }
}

export interface ModelMetrics {
  kvBytesPerTokenPerLayer: number
  kvBytesPerToken: number
  gqaRatio: number
  attentionLabel: string
  moeActiveRatio?: number
}

export function modelMetrics(m: ModelArch): ModelMetrics {
  const perLayer = kvBytesPerTokenPerLayer(m, KV_REF_DTYPE)
  const out: ModelMetrics = {
    kvBytesPerTokenPerLayer: perLayer,
    kvBytesPerToken: perLayer * m.layers,
    gqaRatio: m.numHeads / m.numKvHeads,
    attentionLabel: attentionLabel(m),
  }
  if (m.architecture.type === 'moe') {
    out.moeActiveRatio = activeParams(m) / m.paramCount
  }
  return out
}

export interface OperatingPointMetrics {
  id: string
  label: string
  ridgeByDtype: Partial<Record<string, number>>
  asOf?: string
  notes?: string
  // Resolved human titles, deduped, from tflopsSources+bandwidthSources via SOURCES.
  // Omitted when empty.
  sources?: string[]
}

export interface VariantMetrics {
  id: string
  label: string
  hbmCapacityGB: number
  operatingPoints: OperatingPointMetrics[]
  // achievable ÷ peak TFLOPS per dtype, only for dtypes present in BOTH the
  // 'peak' and 'achievable' operating points. Omitted if either op is absent.
  efficiencyByDtype?: Partial<Record<string, number>>
}

// One flat row per (variant × peak dtype). Variants share an ISA but differ
// in clock/power and HBM, so TFLOPS and ridge are reported per variant rather
// than once for the chip.
export interface PeakRow {
  variantId: string
  variantLabel: string
  hbmCapacityGB: number
  dtype: string
  tflops: number
  ridge: number   // peak FLOP / byte = tflops·1e12 / (hbmBW·1e9)
}

// dtype hardware-support classification, derived from the union of dtypes the
// chip's peak operating points accelerate (ISA-level — independent of the
// memory variant). `via` is the upconvert target for the conversion class.
export interface DtypeSupportRow {
  dtype: string
  support: 'native' | 'conversion' | 'software'
  via?: string
  note: string
}

const DTYPE_WIDTH: Record<string, number> = {
  fp32: 32, fp16: 16, bf16: 16, fp8: 8, int8: 8, fp4: 4, int4: 4,
}
const DTYPE_FAMILY: Record<string, 'float' | 'int'> = {
  fp32: 'float', fp16: 'float', bf16: 'float', fp8: 'float', fp4: 'float',
  int8: 'int', int4: 'int',
}
const DTYPE_LIST = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4']

function classifyDtypes(supported: Set<string>): DtypeSupportRow[] {
  return DTYPE_LIST.map(dt => {
    if (supported.has(dt)) {
      return { dtype: dt, support: 'native', note: 'Hardware-native — full rate' }
    }
    const w = DTYPE_WIDTH[dt]
    // Candidates that can absorb dt by upconvert: supported, width >= dt width.
    const wider = [...supported]
      .filter(s => DTYPE_WIDTH[s] >= w)
      .sort((a, b) => {
        if (DTYPE_WIDTH[a] !== DTYPE_WIDTH[b]) return DTYPE_WIDTH[a] - DTYPE_WIDTH[b]
        const af = DTYPE_FAMILY[a] === DTYPE_FAMILY[dt] ? 0 : 1
        const bf = DTYPE_FAMILY[b] === DTYPE_FAMILY[dt] ? 0 : 1
        if (af !== bf) return af - bf
        return DTYPE_LIST.indexOf(a) - DTYPE_LIST.indexOf(b)
      })
    if (wider.length === 0) {
      return {
        dtype: dt, support: 'software',
        note: 'No hardware path — software-emulated, impractical for serving',
      }
    }
    const via = wider[0]
    return {
      dtype: dt, support: 'conversion', via,
      note: `Upconvert to ${via} — compute at ${via} rate, ${dt} memory footprint`,
    }
  })
}

export type SkuMetrics =
  | {
      kind: 'accelerator'
      variants: VariantMetrics[]
      peakTable: PeakRow[]
      dtypeSupport: DtypeSupportRow[]
    }
  | {
      kind: 'system'
      totalHbmGB: number
      fabricBidirectionalTBs: number
      acceleratorCount: number
    }

function isSystem(s: AcceleratorSpec | MultiAcceleratorSystem): s is MultiAcceleratorSystem {
  return 'aggregate' in s
}

export function skuMetrics(s: AcceleratorSpec | MultiAcceleratorSystem): SkuMetrics {
  if (isSystem(s)) {
    return {
      kind: 'system',
      totalHbmGB: s.aggregate.totalHbmGB,
      fabricBidirectionalTBs: s.aggregate.fabricBidirectionalTBs,
      acceleratorCount: s.accelerator.count,
    }
  }
  const peakTable: PeakRow[] = []
  for (const v of s.variants) {
    const peak = v.operatingPoints.find(o => o.id === 'peak')
    if (!peak) continue
    for (const [dt, tf] of Object.entries(peak.tflops)) {
      if (tf === undefined) continue
      peakTable.push({
        variantId: v.id,
        variantLabel: v.label,
        hbmCapacityGB: v.hbmCapacityGB,
        dtype: dt,
        tflops: tf,
        ridge: (tf * 1e12) / (peak.hbmBandwidthGBs * 1e9),
      })
    }
  }

  const supportedDtypes = new Set<string>()
  for (const v of s.variants) {
    const peak = v.operatingPoints.find(o => o.id === 'peak')
    if (!peak) continue
    for (const [dt, tf] of Object.entries(peak.tflops)) {
      if (tf !== undefined) supportedDtypes.add(dt)
    }
  }

  return {
    kind: 'accelerator',
    peakTable,
    dtypeSupport: classifyDtypes(supportedDtypes),
    variants: s.variants.map(v => {
      const opMetrics: OperatingPointMetrics[] = v.operatingPoints.map(op => {
        const ridgeByDtype: Partial<Record<string, number>> = {}
        for (const [dt, tf] of Object.entries(op.tflops)) {
          if (tf !== undefined) {
            ridgeByDtype[dt] = (tf * 1e12) / (op.hbmBandwidthGBs * 1e9)
          }
        }
        const sourceKeys = [...(op.tflopsSources ?? []), ...(op.bandwidthSources ?? [])]
        const seen = new Set<string>()
        const titles: string[] = []
        for (const k of sourceKeys) {
          const title = SOURCES[k as keyof typeof SOURCES]?.title ?? k
          if (!seen.has(title)) { seen.add(title); titles.push(title) }
        }
        const m: OperatingPointMetrics = { id: op.id, label: op.label, ridgeByDtype }
        if (op.asOf) m.asOf = op.asOf
        if (op.notes) m.notes = op.notes
        if (titles.length > 0) m.sources = titles
        return m
      })

      const peakOp = v.operatingPoints.find(o => o.id === 'peak')
      const achOp = v.operatingPoints.find(o => o.id === 'achievable')
      let efficiencyByDtype: Partial<Record<string, number>> | undefined
      if (peakOp && achOp) {
        efficiencyByDtype = {}
        for (const dt of Object.keys(peakOp.tflops)) {
          const p = peakOp.tflops[dt as keyof typeof peakOp.tflops]
          const a = achOp.tflops[dt as keyof typeof achOp.tflops]
          if (p !== undefined && a !== undefined) {
            efficiencyByDtype[dt] = a / p
          }
        }
      }

      const variant: VariantMetrics = {
        id: v.id, label: v.label, hbmCapacityGB: v.hbmCapacityGB,
        operatingPoints: opMetrics,
      }
      if (efficiencyByDtype) variant.efficiencyByDtype = efficiencyByDtype
      return variant
    }),
  }
}

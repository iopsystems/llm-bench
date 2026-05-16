// Derived static metrics for spec sheets. Pure; reuses engine helpers so the
// numbers can't drift from the calculator. fp16 is the fixed KV reference.
import type {
  ModelArch, AcceleratorSpec, MultiAcceleratorSystem,
} from '../engine/types'
import { kvBytesPerTokenPerLayer, activeParams } from '../engine/memory'

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
}

export interface VariantMetrics {
  id: string
  label: string
  hbmCapacityGB: number
  operatingPoints: OperatingPointMetrics[]
}

export type SkuMetrics =
  | { kind: 'accelerator'; variants: VariantMetrics[] }
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
  return {
    kind: 'accelerator',
    variants: s.variants.map(v => ({
      id: v.id,
      label: v.label,
      hbmCapacityGB: v.hbmCapacityGB,
      operatingPoints: v.operatingPoints.map(op => {
        const ridgeByDtype: Partial<Record<string, number>> = {}
        for (const [dt, tf] of Object.entries(op.tflops)) {
          if (tf !== undefined) {
            ridgeByDtype[dt] = (tf * 1e12) / (op.hbmBandwidthGBs * 1e9)
          }
        }
        return { id: op.id, label: op.label, ridgeByDtype }
      }),
    })),
  }
}

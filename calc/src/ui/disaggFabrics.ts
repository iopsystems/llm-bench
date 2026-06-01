// Eligibility + labeling helpers for the PD-disagg fabric picker.
// Used by both the calc tab's InputPanel and the sim tab's DisaggInputPanel.
//
// Scope of "eligible" (per spec 2026-05-31-single-request-disagg-design.md):
//   - Scale-up fabrics: only those whose `compatibleAcceleratorIds` includes
//     the currently-selected accelerator (NVL72 prefill+decode slices, TPU pods).
//   - Scale-out fabrics: always shown; any GPU can be on IB/EFA/RoCE.
//   - Intra-node and die-to-die fabrics: never shown; too small to host PD-disagg.
//
// Within each group, sort by perGpuBandwidthGBs descending.

import { INTERCONNECTS } from '../data/interconnects'
import type { InterconnectSpec } from '../engine/types'

export interface FabricGroups {
  scaleUp: InterconnectSpec[]
  scaleOut: InterconnectSpec[]
}

export function groupedDisaggFabrics(acceleratorId: string): FabricGroups {
  const byBwDesc = (a: InterconnectSpec, b: InterconnectSpec) =>
    b.perGpuBandwidthGBs - a.perGpuBandwidthGBs
  const scaleUp = INTERCONNECTS
    .filter(i => i.scale === 'scale-up' &&
                 (i.compatibleAcceleratorIds?.includes(acceleratorId) ?? false))
    .sort(byBwDesc)
  const scaleOut = INTERCONNECTS
    .filter(i => i.scale === 'scale-out')
    .sort(byBwDesc)
  return { scaleUp, scaleOut }
}

export function formatFabricLabel(f: InterconnectSpec): string {
  return `${f.name} — ${f.perGpuBandwidthGBs} GB/s/GPU`
}

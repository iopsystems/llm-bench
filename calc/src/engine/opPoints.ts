// Pair operating points across prefill and decode variants for heterogeneous
// PD-disagg. Matches by id ("peak" with "peak", "achievable" with "achievable").
// If a prefill op has no matching id on the decode side, falls back to the
// decode side's first op and synthesizes a composite pair id like
// "prefillId/decodeId" so the UI can disambiguate.

import type { AcceleratorVariant, AcceleratorOperatingPoint } from './types'

export interface OpPointPair {
  prefillOp: AcceleratorOperatingPoint
  decodeOp: AcceleratorOperatingPoint
  id: string
}

export function pairOpPoints(
  prefill: AcceleratorVariant,
  decode: AcceleratorVariant,
): OpPointPair[] {
  return prefill.operatingPoints.map(prefillOp => {
    const matched = decode.operatingPoints.find(o => o.id === prefillOp.id)
    const decodeOp = matched ?? decode.operatingPoints[0]
    const id = matched ? prefillOp.id : `${prefillOp.id}/${decodeOp.id}`
    return { prefillOp, decodeOp, id }
  })
}

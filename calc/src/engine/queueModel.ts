import type { CalcInput } from './types'
import { computeMemory } from './memory'

export interface NMaxResult {
  nMax: number
  boundBy: 'kv' | 'weights'
}

// KV-cap ceiling: how many concurrent in-flight requests can be served before
// HBM exhausts. Decode side (uses decodeMultiDevice/decodeAccelerator when
// present, else falls back to prefill — same as computeMemory). Honest answer
// at the per-rank granularity when multiDevice is configured.
export function computeNMax(input: CalcInput): NMaxResult {
  // Probe at concurrency=1: kvCachePerRequest and activations both scale
  // linearly, so per-request bytes are stable regardless of what the caller
  // passed in for concurrency.
  const probe = { ...input, workload: { ...input.workload, concurrency: 1 } }
  const memory = computeMemory(probe)
  const side = memory.decodeSide

  // perRank present iff multiDevice is configured on the decode side. Use
  // per-rank when available because that's where HBM actually lives.
  const usingPerRank = side.perRank !== undefined
  const capacityBytes = side.hbmCapacityGB * 1024 * 1024 * 1024
  const weightsBytes = usingPerRank ? side.perRank!.weights : side.weights
  const perReqKvBytes = usingPerRank
    ? side.perRank!.kvCachePerRequest
    : memory.kvCachePerRequest
  // decodeActivationsPeak is at concurrency=1 (probed above); perRank.activations
  // already incorporates the per-rank divisor when multiDevice is set.
  const perReqActBytes = usingPerRank
    ? side.perRank!.activations
    : memory.decodeActivationsPeak

  const free = capacityBytes - weightsBytes
  if (free <= 0) return { nMax: 0, boundBy: 'weights' }

  const perReqBytes = perReqKvBytes + perReqActBytes
  if (perReqBytes <= 0) return { nMax: 0, boundBy: 'weights' }

  const nMax = Math.floor(free / perReqBytes)
  return { nMax: Math.max(0, nMax), boundBy: 'kv' }
}

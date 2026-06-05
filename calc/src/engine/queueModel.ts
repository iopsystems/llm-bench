import type { CalcInput } from './types'
import { computeMemory } from './memory'
import { computePrefill } from './prefill'
import { computeDecode } from './decode'
import { INTERCONNECTS } from '../data/interconnects'
import { pairOpPoints } from './opPoints'

export interface NMaxResult {
  nMax: number
  boundBy: 'kv' | 'weights'
}

// KV-cap ceiling: how many concurrent in-flight requests can be served before
// HBM exhausts. Per-rank granularity when multiDevice is configured.
export function computeNMax(input: CalcInput): NMaxResult {
  // Probe at concurrency=1: kvCachePerRequest and activations both scale
  // linearly, so per-request bytes are stable regardless of what the caller
  // passed in for concurrency.
  const probe = { ...input, workload: { ...input.workload, concurrency: 1 } }
  const memory = computeMemory(probe)
  const side = memory.decodeSide

  const capacityBytes  = side.hbmCapacityGB * 1024 * 1024 * 1024
  const weightsBytes   = side.perRank?.weights           ?? side.weights
  const perReqKvBytes  = side.perRank?.kvCachePerRequest ?? memory.kvCachePerRequest
  // decodeActivationsPeak is at concurrency=1 (probed above); perRank.activations
  // already incorporates the per-rank divisor when multiDevice is set.
  const perReqActBytes = side.perRank?.activations       ?? memory.decodeActivationsPeak

  const free = capacityBytes - weightsBytes
  if (free <= 0) return { nMax: 0, boundBy: 'weights' }

  const perReqBytes = perReqKvBytes + perReqActBytes
  if (perReqBytes <= 0) return { nMax: 0, boundBy: 'weights' }

  const nMax = Math.floor(free / perReqBytes)
  return { nMax: Math.max(0, nMax), boundBy: 'kv' }
}

export interface LoadPoint {
  n: number
  tpotS: number
  prefillS: number
  kvTransferS: number
  totalS: number
  throughputTokS: number
  throughputReqS: number
  // < 1 ⇒ decode is the bottleneck (more prefill nodes than decode can fill);
  // > 1 ⇒ prefill is the bottleneck (need more prefill nodes per decode node).
  pdRatio: number
}

// Per-N KPIs computed by reusing the engine's prefill/decode primitives with
// workload.concurrency overridden. Caller passes the disagg-side input
// (concurrency clamped to 1); loadCurve re-introduces N per iteration.
//
// Closed-loop, deterministic, identical-request model: no queue dynamics, no
// percentiles — the math is direct.
export function loadCurve(input: CalcInput, ns: number[]): LoadPoint[] {
  // Resolve op-points the same way calc.ts does, so the chosen perf tier
  // matches what the single-request blocks above show.
  const prefillVariant = input.accelerator.variants.find(v => v.id === input.acceleratorVariantId)
  if (!prefillVariant) return []
  const decodeAccelerator = input.decodeAccelerator ?? input.accelerator
  const decodeVariantId = input.decodeAcceleratorVariantId ?? input.acceleratorVariantId
  const decodeVariant = decodeAccelerator.variants.find(v => v.id === decodeVariantId)
  if (!decodeVariant) return []
  const pairs = pairOpPoints(prefillVariant, decodeVariant)
  if (pairs.length === 0) return []
  const pair = pairs[0]  // v1: use the first (canonical) op-point pair

  // prefillS and kvTransferS are independent of N (per-request, not per-batch).
  const probeMem = computeMemory({ ...input, workload: { ...input.workload, concurrency: 1 } })
  const prefillS = computePrefill(input, pair.prefillOp, probeMem).timeS

  let kvTransferS = 0
  if (input.disaggKvTransferFabricId) {
    const fab = INTERCONNECTS.find(i => i.id === input.disaggKvTransferFabricId)
    if (fab) {
      const bw = fab.perDirectionGBs ?? fab.perGpuBandwidthGBs / 2
      kvTransferS = probeMem.kvCachePerRequest / (bw * 1e9)
    }
  }

  const outputTokens = input.workload.outputTokens

  return ns.map(n => {
    // Memory recomputes at each N because decode-step KV bytes scale with batch.
    const inputN = { ...input, workload: { ...input.workload, concurrency: n } }
    const memN = computeMemory(inputN)
    const tpotS = computeDecode(inputN, pair.decodeOp, memN).timePerTokenS

    // Mirrors calc.ts two-mode latency: overlap hides KV transfer behind first
    // decode token emission; stutter only when transfer outlasts that first token.
    // Sequential (firstTokenOnPrefill=false): no hiding, full serial cost.
    const isOverlap = kvTransferS > 0 && (input.disaggFirstTokenOnPrefill ?? true)
    const stutterS = isOverlap ? Math.max(0, kvTransferS - tpotS) : 0
    const totalS = isOverlap
      ? prefillS + outputTokens * tpotS + stutterS
      : prefillS + kvTransferS + outputTokens * tpotS
    const throughputReqS = Math.min(n / (outputTokens * tpotS), 1 / prefillS)
    const throughputTokS = throughputReqS * outputTokens
    const pdRatio = pdInstanceRatio(prefillS, outputTokens, tpotS, n)

    return { n, tpotS, prefillS, kvTransferS, totalS, throughputTokS, throughputReqS, pdRatio }
  })
}

export function pdInstanceRatio(prefillS: number, outputTokens: number, tpotS: number, n: number): number {
  return (n * prefillS) / (outputTokens * tpotS)
}

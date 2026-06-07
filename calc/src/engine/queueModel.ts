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
// HBM exhausts. The `side` parameter picks the constraining phase:
//   'decode'  — decode-phase memory (kv + tiny decode activations). Correct
//               when the cluster only runs decode (disagg decode cluster).
//   'prefill' — prefill-phase memory (kv + large prefill activations).
//               Correct when the cluster runs both phases monolithically;
//               matches the existing memory-fit check used by the Calc tab.
// Per-rank granularity when multiDevice is configured.
export function computeNMax(input: CalcInput, side: 'prefill' | 'decode' = 'decode'): NMaxResult {
  const probe = { ...input, workload: { ...input.workload, concurrency: 1 } }
  const memory = computeMemory(probe)
  const memSide = side === 'prefill' ? memory.prefillSide : memory.decodeSide

  const capacityBytes  = memSide.hbmCapacityGB * 1024 * 1024 * 1024
  const weightsBytes   = memSide.perRank?.weights           ?? memSide.weights
  const perReqKvBytes  = memSide.perRank?.kvCachePerRequest ?? memory.kvCachePerRequest
  // Activations differ by phase: prefill side uses prompt × hidden activations
  // (large, scales with promptTokens); decode side uses single-token activations
  // (tiny). The Calc tab's memory-fit check uses prefill phase, so nMaxCalc
  // must match. The LoadSection's decode cluster never runs prefill, so it
  // correctly uses decode activations.
  const perReqActBytes = memSide.perRank?.activations
    ?? (side === 'prefill' ? memory.activationsPeak : memory.decodeActivationsPeak)

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
  ttftS: number              // prefill + first-decode-step (overlap) | prefill + kvTransfer (sequential)
  totalS: number
  throughputTokS: number
  throughputReqS: number
  // < 1 ⇒ decode is the bottleneck (more prefill nodes than decode can fill);
  // > 1 ⇒ prefill is the bottleneck (need more prefill nodes per decode node).
  pdRatio: number

  // Per-device throughput surfaced separately for prefill and decode because
  // disagg sizing is fundamentally about P:D node ratios — operators want to
  // see "this prefill SKU does X input tok/s/device, this decode SKU does Y
  // output tok/s/device" and pick the cluster mix from there. Device counts
  // are included so callers can compute aggregate without re-deriving.
  prefillInputTokPerSPerDevice: number   // promptTokens / (prefillS × prefillDevices)
  decodeOutputTokPerSPerDevice: number   // n / (tpotS × decodeDevices)
  prefillDevices: number                 // 1 for single-chip; system.accelerator.count otherwise
  decodeDevices: number                  // falls back to prefillDevices in homogeneous mode

  // Per-request latency in the steady-state closed-loop model. The v1 single-
  // workload assumption means every request sees this same value — no
  // distribution. Real percentile spread requires variance (variable arrivals
  // and/or variable workloads), deferred to v2.
  latencyS: number

  // Aggregate input-side throughput. throughputTokS is the output (decode) rate;
  // inputTokPerS is the input (prompt) rate. Mirrors the monolithic "Input /
  // Output / Req" throughput card breakdown.
  inputTokPerS: number

  // TTFT mode: 'overlap' (first decode step runs on prefill cluster while KV
  // transfers), 'sequential' (KV transfer must complete before decode starts),
  // or 'no-fabric' (single-cluster — kvTransferS=0, TTFT=prefillS).
  ttftMode: 'overlap' | 'sequential' | 'no-fabric'
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

  // First-decode-step latency on the prefill cluster (used for overlap-mode
  // TTFT). Computed at batch=1 — at TTFT time only the just-arrived request
  // is decoding on the prefill cluster. Reuses the existing probeMem (same
  // concurrency=1 probe). Mirrors calc.ts path. N-independent; computed once.
  const probeInput1 = { ...input, workload: { ...input.workload, concurrency: 1 } }
  const decodeOnPrefill1 = computeDecode(probeInput1, pair.prefillOp, probeMem, input.multiDevice)
  const firstStepOnPrefillS = decodeOnPrefill1.timePerTokenS

  const outputTokens = input.workload.outputTokens

  // Disagg convention: input.multiDevice describes the prefill cluster, and
  // input.decodeMultiDevice (when set) describes the decode cluster. In the
  // homogeneous case decodeMultiDevice is absent and we reuse prefill's count.
  const prefillDevices = input.multiDevice?.system.accelerator.count ?? 1
  const decodeDevices = input.decodeMultiDevice?.system.accelerator.count ?? prefillDevices

  return ns.map(n => {
    // Memory recomputes at each N because decode-step KV bytes scale with batch.
    const inputN = { ...input, workload: { ...input.workload, concurrency: n } }
    const memN = computeMemory(inputN)
    const tpotS = computeDecode(inputN, pair.decodeOp, memN).timePerTokenS

    // Mirrors calc.ts two-mode latency: overlap hides KV transfer behind first
    // decode token emission; stutter only when transfer outlasts that first token.
    // Sequential (firstTokenOnPrefill=false): no hiding, full serial cost.
    const isOverlap = kvTransferS > 0 && (input.disaggFirstTokenOnPrefill ?? true)
    const ttftMode: 'overlap' | 'sequential' | 'no-fabric' =
      kvTransferS === 0 ? 'no-fabric'
      : isOverlap       ? 'overlap'
                        : 'sequential'
    const stutterS = isOverlap ? Math.max(0, kvTransferS - tpotS) : 0
    const totalS = isOverlap
      ? prefillS + outputTokens * tpotS + stutterS
      : prefillS + kvTransferS + outputTokens * tpotS
    const ttftS = isOverlap
      ? prefillS + firstStepOnPrefillS
      : prefillS + kvTransferS
    const throughputReqS = Math.min(n / (outputTokens * tpotS), 1 / prefillS)
    const throughputTokS = throughputReqS * outputTokens
    const pdRatio = pdInstanceRatio(prefillS, outputTokens, tpotS, n)

    const prefillInputTokPerSPerDevice = input.workload.promptTokens / (prefillS * prefillDevices)
    const decodeOutputTokPerSPerDevice = n / (tpotS * decodeDevices)
    const latencyS = totalS
    const inputTokPerS = throughputReqS * input.workload.promptTokens

    return {
      n, tpotS, prefillS, kvTransferS, ttftS, totalS,
      throughputTokS, throughputReqS,
      pdRatio,
      prefillInputTokPerSPerDevice, decodeOutputTokPerSPerDevice,
      prefillDevices, decodeDevices,
      latencyS,
      inputTokPerS,
      ttftMode,
    }
  })
}

export function pdInstanceRatio(prefillS: number, outputTokens: number, tpotS: number, n: number): number {
  return (n * prefillS) / (outputTokens * tpotS)
}

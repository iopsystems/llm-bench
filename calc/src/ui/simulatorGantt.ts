// Pure geometry for the SimulatorGantt SVG. Kept separate from the .svelte
// renderer so it's testable in the node-only vitest environment.
//
// Three cases (see spec 2026-05-26-single-request-simulator-design.md §UI):
//   A — non-disagg (kvTransferS = 0): two segments, no overlay.
//   B — disagg + firstTokenOnPrefill=true: two segments + KV-xfer overlay
//       in [prefillS, prefillS + kvTransferS]. The overlay is rendered as
//       a thin bar below the main row to communicate "KV streams in parallel
//       with the prefill cluster's first decode step."
//   C — disagg + firstTokenOnPrefill=false: three serial segments
//       (prefill → kv-xfer → decode).
//
// Segment x-coordinates are in seconds (caller scales to pixels).

export type Regime = 'compute' | 'memory' | 'comms'

export interface GanttInput {
  prefillS: number
  kvTransferS: number       // 0 when integrated (non-disagg)
  tpotS: number             // decode time per token (constant per v1 spec)
  outputTokens: number
  firstTokenOnPrefill: boolean   // ignored when kvTransferS = 0
  ttftS: number             // engine-reported TTFT; placed verbatim as marker x
  prefillRegime: Regime
  decodeRegime: Regime
}

export interface GanttSegment {
  kind: 'prefill' | 'kv-xfer' | 'decode'
  x: number      // seconds from t=0
  width: number  // seconds
  regime: Regime
}

export interface GanttGeometry {
  segments: GanttSegment[]
  kvOverlay?: { x: number; width: number }
  markerX: number
  totalS: number
}

export function computeGanttGeometry(input: GanttInput): GanttGeometry {
  const { prefillS, kvTransferS, tpotS, outputTokens, firstTokenOnPrefill,
          ttftS, prefillRegime, decodeRegime } = input
  // Spec §Behavior contract: Total = TTFT + TPOT × (outputTokens − 1).
  // (Caveat in spec: small undercount in disagg+sequential case; bounded by
  // one TPOT. We use the formula uniformly for v1.)
  const totalS = ttftS + tpotS * (outputTokens - 1)

  // Case A: no disagg.
  if (kvTransferS === 0) {
    return {
      segments: [
        { kind: 'prefill', x: 0,        width: prefillS,           regime: prefillRegime },
        { kind: 'decode',  x: prefillS, width: totalS - prefillS,  regime: decodeRegime },
      ],
      markerX: ttftS,
      totalS,
    }
  }

  // Case C: disagg, sequential handoff.
  if (!firstTokenOnPrefill) {
    return {
      segments: [
        { kind: 'prefill', x: 0,                       width: prefillS,            regime: prefillRegime },
        { kind: 'kv-xfer', x: prefillS,                width: kvTransferS,         regime: 'comms' },
        { kind: 'decode',  x: prefillS + kvTransferS,  width: totalS - prefillS - kvTransferS, regime: decodeRegime },
      ],
      markerX: ttftS,
      totalS,
    }
  }

  // Case B: disagg, overlap. KV streams parallel to the prefill cluster's
  // first decode step — render KV-xfer as an overlay rather than a segment.
  return {
    segments: [
      { kind: 'prefill', x: 0,        width: prefillS,          regime: prefillRegime },
      { kind: 'decode',  x: prefillS, width: totalS - prefillS, regime: decodeRegime },
    ],
    kvOverlay: { x: prefillS, width: kvTransferS },
    markerX: ttftS,
    totalS,
  }
}

// Pure geometry for the SimulatorGantt SVG. Kept separate from the .svelte
// renderer so it's testable in the node-only vitest environment.
//
// Cases (see spec 2026-05-26-single-request-simulator-design.md §UI plus the
// 2026-05-31 disagg spec):
//   A    — non-disagg (kvTransferS = 0): two segments, no overlay.
//   Bfast — disagg + firstTokenOnPrefill=true + kvTransferS ≤ tpotS: KV is
//          fully masked by the prefill cluster's first decode step. Two
//          segments + a hatched KV-xfer overlay below the main row signals
//          "parallel, not blocking."
//   Bslow — disagg + firstTokenOnPrefill=true + kvTransferS > tpotS: KV
//          extends past the first decode step, so the user actually waits.
//          Render as three inline segments (prefill → kv-xfer → decode) so
//          the blocking portion appears on the main timeline. Marker sits
//          inside the kv-xfer segment (token 1 emerges during transfer).
//   C    — disagg + firstTokenOnPrefill=false: three inline segments,
//          purely sequential handoff.
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
  kind: 'prefill' | 'kv-xfer' | 'decode' | 'stutter'
  x: number      // seconds from t=0
  width: number  // seconds
  regime: Regime
}

export interface GanttGeometry {
  segments: GanttSegment[]
  kvOverlay?: { x: number; width: number }
  markerX: number
  totalS: number
  // Only set in the disagg + firstTokenOnPrefill=true case (B). Captures the
  // gap between the prefill cluster emitting token #1 (at TTFT) and the
  // decode cluster being ready to emit token #2 (at prefill + kvTransferS).
  //   stutterS = max(0, kvTransferS - tpotS)
  // Zero when KV transfer finishes within one decode step (no perceptible
  // pause); positive when transfer outlasts the first decode step — the
  // user sees token #1, waits stutterS, then steady cadence from token #2 on.
  stutterS?: number
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

  // Case B: disagg + firstTokenOnPrefill=true. Two sub-cases:
  const stutterS = Math.max(0, kvTransferS - tpotS)

  if (stutterS === 0) {
    // Bfast: KV transfer fully overlaps the prefill cluster's first decode
    // step → render as a hatched overlay below the main row to signal
    // "parallel, doesn't block the timeline."
    return {
      segments: [
        { kind: 'prefill', x: 0,        width: prefillS,          regime: prefillRegime },
        { kind: 'decode',  x: prefillS, width: totalS - prefillS, regime: decodeRegime },
      ],
      kvOverlay: { x: prefillS, width: kvTransferS },
      markerX: ttftS,
      totalS,
      stutterS,
    }
  }

  // Bslow: KV transfer extends past the first decode step. Visualized as
  // four inline segments so the stutter portion stands out as a hatched gap
  // in the decode timeline:
  //   prefill → decode (first token on prefill cluster, width tpotS)
  //   → stutter (KV still transferring, decode cluster waiting; width stutterS)
  //   → decode (decode cluster takes over, width (N-1)·tpotS)
  // The 'stutter' kind tells the renderer to apply hatched fill; the decode
  // segments stay solid in regime color. Marker (TTFT = prefillS + tpotS)
  // lands at the boundary between the first decode step and the stutter.
  const totalSB = totalS + stutterS
  return {
    segments: [
      { kind: 'prefill', x: 0,                       width: prefillS,                          regime: prefillRegime },
      { kind: 'decode',  x: prefillS,                width: tpotS,                             regime: decodeRegime  },
      { kind: 'stutter', x: prefillS + tpotS,        width: stutterS,                          regime: 'comms'       },
      { kind: 'decode',  x: prefillS + kvTransferS,  width: totalSB - prefillS - kvTransferS,  regime: decodeRegime  },
    ],
    markerX: ttftS,
    totalS: totalSB,
    stutterS,
  }
}

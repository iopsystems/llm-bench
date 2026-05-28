<script lang="ts">
  import { computeGanttGeometry, type GanttInput } from './simulatorGantt'
  export let input: GanttInput

  // SVG layout constants. The main row is the prefill/decode timeline;
  // the overlay row (case B only) sits below it for the KV-xfer bar.
  const W = 720
  const ROW_H = 28
  const OVERLAY_H = 10
  const ROW_GAP = 4
  const PADDING = 12  // left/right padding inside the viewBox

  $: geom = computeGanttGeometry(input)
  // Scale: seconds → pixels along the timeline. Reserve PADDING on each side.
  $: pxPerS = (W - 2 * PADDING) / Math.max(geom.totalS, 1e-9)
  $: hasOverlay = geom.kvOverlay !== undefined
  $: totalH = ROW_H + (hasOverlay ? ROW_GAP + OVERLAY_H : 0) + 30  // +30 for axis labels

  // Adaptive anchor so "first token" stays inside the viewBox even when the
  // marker sits near the left edge (common for non-disagg: TTFT = prefill,
  // and prefill is a small fraction of total at typical output lengths).
  // ~32px half-width covers "first token" at 11px bold system-ui.
  const LABEL_HALF_W = 32
  $: markerPx = PADDING + geom.markerX * pxPerS
  $: markerLabelAnchor =
    markerPx < PADDING + LABEL_HALF_W ? 'start'
    : markerPx > W - PADDING - LABEL_HALF_W ? 'end'
    : 'middle'
  $: markerLabelX =
    markerLabelAnchor === 'start' ? PADDING
    : markerLabelAnchor === 'end' ? W - PADDING
    : markerPx

  function ms(s: number): string {
    if (s >= 1)    return `${(s).toFixed(2)} s`
    if (s >= 1e-3) return `${(s * 1e3).toFixed(0)} ms`
    return `${(s * 1e6).toFixed(0)} µs`
  }

  // Map a regime to a CSS class on the rect.
  const regimeClass = (r: 'compute' | 'memory' | 'comms') => `regime-${r}`
</script>

<svg viewBox="0 0 {W} {totalH}" class="gantt" role="img" aria-label="Single-request timeline">
  <!-- Main row: prefill + decode (and kv-xfer in case C). -->
  {#each geom.segments as seg}
    <rect
      class="seg {regimeClass(seg.regime)}"
      x={PADDING + seg.x * pxPerS}
      y={0}
      width={Math.max(seg.width * pxPerS, 1)}
      height={ROW_H}
    >
      <title>{seg.kind} · {ms(seg.width)} · {seg.regime}-bound</title>
    </rect>
  {/each}

  <!-- KV-xfer overlay (case B only). -->
  {#if geom.kvOverlay}
    <rect
      class="seg regime-comms overlay"
      x={PADDING + geom.kvOverlay.x * pxPerS}
      y={ROW_H + ROW_GAP}
      width={Math.max(geom.kvOverlay.width * pxPerS, 1)}
      height={OVERLAY_H}
    >
      <title>kv-xfer · {ms(geom.kvOverlay.width)} · overlapped with first decode step</title>
    </rect>
  {/if}

  <!-- TTFT marker. -->
  <line
    class="marker"
    x1={PADDING + geom.markerX * pxPerS}
    y1={0}
    x2={PADDING + geom.markerX * pxPerS}
    y2={ROW_H + (hasOverlay ? ROW_GAP + OVERLAY_H : 0)}
  />
  <text
    class="marker-label"
    x={markerLabelX}
    y={ROW_H + (hasOverlay ? ROW_GAP + OVERLAY_H : 0) + 14}
    text-anchor={markerLabelAnchor}
  >first token</text>

  <!-- Axis ticks: 0 and Total. (TTFT is shown by the marker line and label above.) -->
  <text class="tick" x={PADDING}                                    y={totalH - 4} text-anchor="start">0</text>
  <text class="tick" x={PADDING + geom.totalS * pxPerS}             y={totalH - 4} text-anchor="end">{ms(geom.totalS)}</text>
</svg>

<style>
  .gantt { width: 100%; height: auto; display: block; }
  .seg { stroke: #fff; stroke-width: 1; }
  /* Regime palette matches the Calculator tab: compute=warm/orange,
     memory=cool/blue. */
  .seg.regime-compute { fill: #c05621; }
  .seg.regime-memory  { fill: #2b6cb0; }
  .seg.regime-comms   { fill: #6b46c1; }
  .seg.overlay { stroke-dasharray: 3 2; opacity: 0.85; }
  .marker { stroke: #111; stroke-width: 1.5; stroke-dasharray: 2 2; }
  .marker-label { font: 600 11px system-ui; fill: #111; }
  .tick { font: 11px system-ui; fill: #555; }
</style>

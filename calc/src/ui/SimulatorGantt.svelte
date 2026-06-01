<script lang="ts">
  import { computeGanttGeometry, type GanttInput } from './simulatorGantt'
  export let input: GanttInput

  // Main row hosts prefill/decode; case-Bfast adds an overlay row below.
  const W = 720
  const ROW_H = 28
  const OVERLAY_H = 10
  const ROW_GAP = 4
  const PADDING = 12

  $: geom = computeGanttGeometry(input)
  $: pxPerS = (W - 2 * PADDING) / Math.max(geom.totalS, 1e-9)
  $: hasOverlay = geom.kvOverlay !== undefined
  $: totalH = ROW_H + (hasOverlay ? ROW_GAP + OVERLAY_H : 0) + 30

  function ms(s: number): string {
    if (s >= 1)    return `${(s).toFixed(2)} s`
    if (s >= 1e-3) return `${(s * 1e3).toFixed(0)} ms`
    return `${(s * 1e6).toFixed(0)} µs`
  }

  // case A/C → "first token"; case B → "(no stutter)" or "(stutter: …)"
  $: markerLabel = geom.stutterS === undefined
    ? 'first token'
    : geom.stutterS === 0
      ? 'first token (no stutter)'
      : `first token (stutter: ${ms(geom.stutterS)})`

  // Anchor label inside the viewBox when the marker hugs an edge.
  $: labelHalfW = geom.stutterS !== undefined ? 80 : 32
  $: markerPx = PADDING + geom.markerX * pxPerS
  $: markerLabelAnchor =
    markerPx < PADDING + labelHalfW ? 'start'
    : markerPx > W - PADDING - labelHalfW ? 'end'
    : 'middle'
  $: markerLabelX =
    markerLabelAnchor === 'start' ? PADDING
    : markerLabelAnchor === 'end' ? W - PADDING
    : markerPx

  const regimeClass = (r: 'compute' | 'memory' | 'comms') => `regime-${r}`
</script>

<svg viewBox="0 0 {W} {totalH}" class="gantt" role="img" aria-label="Single-request timeline">
  <defs>
    <pattern id="commsHatch" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
      <rect width="5" height="5" fill="#fff" />
      <line x1="0" y1="0" x2="0" y2="5" stroke="#6b46c1" stroke-width="2" />
    </pattern>
  </defs>

  {#each geom.segments as seg}
    <rect
      class="seg {regimeClass(seg.regime)} {seg.kind === 'stutter' ? 'hatched' : ''}"
      x={PADDING + seg.x * pxPerS}
      y={0}
      width={Math.max(seg.width * pxPerS, 1)}
      height={ROW_H}
    >
      <title>{seg.kind} · {ms(seg.width)} · {seg.regime}-bound</title>
    </rect>
    {#if seg.kind === 'stutter'}
      <!-- Inset 1px so the stroke matches neighbors' visible band. -->
      <rect
        class="hatched-border"
        x={PADDING + seg.x * pxPerS + 1}
        y={1}
        width={Math.max(seg.width * pxPerS - 2, 1)}
        height={ROW_H - 2}
      />
    {/if}
  {/each}

  {#if geom.kvOverlay}
    <rect
      class="seg regime-comms hatched"
      x={PADDING + geom.kvOverlay.x * pxPerS}
      y={ROW_H + ROW_GAP}
      width={Math.max(geom.kvOverlay.width * pxPerS, 1)}
      height={OVERLAY_H}
    >
      <title>kv-xfer · {ms(geom.kvOverlay.width)} · overlapped with first decode step</title>
    </rect>
  {/if}

  <!-- TTFT marker — main row only, regardless of overlay. -->
  <line
    class="marker"
    x1={PADDING + geom.markerX * pxPerS}
    y1={0}
    x2={PADDING + geom.markerX * pxPerS}
    y2={ROW_H}
  />
  <text
    class="marker-label"
    x={markerLabelX}
    y={ROW_H + (hasOverlay ? ROW_GAP + OVERLAY_H : 0) + 14}
    text-anchor={markerLabelAnchor}
  >{markerLabel}</text>

  <text class="tick" x={PADDING}                        y={totalH - 4} text-anchor="start">0</text>
  <text class="tick" x={PADDING + geom.totalS * pxPerS} y={totalH - 4} text-anchor="end">{ms(geom.totalS)}</text>
</svg>

<style>
  .gantt { width: 100%; height: auto; display: block; }
  .seg { stroke: #fff; stroke-width: 1; }
  /* Compute=orange, memory=blue — matches Calculator tab. */
  .seg.regime-compute { fill: #c05621; }
  .seg.regime-memory  { fill: #2b6cb0; }
  .seg.regime-comms   { fill: #6b46c1; }
  /* Hatched purple for parallel KV-xfer (case-Bfast overlay, case-Bslow inline
     stutter). Inherits .seg's white stroke so visible bounds match neighbors. */
  .seg.hatched { fill: url(#commsHatch); }
  .hatched-border { fill: none; stroke: #6b46c1; stroke-width: 1; pointer-events: none; }
  /* Pale yellow stands out against all three regime colors. */
  .marker { stroke: #fcd34d; stroke-width: 2.5; }
  .marker-label { font: 600 11px system-ui; fill: #111; }
  .tick { font: 11px system-ui; fill: #555; }
</style>

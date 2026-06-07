<script lang="ts">
  import type { LoadPoint } from '../engine/queueModel'

  export let points: LoadPoint[]
  // The selected operating point — parent owns the selection logic (incl. the
  // nearest-neighbor fallback when nMax > 256 strides the ns sweep), so we
  // just render whatever it hands us. Used to be `selectedN: number` and the
  // chart did its own find/fallback, but that fallback was "rightmost point"
  // which snapped the marker to N=nMax whenever the find failed.
  export let selectedPoint: LoadPoint | null
  export let nMax: number

  // Single SVG with dual y-axes: throughput (left, blue) + latency (right, orange).
  // Wider than the old split panels to fill the 2/3 chart-col.
  const W = 520
  const H = 200
  const ML = 44   // left margin — throughput axis labels
  const MR = 44   // right margin — latency axis labels
  const MT = 12
  const MB = 24
  const PW = W - ML - MR
  const PH = H - MT - MB

  // Snap to {1,2,5,10}×10^exp so axis labels land on round numbers.
  function niceMax(values: number[]): number {
    const max = Math.max(...values, 0)
    if (max === 0) return 1
    const exp = Math.floor(Math.log10(max))
    const base = Math.pow(10, exp)
    const norm = max / base
    const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10
    return nice * base
  }

  $: throughputMax = niceMax(points.map(p => p.throughputTokS))
  $: latencyMax    = niceMax(points.map(p => p.latencyS))

  function xPx(n: number): number { return ML + (n - 1) / Math.max(1, nMax - 1) * PW }
  function yPxThru(v: number): number { return MT + PH - (v / throughputMax) * PH }
  function yPxLat(v: number):  number { return MT + PH - (v / latencyMax)    * PH }

  $: thruPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${xPx(p.n).toFixed(2)},${yPxThru(p.throughputTokS).toFixed(2)}`
  ).join(' ')
  $: latPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${xPx(p.n).toFixed(2)},${yPxLat(p.latencyS).toFixed(2)}`
  ).join(' ')

  function fmtThru(v: number): string {
    if (v >= 1e6) return `${(v / 1e6).toPrecision(3)}M`
    if (v >= 1e3) return `${(v / 1e3).toPrecision(3)}k`
    return v.toPrecision(3)
  }
  function fmtLat(v: number): string {
    if (v >= 1) return `${v.toPrecision(3)}s`
    return `${(v * 1000).toPrecision(3)}ms`
  }
</script>

<div class="chart">
  <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}>
    <!-- left y-axis (throughput) -->
    <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#bbb" stroke-width="1" />
    <text x={ML - 4} y={MT + 4} text-anchor="end" font-size="9" fill="#2b6cb0">{fmtThru(throughputMax)}</text>
    <text x={ML - 4} y={MT + PH} text-anchor="end" font-size="9" fill="#2b6cb0">0</text>

    <!-- right y-axis (latency) -->
    <line x1={ML + PW} y1={MT} x2={ML + PW} y2={MT + PH} stroke="#bbb" stroke-width="1" />
    <text x={ML + PW + 4} y={MT + 4} text-anchor="start" font-size="9" fill="#c05621">{fmtLat(latencyMax)}</text>
    <text x={ML + PW + 4} y={MT + PH} text-anchor="start" font-size="9" fill="#c05621">0</text>

    <!-- x-axis -->
    <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#bbb" stroke-width="1" />
    <text x={ML} y={H - 6} text-anchor="start" font-size="9" fill="#666">N=1</text>
    <text x={ML + PW} y={H - 6} text-anchor="end" font-size="9" fill="#666">N={nMax}</text>

    <!-- throughput curve (blue) -->
    <path d={thruPath} fill="none" stroke="#2b6cb0" stroke-width="1.5" />

    <!-- latency curve (orange) -->
    <path d={latPath} fill="none" stroke="#c05621" stroke-width="1.5" />

    <!-- vertical marker + dots at selectedN -->
    {#if selectedPoint}
      <line x1={xPx(selectedPoint.n)} y1={MT} x2={xPx(selectedPoint.n)} y2={MT + PH}
            stroke="#888" stroke-width="2.5" />
      <circle cx={xPx(selectedPoint.n)} cy={yPxThru(selectedPoint.throughputTokS)}
              r="3.5" fill="#2b6cb0" />
      <circle cx={xPx(selectedPoint.n)} cy={yPxLat(selectedPoint.latencyS)}
              r="3.5" fill="#c05621" />
    {/if}

    <!-- legend (top-left, inside plot area) -->
    <g transform={`translate(${ML + 8}, ${MT + 4})`}>
      <rect x="0" y="0" width="10" height="3" fill="#2b6cb0" />
      <text x="14" y="3" font-size="9" fill="#555" dominant-baseline="middle">Throughput</text>
      <rect x="80" y="0" width="10" height="3" fill="#c05621" />
      <text x="94" y="3" font-size="9" fill="#555" dominant-baseline="middle">Latency</text>
    </g>
  </svg>
</div>

<style>
  /* No border — parent (LoadSection) owns the wrapper border so the slider
     and chart sit in one container. */
  .chart { width: 100%; }
</style>

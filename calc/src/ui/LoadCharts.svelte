<script lang="ts">
  import type { LoadPoint } from '../engine/queueModel'

  export let points: LoadPoint[]
  export let selectedN: number
  export let nMax: number

  // Chart dimensions: each panel 280×140 px, with 32-px left margin for y-axis
  // labels and 24-px bottom margin for x-axis labels. Inline SVG, no charting
  // dep — keeps bundle small and matches the gantt's style.
  const W = 280
  const H = 140
  const ML = 36
  const MB = 24
  const MT = 8
  const MR = 8
  const PW = W - ML - MR
  const PH = H - MT - MB

  // Derive both y-axis maxima from the data; round up to a "nice" number so
  // tick labels are readable.
  $: throughputMax = niceMax(points.map(p => p.throughputTokS))
  $: latencyMax    = niceMax(points.map(p => p.totalS))

  // Snap to {1,2,5,10}×10^exp — same grid the human eye expects from axes.
  // Without this, a 387 tok/s max produces an axis label like "387" at the
  // top and an implicit scale that makes the line look wrong.
  function niceMax(values: number[]): number {
    const max = Math.max(...values, 0)
    if (max === 0) return 1
    const exp = Math.floor(Math.log10(max))
    const base = Math.pow(10, exp)
    const norm = max / base
    const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10
    return nice * base
  }

  function xPx(n: number): number { return ML + (n - 1) / Math.max(1, nMax - 1) * PW }
  function yPxThru(v: number): number { return MT + PH - (v / throughputMax) * PH }
  function yPxLat(v: number):  number { return MT + PH - (v / latencyMax)    * PH }

  $: thruPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${xPx(p.n).toFixed(2)},${yPxThru(p.throughputTokS).toFixed(2)}`
  ).join(' ')
  $: latPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${xPx(p.n).toFixed(2)},${yPxLat(p.totalS).toFixed(2)}`
  ).join(' ')

  $: selectedPoint = points.find(p => p.n === selectedN) ?? points[points.length - 1]

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

<div class="lc-charts">
  <div class="lc-chart">
    <div class="lc-title">Throughput (tok/s)</div>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
      <!-- y-axis -->
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#bbb" stroke-width="1" />
      <text x={ML - 4} y={MT + 4} text-anchor="end" font-size="9" fill="#666">{fmtThru(throughputMax)}</text>
      <text x={ML - 4} y={MT + PH} text-anchor="end" font-size="9" fill="#666">0</text>
      <!-- x-axis -->
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#bbb" stroke-width="1" />
      <text x={ML} y={H - 6} text-anchor="start" font-size="9" fill="#666">N=1</text>
      <text x={ML + PW} y={H - 6} text-anchor="end" font-size="9" fill="#666">N={nMax}</text>
      <!-- curve -->
      <path d={thruPath} fill="none" stroke="#2b6cb0" stroke-width="1.5" />
      <!-- selected marker -->
      {#if selectedPoint}
        <line x1={xPx(selectedPoint.n)} y1={MT} x2={xPx(selectedPoint.n)} y2={MT + PH}
              stroke="#fcd34d" stroke-width="2.5" />
        <circle cx={xPx(selectedPoint.n)} cy={yPxThru(selectedPoint.throughputTokS)}
                r="3.5" fill="#2b6cb0" />
      {/if}
    </svg>
  </div>

  <div class="lc-chart">
    <div class="lc-title">Per-request latency</div>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#bbb" stroke-width="1" />
      <text x={ML - 4} y={MT + 4} text-anchor="end" font-size="9" fill="#666">{fmtLat(latencyMax)}</text>
      <text x={ML - 4} y={MT + PH} text-anchor="end" font-size="9" fill="#666">0</text>
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#bbb" stroke-width="1" />
      <text x={ML} y={H - 6} text-anchor="start" font-size="9" fill="#666">N=1</text>
      <text x={ML + PW} y={H - 6} text-anchor="end" font-size="9" fill="#666">N={nMax}</text>
      <path d={latPath} fill="none" stroke="#c05621" stroke-width="1.5" />
      {#if selectedPoint}
        <line x1={xPx(selectedPoint.n)} y1={MT} x2={xPx(selectedPoint.n)} y2={MT + PH}
              stroke="#fcd34d" stroke-width="2.5" />
        <circle cx={xPx(selectedPoint.n)} cy={yPxLat(selectedPoint.totalS)}
                r="3.5" fill="#c05621" />
      {/if}
    </svg>
  </div>
</div>

<style>
  .lc-charts {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;
  }
  .lc-chart {
    padding: 0.6rem 0.9rem;
    background: #fff; border: 1px solid #d4d4d4; border-radius: 0.4rem;
  }
  .lc-title {
    font-size: 0.8rem; font-weight: 600; color: #555; margin-bottom: 0.3rem;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  @media (max-width: 800px) {
    .lc-charts { grid-template-columns: 1fr; }
  }
</style>

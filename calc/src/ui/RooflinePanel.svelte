<script lang="ts">
  import { input, result } from './stores'

  // Log-log plot dimensions. Log axes because compute peaks and arithmetic
  // intensities span many orders of magnitude across GPUs and phases.
  const W = 620, H = 380
  const M = { top: 20, right: 24, bottom: 50, left: 70 }
  const plotW = W - M.left - M.right
  const plotH = H - M.top - M.bottom

  // Distinct color per operating point. Order chosen so 'peak' (first) stays
  // bright; subsequent tiers (achievable, etc.) step toward warmer tones.
  const OP_COLORS = ['#3a6df0', '#e07a1f', '#21a87a', '#a64ac9'] as const

  type OpData = {
    id: string; label: string; color: string
    peakFlops: number             // FLOPs/sec at peak compute (for activations dtype)
    peakBw: number                // bytes/sec at peak BW
    ridgeAi: number               // FLOPs/byte at the ridge
    prefill: { ai: number; perf: number; regime: 'compute' | 'memory' }
    decode:  { ai: number; perf: number; regime: 'compute' | 'memory' }
  }

  $: ops = (() => {
    if (!$input || !$result) return [] as OpData[]
    const variant = $input.gpu.variants.find(v => v.id === $input.gpuVariantId)
    if (!variant) return [] as OpData[]
    const out: OpData[] = []
    variant.operatingPoints.forEach((op, idx) => {
      const t = op.tflops[$input!.quant.activations]
      const p = $result!.perf[op.id]
      if (t === undefined || !p) return
      const peakFlops = t * 1e12
      const peakBw = op.hbmBandwidthGBs * 1e9
      out.push({
        id: op.id, label: op.label,
        color: OP_COLORS[idx % OP_COLORS.length],
        peakFlops, peakBw,
        ridgeAi: peakFlops / peakBw,
        prefill: {
          ai: p.prefill.flops / p.prefill.bytes,
          perf: p.prefill.flops / p.prefill.timeS,
          regime: p.prefill.regime
        },
        decode: {
          ai: p.decode.flopsPerStep / p.decode.bytesPerStep,
          perf: p.decode.flopsPerStep / p.decode.timePerTokenS,
          regime: p.decode.regime
        }
      })
    })
    return out
  })()

  // Domain: pad the spread of observed values by ~half a decade in each direction
  // so the markers don't sit on the plot edge. Clamp x to >=0.1 since AI below
  // that is theoretical noise (loads more bytes than FLOPs being done).
  $: domain = (() => {
    if (ops.length === 0) return null
    const ais = ops.flatMap(o => [o.prefill.ai, o.decode.ai, o.ridgeAi])
    const perfs = ops.flatMap(o => [o.prefill.perf, o.decode.perf, o.peakFlops])
    const xMin = Math.max(0.1, Math.min(...ais) / 3)
    const xMax = Math.max(...ais) * 3
    const yMin = Math.min(...perfs) / 5
    const yMax = Math.max(...perfs) * 2
    return { xMin, xMax, yMin, yMax }
  })()

  function xPx(ai: number, d: NonNullable<typeof domain>): number {
    const l0 = Math.log10(d.xMin), l1 = Math.log10(d.xMax)
    return M.left + ((Math.log10(ai) - l0) / (l1 - l0)) * plotW
  }
  function yPx(perf: number, d: NonNullable<typeof domain>): number {
    const l0 = Math.log10(d.yMin), l1 = Math.log10(d.yMax)
    return M.top + plotH - ((Math.log10(perf) - l0) / (l1 - l0)) * plotH
  }

  // Roof = piecewise linear in log-log: rising line (perf = ai × BW) until ridge,
  // then flat at peakFlops. Compute three anchor points and let SVG join them.
  function roofPath(op: OpData, d: NonNullable<typeof domain>): string {
    const leftPerf = Math.max(d.yMin / 10, d.xMin * op.peakBw)
    const rightPerf = op.peakFlops
    return [
      `M ${xPx(d.xMin, d).toFixed(1)} ${yPx(leftPerf, d).toFixed(1)}`,
      `L ${xPx(op.ridgeAi, d).toFixed(1)} ${yPx(op.peakFlops, d).toFixed(1)}`,
      `L ${xPx(d.xMax, d).toFixed(1)} ${yPx(rightPerf, d).toFixed(1)}`
    ].join(' ')
  }

  function logTicks(min: number, max: number): number[] {
    const out: number[] = []
    const lo = Math.floor(Math.log10(min)), hi = Math.ceil(Math.log10(max))
    for (let e = lo; e <= hi; e++) {
      const v = 10 ** e
      if (v >= min && v <= max) out.push(v)
    }
    return out
  }
  function fmtAxis(v: number): string {
    if (v >= 1e15) return `${(v / 1e15).toFixed(0)} PF`
    if (v >= 1e12) return `${(v / 1e12).toFixed(0)} TF`
    if (v >= 1e9)  return `${(v / 1e9).toFixed(0)} GF`
    if (v >= 1000) return `${(v / 1000).toFixed(0)}k`
    if (v >= 1)    return v.toFixed(v < 10 ? 1 : 0)
    return v.toString()
  }
</script>

{#if domain && ops.length > 0}
  <section class="roofline">
    <h3>Roofline</h3>
    <p class="caption">
      Arithmetic intensity (FLOPs/byte) vs attainable performance.
      The sloped segment is memory-bound; the flat segment is compute-bound.
      Each operating point uses {$input?.quant.activations} compute throughput.
    </p>
    <svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Roofline plot — log-log axes, attainable performance vs arithmetic intensity">
      <!-- gridlines -->
      {#each logTicks(domain.xMin, domain.xMax) as t}
        <line x1={xPx(t, domain)} y1={M.top}
              x2={xPx(t, domain)} y2={M.top + plotH}
              class="grid" />
      {/each}
      {#each logTicks(domain.yMin, domain.yMax) as t}
        <line x1={M.left} y1={yPx(t, domain)}
              x2={M.left + plotW} y2={yPx(t, domain)}
              class="grid" />
      {/each}

      <!-- axes -->
      <line x1={M.left} y1={M.top + plotH} x2={M.left + plotW} y2={M.top + plotH} class="axis"/>
      <line x1={M.left} y1={M.top} x2={M.left} y2={M.top + plotH} class="axis"/>

      <!-- x tick labels -->
      {#each logTicks(domain.xMin, domain.xMax) as t}
        <text x={xPx(t, domain)} y={M.top + plotH + 16} class="tick">{fmtAxis(t)}</text>
      {/each}
      <!-- y tick labels (in FLOPs, formatted as TF/PF) -->
      {#each logTicks(domain.yMin, domain.yMax) as t}
        <text x={M.left - 8} y={yPx(t, domain) + 4} class="tick" text-anchor="end">{fmtAxis(t)}/s</text>
      {/each}

      <!-- axis titles -->
      <text x={M.left + plotW / 2} y={H - 12} class="axis-title">Arithmetic intensity (FLOPs/byte)</text>
      <text x={16} y={M.top + plotH / 2} class="axis-title"
            transform="rotate(-90, 16, {M.top + plotH / 2})">Performance</text>

      <!-- rooflines and markers, one set per operating point -->
      {#each ops as op}
        <path d={roofPath(op, domain)} fill="none" stroke={op.color} stroke-width="2"
              stroke-dasharray={op.id === 'peak' ? '0' : '6 4'} />
        <!-- ridge marker -->
        <circle cx={xPx(op.ridgeAi, domain)} cy={yPx(op.peakFlops, domain)}
                r="2.5" fill={op.color} opacity="0.5" />
        <!-- prefill point: filled square -->
        <rect x={xPx(op.prefill.ai, domain) - 5} y={yPx(op.prefill.perf, domain) - 5}
              width="10" height="10" fill={op.color}
              stroke="#fff" stroke-width="1.5">
          <title>{op.label} prefill — AI {op.prefill.ai.toFixed(1)}, {(op.prefill.perf / 1e12).toFixed(1)} TF/s ({op.prefill.regime}-bound)</title>
        </rect>
        <!-- decode point: filled circle -->
        <circle cx={xPx(op.decode.ai, domain)} cy={yPx(op.decode.perf, domain)}
                r="6" fill={op.color} stroke="#fff" stroke-width="1.5">
          <title>{op.label} decode — AI {op.decode.ai.toFixed(2)}, {(op.decode.perf / 1e12).toFixed(2)} TF/s ({op.decode.regime}-bound)</title>
        </circle>
      {/each}
    </svg>

    <div class="legend">
      {#each ops as op}
        <span class="entry" style="--c: {op.color}">
          <span class="swatch" class:dashed={op.id !== 'peak'}></span>
          <span class="name">{op.label}</span>
        </span>
      {/each}
      <span class="entry">
        <span class="shape square"></span>
        <span class="name">prefill</span>
      </span>
      <span class="entry">
        <span class="shape circle"></span>
        <span class="name">decode</span>
      </span>
    </div>
  </section>
{/if}

<style>
  .roofline { margin-top: 1.5rem; }
  h3 { margin-bottom: 0.25rem; }
  .caption {
    font-size: 0.85rem; color: #555; margin: 0 0 0.5rem; font-style: italic;
  }
  svg { max-width: 100%; height: auto; display: block; }
  .grid { stroke: #eee; stroke-width: 1; }
  .axis { stroke: #888; stroke-width: 1; }
  .tick {
    font-size: 11px; fill: #555; font-family: ui-monospace, monospace;
    dominant-baseline: middle; text-anchor: middle;
  }
  .axis-title {
    font-size: 12px; fill: #333; font-weight: 500; text-anchor: middle;
  }

  .legend {
    display: flex; flex-wrap: wrap; gap: 0.75rem 1.2rem;
    font-size: 0.85rem; margin-top: 0.5rem; color: #333;
  }
  .entry { display: inline-flex; align-items: center; gap: 0.35rem; }
  .swatch {
    width: 18px; height: 0; border-top: 2px solid var(--c, #888);
  }
  .swatch.dashed { border-top-style: dashed; }
  .shape {
    width: 10px; height: 10px; background: #555;
    border: 1.5px solid #fff; box-shadow: 0 0 0 1px #555;
  }
  .shape.square { border-radius: 0; }
  .shape.circle { border-radius: 50%; }
</style>

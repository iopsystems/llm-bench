<script lang="ts">
  import * as Plot from '@observablehq/plot'
  import { input, result } from './stores'

  let container: HTMLDivElement | undefined = $state(undefined)

  // The roofline is piecewise: rising line (perf = AI × peak_BW) up to the
  // ridge, then flat at peak compute. We emit three anchor points per op so
  // Plot.line draws the bent ceiling. X anchors track the plot domain so the
  // segments span the full x extent independent of how points fall.
  type Row = {
    op: string
    kind: 'roof' | 'prefill' | 'decode'
    ai: number
    perf: number
    regime?: 'compute' | 'memory'
    dash?: boolean
  }

  const data = $derived.by(() => {
    const empty = { roofs: [] as Row[], points: [] as Row[],
                    xMin: 0.1, xMax: 1000, yMin: 1e10, yMax: 1e15 }
    if (!$input || !$result) return empty
    const variant = $input.gpu.variants.find(v => v.id === $input.gpuVariantId)
    if (!variant) return empty

    const roofs: Row[] = []
    const points: Row[] = []
    const ais: number[] = []
    const perfs: number[] = []

    for (const op of variant.operatingPoints) {
      const t = op.tflops[$input.quant.activations]
      const p = $result.perf[op.id]
      if (t === undefined || !p) continue
      const peakFlops = t * 1e12
      const peakBw = op.hbmBandwidthGBs * 1e9
      const ridge = peakFlops / peakBw
      const isPeak = op.id === 'peak'

      // Three anchors: low-x, ridge, high-x. We pad domain ends generously
      // because the actual x bounds are set globally after this loop.
      roofs.push({ op: op.label, kind: 'roof', ai: 1e-3, perf: 1e-3 * peakBw, dash: !isPeak })
      roofs.push({ op: op.label, kind: 'roof', ai: ridge,  perf: peakFlops,     dash: !isPeak })
      roofs.push({ op: op.label, kind: 'roof', ai: 1e6,    perf: peakFlops,     dash: !isPeak })

      const prefAi = p.prefill.flops / p.prefill.bytes
      const prefPerf = p.prefill.flops / p.prefill.timeS
      const decAi = p.decode.flopsPerStep / p.decode.bytesPerStep
      const decPerf = p.decode.flopsPerStep / p.decode.timePerTokenS

      points.push({ op: op.label, kind: 'prefill', ai: prefAi, perf: prefPerf, regime: p.prefill.regime })
      points.push({ op: op.label, kind: 'decode',  ai: decAi,  perf: decPerf,  regime: p.decode.regime })

      ais.push(ridge, prefAi, decAi)
      perfs.push(peakFlops, prefPerf, decPerf)
    }

    // Pad observed range by half a decade in each direction so points aren't on the edge.
    const xMin = Math.max(0.05, Math.min(...ais) / 3)
    const xMax = Math.max(...ais) * 3
    const yMin = Math.min(...perfs) / 5
    const yMax = Math.max(...perfs) * 2
    return { roofs, points, xMin, xMax, yMin, yMax }
  })

  function fmtPerf(v: number): string {
    if (v >= 1e15) return `${(v / 1e15).toFixed(1)} PFLOPS`
    if (v >= 1e12) return `${(v / 1e12).toFixed(0)} TFLOPS`
    if (v >= 1e9)  return `${(v / 1e9).toFixed(0)} GFLOPS`
    return `${v.toExponential(1)} F`
  }

  const chart = $derived.by(() => {
    if (data.roofs.length === 0) return null
    return Plot.plot({
      width: 640, height: 380,
      marginLeft: 70, marginBottom: 50, marginRight: 24, marginTop: 24,
      x: {
        type: 'log',
        domain: [data.xMin, data.xMax],
        label: 'Arithmetic intensity (FLOPs/byte) →',
        grid: true
      },
      y: {
        type: 'log',
        domain: [data.yMin, data.yMax],
        label: '↑ Attainable performance',
        tickFormat: (d: number) => fmtPerf(d),
        grid: true
      },
      color: { legend: true, label: 'Operating point' },
      symbol: {
        legend: true,
        label: 'Phase',
        domain: ['prefill', 'decode'],
        range: ['square', 'circle']
      },
      marks: [
        // Solid for peak, dashed for non-peak (achievable etc.)
        Plot.line(data.roofs.filter(r => !r.dash), {
          x: 'ai', y: 'perf', stroke: 'op', strokeWidth: 2
        }),
        Plot.line(data.roofs.filter(r => r.dash), {
          x: 'ai', y: 'perf', stroke: 'op', strokeWidth: 2, strokeDasharray: '6 4'
        }),
        Plot.dot(data.points, {
          x: 'ai', y: 'perf', stroke: 'op', fill: 'op', symbol: 'kind',
          r: 7, strokeWidth: 1.5,
          tip: {
            format: { x: '.3~f', y: (d: number) => fmtPerf(d) + '/s' }
          }
        })
      ]
    })
  })

  $effect(() => {
    if (!container) return
    container.replaceChildren()
    if (chart) container.appendChild(chart)
  })
</script>

{#if data.roofs.length > 0}
  <section class="roofline">
    <h3>Roofline</h3>
    <p class="caption">
      Sloped roof = memory-bound (perf grows with AI × BW). Flat roof = compute-bound (perf
      capped at peak FLOPS for {$input?.quant.activations}). Markers show where this
      workload's prefill and decode land.
    </p>
    <div bind:this={container} class="plot"></div>
  </section>
{/if}

<style>
  .roofline { margin-top: 1.5rem; }
  h3 { margin-bottom: 0.25rem; }
  .caption {
    font-size: 0.85rem; color: #555; margin: 0 0 0.5rem; font-style: italic;
  }
  .plot { max-width: 100%; overflow-x: auto; }
  .plot :global(svg) { max-width: 100%; height: auto; }
</style>

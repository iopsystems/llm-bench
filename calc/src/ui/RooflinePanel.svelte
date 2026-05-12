<script lang="ts">
  import * as Plot from '@observablehq/plot'
  import { input, result } from './stores'

  let container: HTMLDivElement | undefined = $state(undefined)

  // The roofline is piecewise: rising line (perf = AI × peak_BW) up to the
  // ridge, then flat at peak compute. We emit three anchor points per op so
  // Plot.line draws the bent ceiling. X anchors track the plot domain so the
  // segments span the full x extent independent of how points fall.
  // One roofline (the theoretical peak — the absolute ceiling). All operating
  // points contribute markers showing where the workload's prefill and decode
  // actually land for that tier. Gap between a marker and the roof above it
  // is the hardware-efficiency loss for this workload/quant combo.
  type RoofRow  = { ai: number; perf: number }
  type PointRow = {
    tier: 'Theoretical' | 'Attainable'
    phase: 'prefill' | 'decode'
    ai: number
    perf: number
    regime: 'compute' | 'memory'
  }
  type GapRow = {
    phase: 'prefill' | 'decode'
    ai: number
    perf: number
  }

  const data = $derived.by(() => {
    const empty = { roofs: [] as RoofRow[], points: [] as PointRow[],
                    gaps: [] as GapRow[],
                    xMin: 0.1, xMax: 1000, yMin: 1e10, yMax: 1e15 }
    if (!$input || !$result) return empty
    const variant = $input.gpu.variants.find(v => v.id === $input.gpuVariantId)
    if (!variant) return empty

    const peakOp = variant.operatingPoints.find(o => o.id === 'peak')
      ?? variant.operatingPoints[0]
    if (!peakOp) return empty
    const peakT = peakOp.tflops[$input.quant.activations]
    if (peakT === undefined) return empty

    const peakFlops = peakT * 1e12
    const peakBw = peakOp.hbmBandwidthGBs * 1e9
    const ridge = peakFlops / peakBw

    const roofs: RoofRow[] = [
      { ai: 1e-3, perf: 1e-3 * peakBw },
      { ai: ridge, perf: peakFlops },
      { ai: 1e6,   perf: peakFlops }
    ]

    const points: PointRow[] = []
    const gaps: GapRow[] = []
    const ais: number[] = [ridge]
    const perfs: number[] = [peakFlops]

    for (const op of variant.operatingPoints) {
      const t = op.tflops[$input.quant.activations]
      const p = $result.perf[op.id]
      if (t === undefined || !p) continue
      const tier: PointRow['tier'] = op.id === 'peak' ? 'Theoretical' : 'Attainable'

      const prefAi = p.prefill.flops / p.prefill.bytes
      const prefPerf = p.prefill.flops / p.prefill.timeS
      const decAi = p.decode.flopsPerStep / p.decode.bytesPerStep
      const decPerf = p.decode.flopsPerStep / p.decode.timePerTokenS

      points.push({ tier, phase: 'prefill', ai: prefAi, perf: prefPerf, regime: p.prefill.regime })
      points.push({ tier, phase: 'decode',  ai: decAi,  perf: decPerf,  regime: p.decode.regime })

      // For non-peak tiers, emit a connector segment from this point up to
      // where the same AI hits the peak roofline. Visualizes the gap directly.
      if (op.id !== 'peak') {
        const prefCeil = Math.min(peakFlops, prefAi * peakBw)
        const decCeil  = Math.min(peakFlops, decAi  * peakBw)
        gaps.push({ phase: 'prefill', ai: prefAi, perf: prefPerf })
        gaps.push({ phase: 'prefill', ai: prefAi, perf: prefCeil })
        gaps.push({ phase: 'decode',  ai: decAi,  perf: decPerf })
        gaps.push({ phase: 'decode',  ai: decAi,  perf: decCeil })
      }

      ais.push(prefAi, decAi)
      perfs.push(prefPerf, decPerf)
    }

    const xMin = Math.max(0.05, Math.min(...ais) / 3)
    const xMax = Math.max(...ais) * 3
    const yMin = Math.min(...perfs) / 5
    const yMax = Math.max(...perfs) * 2
    return { roofs, points, gaps, xMin, xMax, yMin, yMax }
  })

  function fmtPerf(v: number): string {
    if (v >= 1e15) return `${(v / 1e15).toFixed(1)} PFLOPS`
    if (v >= 1e12) return `${(v / 1e12).toFixed(0)} TFLOPS`
    if (v >= 1e9)  return `${(v / 1e9).toFixed(0)} GFLOPS`
    return `${v.toExponential(1)} F`
  }

  const chart = $derived.by(() => {
    if (data.roofs.length === 0) return null
    // Only show the tier legend when there's actually a comparison to make.
    const hasAttainable = data.points.some(p => p.tier === 'Attainable')
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
        label: '↑ Performance',
        tickFormat: (d: number) => fmtPerf(d),
        grid: true
      },
      color: {
        legend: hasAttainable,
        label: 'Tier',
        domain: ['Theoretical', 'Attainable'],
        range: ['#888', '#e07a1f']
      },
      symbol: {
        legend: true,
        label: 'Phase',
        domain: ['prefill', 'decode'],
        range: ['square', 'circle']
      },
      marks: [
        // Theoretical-peak roofline — the absolute ceiling for the chosen dtype.
        Plot.line(data.roofs, { x: 'ai', y: 'perf', stroke: '#888', strokeWidth: 2 }),
        // Gap connectors from attainable points up to the roof at their AI.
        Plot.line(data.gaps, {
          x: 'ai', y: 'perf', stroke: '#bbb', strokeWidth: 1, strokeDasharray: '2 3', z: 'phase'
        }),
        Plot.dot(data.points, {
          x: 'ai', y: 'perf',
          stroke: 'tier', fill: 'tier', symbol: 'phase',
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
      Roof = theoretical ceiling at peak {$input?.quant.activations} (sloped = memory-bound,
      flat = compute-bound). Markers are the workload's prefill and decode; the gap between
      the attainable marker and the roof above it is the hardware-efficiency loss.
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

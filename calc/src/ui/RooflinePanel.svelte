<script lang="ts">
  import * as Plot from '@observablehq/plot'
  import { input, result, multiDevice } from './stores'
  import { INTERCONNECTS } from '../data/interconnects'
  import { PLOT_STYLE } from './plotDefaults'

  let container: HTMLDivElement | undefined = $state(undefined)

  // Each operating tier (Theoretical = peak, Achievable = non-peak) gets its
  // own roofline AND markers, sharing one color via the tier scale.
  // Single-device, markers sit on their tier's roof by construction — time =
  // max(F/C, B/M), so achieved rate is min(C, AI×M), i.e. exactly the roof.
  // Multi-device adds a third term (comms): a comms-bound marker plots BELOW
  // the roof, and not on the purple interconnect line either — that line is
  // drawn in HBM-AI space (x = flops ÷ HBM bytes) while comms time divides by
  // comms bytes, a different denominator.
  type RoofRow = {
    tier: 'Theoretical' | 'Achievable'
    ai: number
    perf: number
  }
  type PointRow = {
    tier: 'Theoretical' | 'Achievable'
    phase: 'prefill' | 'decode'
    ai: number
    perf: number
    regime: 'compute' | 'memory' | 'comms'
    // X-factors that lift the plotted rate off its tier's roof (currently
    // MTP — decode emits multiple tokens per forward pass). Shown as a
    // "Note: …" row in the tooltip; absent for points that sit on the roof.
    note?: string
  }
  type GapRow = {
    phase: 'prefill' | 'decode'
    ai: number
    perf: number
  }

  const data = $derived.by(() => {
    const empty = { roofs: [] as RoofRow[], points: [] as PointRow[],
                    gaps: [] as GapRow[], ridge: 1,
                    xMin: 0.1, xMax: 1000, yMin: 1e10, yMax: 1e15 }
    if (!$input || !$result) return empty
    const variant = $input.accelerator.variants.find(v => v.id === $input.acceleratorVariantId)
    if (!variant) return empty

    const peakOp = variant.operatingPoints.find(o => o.id === 'peak')
      ?? variant.operatingPoints[0]
    if (!peakOp) return empty
    const peakT = peakOp.tflops[$input.quant.activations]
    if (peakT === undefined) return empty
    const peakFlops = peakT * 1e12
    const peakBw = peakOp.hbmBandwidthGBs * 1e9

    const roofs: RoofRow[] = []
    const points: PointRow[] = []
    const gaps: GapRow[] = []
    const ais: number[] = []
    const perfs: number[] = []

    for (const op of variant.operatingPoints) {
      const t = op.tflops[$input.quant.activations]
      const p = $result.perf[op.id]
      if (t === undefined || !p) continue
      const tier: RoofRow['tier'] = op.id === 'peak' ? 'Theoretical' : 'Achievable'
      const opFlops = t * 1e12
      const opBw = op.hbmBandwidthGBs * 1e9
      const opRidge = opFlops / opBw

      // Three anchors per tier: low-x rising segment, ridge, high-x flat.
      roofs.push({ tier, ai: 1e-3, perf: 1e-3 * opBw })
      roofs.push({ tier, ai: opRidge, perf: opFlops })
      roofs.push({ tier, ai: 1e6,    perf: opFlops })

      const prefAi = p.prefill.flops / p.prefill.bytes
      const prefPerf = p.prefill.flops / p.prefill.timeS
      const decAi = p.decode.flopsPerStep / p.decode.bytesPerStep
      const decPerf = p.decode.flopsPerStep / p.decode.timePerTokenS

      const mtpFactor = 1 + $input.model.numNextnLayers
      const decNote = mtpFactor > 1
        ? `MTP ×${mtpFactor} (${mtpFactor} tokens per forward pass)`
        : undefined

      points.push({ tier, phase: 'prefill', ai: prefAi, perf: prefPerf, regime: p.prefill.regime })
      points.push({ tier, phase: 'decode',  ai: decAi,  perf: decPerf,  regime: p.decode.regime, note: decNote })

      // Connector from achievable marker up to peak ceiling at the same AI.
      // The vertical span is the hardware-efficiency gap for this phase.
      if (op.id !== 'peak') {
        const prefCeil = Math.min(peakFlops, prefAi * peakBw)
        const decCeil  = Math.min(peakFlops, decAi  * peakBw)
        gaps.push({ phase: 'prefill', ai: prefAi, perf: prefPerf })
        gaps.push({ phase: 'prefill', ai: prefAi, perf: prefCeil })
        gaps.push({ phase: 'decode',  ai: decAi,  perf: decPerf })
        gaps.push({ phase: 'decode',  ai: decAi,  perf: decCeil })
      }

      ais.push(opRidge, prefAi, decAi)
      perfs.push(opFlops, prefPerf, decPerf)
    }

    const xMin = Math.max(0.05, Math.min(...ais) / 3)
    const xMax = Math.max(...ais) * 3
    const yMin = Math.min(...perfs) / 5
    const yMax = Math.max(...perfs) * 2
    // Theoretical ridge — splits memory-bound (AI < ridge) from compute-bound
    // (AI > ridge). Used by the background shading.
    const ridge = peakFlops / peakBw
    return { roofs, points, gaps, ridge, xMin, xMax, yMin, yMax }
  })

  function fmtPerf(v: number): string {
    if (v >= 1e15) return `${(v / 1e15).toFixed(1)} PFLOPS`
    if (v >= 1e12) return `${(v / 1e12).toFixed(0)} TFLOPS`
    if (v >= 1e9)  return `${(v / 1e9).toFixed(0)} GFLOPS`
    return `${v.toExponential(1)} F`
  }

  // Arithmetic intensity — 3 sig figs with K/M/G suffix. toPrecision(3) plus
  // parseFloat→toString strips trailing zeros (12345 → "12.3K", 1.5 → "1.5").
  function fmtAi(v: number): string {
    const sig = (x: number) => parseFloat(x.toPrecision(3)).toString()
    if (v >= 1e9) return `${sig(v / 1e9)}G`
    if (v >= 1e6) return `${sig(v / 1e6)}M`
    if (v >= 1e3) return `${sig(v / 1e3)}K`
    return sig(v)
  }

  function fmtRegime(r: 'compute' | 'memory' | 'comms'): string {
    if (r === 'memory')  return 'Memory-bound'
    if (r === 'compute') return 'Compute-bound'
    return 'Comms-bound'
  }

  const hasAchievable = $derived(data.points.some(p => p.tier === 'Achievable'))

  // Interconnect ceiling — only when a multi-device system is active.
  const interconnectBwGBs = $derived.by(() => {
    if (!$multiDevice) return null
    const ic = INTERCONNECTS.find(i => i.id === $multiDevice!.system.interconnectId)
    if (!ic) return null
    // Use explicit perDirectionGBs if set; fall back to half of bidirectional.
    return ic.perDirectionGBs ?? ic.perGpuBandwidthGBs / 2
  })

  const chart = $derived.by(() => {
    if (data.roofs.length === 0) return null
    return Plot.plot({
      width: 640, height: 380,
      // marginLeft bumped from 70 → 100 so longer y-tick labels like
      // "1.0 PFLOPS/s" fit without clipping.
      marginLeft: 100, marginBottom: 50, marginRight: 24, marginTop: 24,
      style: PLOT_STYLE,
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
        // We render our own legend below the chart so the swatch can match
        // the actual stroke style (solid for Theoretical, dashed for Achievable).
        legend: false,
        domain: ['Theoretical', 'Achievable'],
        // Green avoids collision with the compute-regime badge (orange) and
        // the memory-regime badge (blue), so the Achievable color doesn't
        // accidentally suggest a particular bottleneck.
        range: ['#888', '#21a87a']
      },
      symbol: {
        legend: false,
        domain: ['prefill', 'decode'],
        range: ['square', 'circle']
      },
      marks: [
        // Background shading by regime — drawn first so it sits behind the
        // rooflines and markers. Tints reuse the regime-badge palette
        // (light blue for memory, light orange for compute) so the visual
        // language is consistent between the chart and the perf table.
        Plot.rect([
          { x1: data.xMin, x2: data.ridge, y1: data.yMin, y2: data.yMax }
        ], { x1: 'x1', x2: 'x2', y1: 'y1', y2: 'y2',
             fill: '#c8dcfd', fillOpacity: 0.25, stroke: null, clip: true }),
        Plot.rect([
          { x1: data.ridge, x2: data.xMax, y1: data.yMin, y2: data.yMax }
        ], { x1: 'x1', x2: 'x2', y1: 'y1', y2: 'y2',
             fill: '#fde6c8', fillOpacity: 0.25, stroke: null, clip: true }),
        // Theoretical-peak roofline (solid). The roof anchors extend well
        // beyond the visible domain (so the rising/flat segments span the
        // whole plot regardless of where the data falls); clip: true trims
        // the rendered stroke to the plot frame.
        Plot.line(data.roofs.filter(r => r.tier === 'Theoretical'), {
          x: 'ai', y: 'perf', stroke: 'tier', strokeWidth: 2, clip: true
        }),
        // Achievable roofline (dashed) — separate mark so we can dash it.
        Plot.line(data.roofs.filter(r => r.tier === 'Achievable'), {
          x: 'ai', y: 'perf', stroke: 'tier', strokeWidth: 2, strokeDasharray: '6 4', clip: true
        }),
        // Gap connectors from achievable points up to the peak ceiling at their AI.
        Plot.line(data.gaps, {
          x: 'ai', y: 'perf', stroke: '#bbb', strokeWidth: 1,
          strokeDasharray: '2 3', z: 'phase', clip: true
        }),
        // Interconnect bandwidth ceiling (dashed purple) — lower slope than HBM,
        // represents the comms bottleneck when collective traffic saturates the fabric.
        ...(interconnectBwGBs ? [
          Plot.line(
            [{ x: data.xMin, y: data.xMin * interconnectBwGBs * 1e9 },
             { x: data.xMax, y: data.xMax * interconnectBwGBs * 1e9 }],
            { x: 'x', y: 'y', stroke: '#9b59b6', strokeWidth: 1.5,
              strokeDasharray: '4,2', clip: true }
          )
        ] : []),
        // Single dot mark = single interactive tip. Splitting note/no-note
        // points into two `tip`-bearing marks installs two competing pointer
        // handlers that re-render each other's tip layer on every pointermove —
        // the tooltip flickers and won't stick. One mark fixes that; the Note
        // channel returns null for no-note points, and Plot's tip omits
        // null-valued channel rows, so we still avoid an empty "Note: " line.
        Plot.dot(data.points, {
          x: 'ai', y: 'perf',
          stroke: 'tier', fill: 'tier', fillOpacity: 0.7, symbol: 'phase',
          r: 7, strokeWidth: 1.5,
          // Custom channels give the tooltip its own labels — independent of
          // the axis titles — and a controlled display order (Performance,
          // then a blank spacer, then the rest).
          channels: {
            Performance: { value: 'perf', label: 'Performance' },
            ' ': { value: () => '', label: ' ' },
            'Arithmetic Intensity': { value: 'ai', label: 'Arithmetic Intensity' },
            Bottleneck: { value: 'regime', label: 'Bottleneck' },
            Note: { value: (d: { note?: string }) => d.note ?? null, label: 'Note' }
          },
          tip: {
            format: {
              x: false, y: false,
              stroke: false, fill: false,
              Performance: (d: number) => fmtPerf(d) + '/s',
              'Arithmetic Intensity': fmtAi,
              Bottleneck: fmtRegime,
              Note: (n: string) => n
            }
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
      flat = compute-bound{#if interconnectBwGBs}, dashed purple = interconnect-bound{/if}).
      Markers are the workload's prefill and decode; the gap between the achievable marker
      and the roof above it is the hardware-efficiency loss.
      {#if interconnectBwGBs}Comms-bound markers sit below the roof: the on-chip
      ceilings don't include collective traffic (check the marker tooltip's
      Bottleneck row).{/if}
    </p>
    <div class="chart-row">
    <div bind:this={container} class="plot"></div>
    <div class="legend">
      <div class="legend-group">
        {#if hasAchievable}
          <span class="entry">
            <svg class="line-swatch" viewBox="0 0 22 10" aria-hidden="true">
              <line x1="1" y1="5" x2="21" y2="5" stroke="#888" stroke-width="2"/>
            </svg>
            <span>Theoretical</span>
          </span>
          <span class="entry">
            <svg class="line-swatch" viewBox="0 0 22 10" aria-hidden="true">
              <line x1="1" y1="5" x2="21" y2="5" stroke="#21a87a" stroke-width="2" stroke-dasharray="6 4"/>
            </svg>
            <span>Achievable</span>
          </span>
        {/if}
        {#if interconnectBwGBs}
          <span class="entry">
            <svg class="line-swatch" viewBox="0 0 22 10" aria-hidden="true">
              <line x1="1" y1="5" x2="21" y2="5" stroke="#9b59b6" stroke-width="2" stroke-dasharray="4 2"/>
            </svg>
            <span>Interconnect BW</span>
          </span>
        {/if}
      </div>
      <div class="legend-group">
        <span class="entry">
          <svg class="shape-swatch" viewBox="0 0 12 12" aria-hidden="true">
            <rect x="1" y="1" width="10" height="10" fill="#888" stroke="#fff" stroke-width="1"/>
          </svg>
          <span>prefill</span>
        </span>
        <span class="entry">
          <svg class="shape-swatch" viewBox="0 0 12 12" aria-hidden="true">
            <circle cx="6" cy="6" r="5" fill="#888" stroke="#fff" stroke-width="1"/>
          </svg>
          <span>decode</span>
        </span>
      </div>
    </div>
    </div>
  </section>
{/if}

<style>
  .roofline { margin-top: 1.5rem; }
  h3 { margin-bottom: 0.25rem; }
  .caption {
    font-size: 0.85rem; color: #555; margin: 0 0 0.5rem; font-style: italic;
  }
  .chart-row {
    display: flex; flex-direction: row; align-items: center;
    gap: 1rem;
    /* Shrink to chart + legend natural width and center as a unit
       within the section, aligning with the surrounding text. */
    width: fit-content; max-width: 100%; margin: 0 auto;
  }
  .plot {
    flex: 0 0 auto; min-width: 0;
    max-width: 100%; overflow-x: auto;
  }
  .plot :global(svg) { max-width: 100%; height: auto; display: inline-block; }
  /* Legend default = vertical column to the right of the chart, split into
     two sub-groups (line legends ↑, marker legends ↓) so readers don't
     confuse line styles with marker shapes. */
  .legend {
    display: flex; flex-direction: column; gap: 2.4rem;
    flex: 0 0 auto;
    font-size: 0.85rem; color: #333;
  }
  .legend-group {
    display: flex; flex-direction: column; gap: 0.4rem;
  }
  .entry { display: inline-flex; align-items: center; gap: 0.35rem; }
  .line-swatch { width: 22px; height: 10px; }
  .shape-swatch { width: 12px; height: 12px; }
  /* Below ~720px viewport, drop the legend under the chart and lay it out
     horizontally so it occupies less vertical real estate. The two
     sub-groups stay visually separated via a generous inter-group gap. */
  @media (max-width: 720px) {
    .chart-row { flex-direction: column; }
    .legend {
      flex-direction: row; flex-wrap: wrap; gap: 0.4rem 3.5rem;
      margin: 0.4rem 0 0; padding-left: 100px;
    }
    .legend-group {
      flex-direction: row; flex-wrap: wrap; gap: 0.4rem 1.1rem;
    }
  }
</style>

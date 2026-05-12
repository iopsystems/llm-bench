<script lang="ts">
  import * as Plot from '@observablehq/plot'
  import { result } from './stores'
  import { PLOT_STYLE } from './plotDefaults'

  let container: HTMLDivElement | undefined = $state(undefined)
  // Track the container's actual width so we can size the Plot SVG to match
  // exactly — eliminates the SVG-aspect-ratio letterboxing that previously
  // forced a `preserveAspectRatio="none"` hack (which stretched the tooltip
  // text along with everything else).
  let containerWidth = $state(640)

  const GB = 1024 ** 3
  function gb(bytes: number): string { return (bytes / GB).toFixed(2) }

  // Each region uses a textured fill — a base color plus a pattern (stripes at
  // distinct angles, or dots). Patterns are robust against printing, color-
  // blindness, and overlap with other plots' palettes.
  type Component = 'Weights' | 'KV cache' | 'Activations'
  const PATTERN_IDS: Record<Component, string> = {
    Weights: 'mem-pat-weights',
    'KV cache': 'mem-pat-kv',
    Activations: 'mem-pat-acts'
  }
  // Base colors are kept light so the darker pattern strokes stand out.
  const PATTERN_BASE: Record<Component, string> = {
    Weights: '#cfdcec',
    'KV cache': '#dcd0f5',
    Activations: '#c6e6e1'
  }
  const PATTERN_STROKE: Record<Component, string> = {
    Weights: '#4682b4',
    'KV cache': '#7c5fc7',
    Activations: '#1d8a7e'
  }

  function patternDefsSvg(): string {
    // Inline pattern definitions appended to Plot's SVG before render.
    return `
      <pattern id="${PATTERN_IDS['Weights']}" patternUnits="userSpaceOnUse"
               width="6" height="6" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="${PATTERN_BASE['Weights']}"/>
        <line x1="0" y1="0" x2="0" y2="6"
              stroke="${PATTERN_STROKE['Weights']}" stroke-width="2"/>
      </pattern>
      <pattern id="${PATTERN_IDS['KV cache']}" patternUnits="userSpaceOnUse"
               width="6" height="6">
        <rect width="6" height="6" fill="${PATTERN_BASE['KV cache']}"/>
        <circle cx="3" cy="3" r="1.2" fill="${PATTERN_STROKE['KV cache']}"/>
      </pattern>
      <pattern id="${PATTERN_IDS['Activations']}" patternUnits="userSpaceOnUse"
               width="6" height="6" patternTransform="rotate(-45)">
        <rect width="6" height="6" fill="${PATTERN_BASE['Activations']}"/>
        <line x1="0" y1="0" x2="0" y2="6"
              stroke="${PATTERN_STROKE['Activations']}" stroke-width="2"/>
      </pattern>
    `
  }

  const chart = $derived.by(() => {
    if (!$result) return null
    const m = $result.memory
    const capBytes = m.hbmCapacityGB * GB
    // Build explicit x1/x2 ranges (cumulative left-edge) so the bars don't
    // rely on Plot's implicit stack transform, which was producing a small
    // left-edge shift in some OOM cases.
    const rawParts = [
      { component: 'Weights',     bytes: m.weights },
      { component: 'KV cache',    bytes: m.kvCacheTotal },
      { component: 'Activations', bytes: m.activationsPeak }
    ]
    let cum = 0
    const parts = rawParts.map(p => {
      const x1 = cum
      cum += p.bytes
      return { ...p, x1, x2: cum }
    })

    return Plot.plot({
      width: containerWidth, height: 28,
      marginLeft: 0, marginRight: 0, marginTop: 0, marginBottom: 0,
      style: PLOT_STYLE,
      // Domain represents capacity. When the workload fits, the stacked bars
      // end short of the right edge and the unused remainder shows as the
      // gray background. When OOM, the stack extends past the right edge and
      // gets truncated by clip:true on the bar mark — the overflow
      // components (or their tail) simply don't render.
      x: { domain: [0, capBytes], axis: null },
      // Numeric y domain so we can pin rects to y1=0, y2=1 explicitly.
      y: { domain: [0, 1], axis: null },
      color: {
        domain: Object.keys(PATTERN_IDS),
        range: Object.values(PATTERN_IDS).map(id => `url(#${id})`),
        legend: false
      },
      marks: [
        Plot.rect(parts, {
          // Explicit x1/x2/y1/y2 — pure rectangles with no implicit transform
          // or band-scale insetting. First rect's x1=0 maps to pixel 0 of
          // the plot area (marginLeft is 0), so the bar's left edge is flush
          // with the container's inner left in both fits and OOM states.
          x1: 'x1', x2: 'x2', y1: 0, y2: 1,
          fill: 'component', clip: true,
          insetLeft: 0, insetRight: 0, insetTop: 0, insetBottom: 0,
          tip: {
            format: { x1: false, x2: false, y1: false, y2: false, fill: false }
          },
          channels: {
            // Single channel renders as "Weights: 141.10 GB" — component
            // name plays the role of the label, size formatted with units.
            // Uses a non-breaking space as the channel key so the tooltip
            // shows just the value with no extra "key:" prefix.
            ' ': {
              value: (d: { component: string; bytes: number }) =>
                `${d.component}: ${gb(d.bytes)} GB`,
              label: ''
            }
          }
        })
      ]
    })
  })

  // Track container width so Plot can size the SVG natively — no stretching,
  // no letterboxing, no preserveAspectRatio hack (which would also stretch
  // tooltip text).
  $effect(() => {
    if (!container) return
    const ro = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect.width ?? 0
      if (w > 0 && Math.abs(w - containerWidth) > 0.5) containerWidth = w
    })
    ro.observe(container)
    return () => ro.disconnect()
  })

  $effect(() => {
    if (!container) return
    container.replaceChildren()
    if (chart) {
      // Inject pattern <defs> into Plot's SVG so the fill: url(#...) on each
      // bar can resolve. Plot doesn't have a native pattern API; we splice
      // the defs in before the marks render.
      const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs')
      defs.innerHTML = patternDefsSvg()
      chart.insertBefore(defs, chart.firstChild)
      container.appendChild(chart)
    }
  })
</script>

{#if $result}
  {@const m = $result.memory}
  {@const cap = m.hbmCapacityGB * GB}
  <section class="memory-panel">
    <h3>Memory budget — {gb(cap)} GB</h3>
    <div bind:this={container} class="bar-chart" class:oom={!m.fits}></div>
    <div class="legend">
      {#each Object.keys(PATTERN_IDS) as name (name)}
        {@const id = PATTERN_IDS[name as Component]}
        <span class="entry">
          <svg class="swatch" viewBox="0 0 14 10" aria-hidden="true">
            <defs>{@html patternDefsSvg()}</defs>
            <rect width="14" height="10" fill="url(#{id})"/>
          </svg>
          <span>{name}</span>
        </span>
      {/each}
    </div>
    <table>
      <tbody>
        <tr><td>Weights</td>          <td>{gb(m.weights)} GB</td></tr>
        <tr><td>KV cache (total)</td> <td>{gb(m.kvCacheTotal)} GB</td></tr>
        <tr><td>Activations (~)</td>  <td>{gb(m.activationsPeak)} GB</td></tr>
        <tr class="total"><td>Total</td><td>{gb(m.total)} GB</td></tr>
        <tr>
          <td>Headroom</td>
          <td class="headroom-value">
            {gb(m.headroom)} GB
            <span class="status-badge" class:fits={m.fits} class:oom={!m.fits}>
              {m.fits ? '✓ fits' : '✗ OOM'}
            </span>
          </td>
        </tr>
      </tbody>
    </table>
    <p class="caveat">~ activations estimate assumes FlashAttention-style kernels</p>
  </section>
{/if}

<style>
  .memory-panel { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem; }
  /* No overflow: hidden on container — the Plot.barX clip: true option
     already trims OOM bars at the plot frame, and we want the tooltip
     popup (which appears above the bar) to escape the 28px tall canvas. */
  .bar-chart {
    width: 100%; box-sizing: border-box;
    border: 2px solid #888; background: #f0f0f0;
  }
  /* OOM only changes color — width/thickness stays put so the box doesn't
     visibly grow. */
  .bar-chart.oom { border-color: #c33; }
  .bar-chart :global(svg) { display: block; overflow: visible; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem;
    font-size: 0.85rem; color: #333;
  }
  .entry { display: inline-flex; align-items: center; gap: 0.35rem; }
  .swatch {
    width: 14px; height: 10px;
    display: inline-block;
  }
  /* align-self overrides the parent flex's default align-items:stretch so
     the table sizes to content instead of filling the panel width, then
     centers horizontally within the panel. */
  table {
    font-variant-numeric: tabular-nums; border-collapse: collapse;
    align-self: center;
  }
  td:first-child { padding-right: 2.5rem; }
  /* Memory size column: right-aligned so digits stack to a common edge.
     Extra horizontal padding so the table doesn't look cramped at its
     content-determined width. */
  td:last-child { text-align: right; padding-left: 1rem; }
  /* Headroom row: the GB number stays inside the column (right-aligned with
     all other numbers); the status badge is positioned just outside the
     table's right edge, so it doesn't widen the table. */
  td.headroom-value { position: relative; }
  .status-badge {
    position: absolute;
    left: calc(100% + 0.5rem); top: 50%;
    transform: translateY(-50%);
    white-space: nowrap;
  }
  /* Divider above Total separates the breakdown from the summed totals. */
  tr.total td { border-top: 1px solid #ccc; padding-top: 0.3rem; }
  tr.total { font-weight: bold; }
  /* Status background shading on the Headroom cell. Green for fits, red for
     OOM — matches the conventional traffic-light reading. */
  .fits {
    color: #1d6b45; background: #e6f5ec;
    padding: 0.15rem 0.5rem; border-radius: 0.2rem; font-weight: 600;
  }
  .oom {
    color: #c33; background: #fde8e8;
    padding: 0.15rem 0.5rem; border-radius: 0.2rem; font-weight: 600;
  }
  .caveat { font-size: 0.8rem; color: #666; font-style: italic; }
</style>

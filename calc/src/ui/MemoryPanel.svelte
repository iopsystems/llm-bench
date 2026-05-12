<script lang="ts">
  import * as Plot from '@observablehq/plot'
  import { result } from './stores'

  let container: HTMLDivElement | undefined = $state(undefined)
  // Track the container's actual width so we can size the Plot SVG to match
  // exactly — eliminates the SVG-aspect-ratio letterboxing that previously
  // forced a `preserveAspectRatio="none"` hack (which stretched the tooltip
  // text along with everything else).
  let containerWidth = $state(640)

  const GB = 1024 ** 3
  function gb(bytes: number): string { return (bytes / GB).toFixed(2) }

  // Palette deliberately avoids green and orange — those are reserved for the
  // roofline (green = Achievable tier) and the regime badges (orange = compute-
  // bound). Stays in blue/purple/slate so memory composition reads as its own
  // category visually.
  const COLORS = {
    Weights: '#4a90e2',
    'KV cache': '#8e44ad',
    Activations: '#6b7a8c'
  } as const

  const chart = $derived.by(() => {
    if (!$result) return null
    const m = $result.memory
    const capBytes = m.hbmCapacityGB * GB
    const parts = [
      { component: 'Weights',     bytes: m.weights },
      { component: 'KV cache',    bytes: m.kvCacheTotal },
      { component: 'Activations', bytes: m.activationsPeak }
    ]

    return Plot.plot({
      width: containerWidth, height: 28,
      marginLeft: 0, marginRight: 0, marginTop: 0, marginBottom: 0,
      // Bump the SVG's base font-size so the hover tip reads at a comfortable
      // size; the chart itself has no axis text to be affected.
      style: 'font-size: 13px',
      // Container width represents capacity exactly. Overflow on OOM is
      // clipped here and signaled by the container's red border instead.
      x: { domain: [0, capBytes], axis: null },
      y: { axis: null, padding: 0 },
      color: {
        domain: Object.keys(COLORS),
        range: Object.values(COLORS),
        legend: false
      },
      marks: [
        Plot.barX(parts, {
          x: 'bytes', y: () => '',
          fill: 'component', clip: true,
          insetLeft: 0, insetRight: 0, insetTop: 0, insetBottom: 0,
          tip: {
            // Hide all default channels; we render a single composed line.
            format: { x: false, y: false, fill: false }
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
    if (chart) container.appendChild(chart)
  })
</script>

{#if $result}
  {@const m = $result.memory}
  {@const cap = m.hbmCapacityGB * GB}
  <section class="memory-panel">
    <h3>Memory budget — {gb(cap)} GB</h3>
    <div bind:this={container} class="bar-chart" class:oom={!m.fits}></div>
    <div class="legend">
      {#each Object.entries(COLORS) as [name, color]}
        <span class="entry">
          <span class="swatch" style="background: {color}"></span>
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
          <td>
            {gb(m.headroom)} GB &nbsp;
            <span class:fits={m.fits} class:oom={!m.fits}>
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
    width: 100%;
    border: 1px solid #888; background: #f0f0f0;
  }
  .bar-chart.oom { border-color: #c33; }
  .bar-chart :global(svg) { display: block; overflow: visible; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem;
    font-size: 0.85rem; color: #333;
  }
  .entry { display: inline-flex; align-items: center; gap: 0.35rem; }
  .swatch {
    width: 14px; height: 10px; border-radius: 2px;
    display: inline-block;
  }
  table { font-variant-numeric: tabular-nums; border-collapse: collapse; }
  td:first-child { padding-right: 1rem; }
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

<script lang="ts">
  import * as Plot from '@observablehq/plot'
  import { result } from './stores'

  let container: HTMLDivElement | undefined = $state(undefined)

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
      width: 640, height: 28,
      marginLeft: 0, marginRight: 0, marginTop: 0, marginBottom: 0,
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
          // Force zero inset so the first bar's left edge sits flush with
          // the plot area's left edge (== container's left edge after
          // marginLeft: 0). Plot otherwise adds half-pixel insets.
          insetLeft: 0, insetRight: 0, insetTop: 0, insetBottom: 0,
          tip: { format: { x: false, y: false, fill: false } },
          channels: {
            Component: { value: 'component', label: 'Component' },
            Size: { value: 'bytes', label: 'Size' }
          }
        })
      ]
    })
  })

  $effect(() => {
    if (!container) return
    container.replaceChildren()
    if (chart) {
      // Force the SVG to stretch to the container instead of letterboxing.
      // Default 'xMidYMid meet' preserves the 640×28 aspect ratio and centers
      // the bar within the wider container, leaving visible gaps on both sides.
      chart.setAttribute('preserveAspectRatio', 'none')
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
          <td class:oom={!m.fits}>
            {gb(m.headroom)} GB &nbsp; {m.fits ? '✓ fits' : '✗ OOM'}
          </td>
        </tr>
      </tbody>
    </table>
    <p class="caveat">~ activations estimate assumes FlashAttention-style kernels</p>
  </section>
{/if}

<style>
  .memory-panel { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem; }
  .bar-chart {
    width: 100%; overflow: hidden;
    border: 1px solid #888; background: #f0f0f0;
  }
  .bar-chart.oom { border-color: #c33; }
  /* Force SVG to fill the container width regardless of its intrinsic 640px;
     fixed height since the bar visual has no axis to stretch. */
  .bar-chart :global(svg) { width: 100%; height: 28px; display: block; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem;
    font-size: 0.85rem; color: #333;
  }
  .entry { display: inline-flex; align-items: center; gap: 0.35rem; }
  .swatch {
    width: 14px; height: 10px; border-radius: 2px;
    display: inline-block;
  }
  table { font-variant-numeric: tabular-nums; }
  td:first-child { padding-right: 1rem; }
  tr.total { font-weight: bold; }
  .oom { color: #c33; font-weight: bold; }
  .caveat { font-size: 0.8rem; color: #666; font-style: italic; }
</style>

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
    // Domain extends to whichever is larger so the OOM case visibly overshoots
    // the capacity line.
    const xMax = Math.max(capBytes, m.total) * 1.02

    return Plot.plot({
      width: 640, height: 28,
      marginLeft: 0, marginRight: 0, marginTop: 2, marginBottom: 2,
      x: {
        domain: [0, xMax],
        axis: null
      },
      y: { axis: null },
      color: {
        domain: Object.keys(COLORS),
        range: Object.values(COLORS),
        legend: false
      },
      marks: [
        Plot.barX(parts, {
          x: 'bytes', y: () => '',
          fill: 'component',
          tip: {
            format: {
              x: false, y: false, fill: false
            }
          },
          channels: {
            Component: { value: 'component', label: 'Component' },
            Size: { value: 'bytes', label: 'Size' }
          }
        }),
        // Format the tooltip's Size channel as human-readable GB.
        // (Plot's channels above use the raw byte value; the format
        // option below renders it with units.)
        Plot.ruleX([m.hbmCapacityGB * GB], {
          stroke: m.fits ? '#666' : '#c33',
          strokeWidth: 1.5,
          strokeDasharray: '4 3'
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

{#if $result}
  {@const m = $result.memory}
  {@const cap = m.hbmCapacityGB * GB}
  <section class="memory-panel">
    <h3>Memory budget — {gb(cap)} GB</h3>
    <div bind:this={container} class="bar-chart"></div>
    <div class="legend">
      {#each Object.entries(COLORS) as [name, color]}
        <span class="entry">
          <span class="swatch" style="background: {color}"></span>
          <span>{name}</span>
        </span>
      {/each}
      <span class="entry">
        <span class="capacity-line"></span>
        <span>Capacity</span>
      </span>
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
    max-width: 100%; overflow-x: auto;
    border: 1px solid #888; background: #f0f0f0;
  }
  .bar-chart :global(svg) { max-width: 100%; height: auto; display: block; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem;
    font-size: 0.85rem; color: #333;
  }
  .entry { display: inline-flex; align-items: center; gap: 0.35rem; }
  .swatch {
    width: 14px; height: 10px; border-radius: 2px;
    display: inline-block;
  }
  .capacity-line {
    width: 18px; height: 0; border-top: 1.5px dashed #666;
    display: inline-block;
  }
  table { font-variant-numeric: tabular-nums; }
  td:first-child { padding-right: 1rem; }
  tr.total { font-weight: bold; }
  .oom { color: #c33; font-weight: bold; }
  .caveat { font-size: 0.8rem; color: #666; font-style: italic; }
</style>

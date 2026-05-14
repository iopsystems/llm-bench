<script lang="ts">
  import * as Plot from '@observablehq/plot'
  import { result, multiDevice } from './stores'
  import { PLOT_STYLE } from './plotDefaults'

  let container: HTMLDivElement | undefined = $state(undefined)
  let containerWidth = $state(640)

  const GB = 1024 ** 3
  function gb(bytes: number): string { return (bytes / GB).toFixed(2) }

  // Weights is solid; KV cache and Activations use SVG patterns.
  type Component = 'Weights' | 'KV cache' | 'Activations'

  const PATTERN_IDS = {
    'KV cache': 'mem-pat-kv',
    Activations: 'mem-pat-acts'
  } as const

  // Fill resolution per component: either a solid color (Weights) or a
  // url(#pattern) reference (KV cache, Activations).
  const FILL: Record<Component, string> = {
    Weights: '#4682b4',
    'KV cache': `url(#${PATTERN_IDS['KV cache']})`,
    Activations: `url(#${PATTERN_IDS['Activations']})`
  }

  function patternDefsSvg(): string {
    // Inline SVG pattern defs spliced into Plot's generated <svg>.
    return `
      <pattern id="${PATTERN_IDS['KV cache']}" patternUnits="userSpaceOnUse"
               width="6" height="6">
        <rect width="6" height="6" fill="#dcd0f5"/>
        <circle cx="3" cy="3" r="1.2" fill="#7c5fc7"/>
      </pattern>
      <pattern id="${PATTERN_IDS['Activations']}" patternUnits="userSpaceOnUse"
               width="6" height="6" patternTransform="rotate(-45)">
        <rect width="6" height="6" fill="#c6e6e1"/>
        <line x1="0" y1="0" x2="0" y2="6" stroke="#1d8a7e" stroke-width="2"/>
      </pattern>
    `
  }

  const chart = $derived.by(() => {
    if (!$result) return null
    const m = $result.memory
    // Per-GPU capacity is always m.hbmCapacityGB (engine echoes variant capacity).
    const capBytes = m.hbmCapacityGB * GB
    // When perRank is present, show per-GPU segments against per-GPU HBM cap.
    const pr = m.perRank
    const raw: { component: Component; bytes: number }[] = [
      { component: 'Weights',     bytes: pr ? pr.weights           : m.weights },
      { component: 'KV cache',    bytes: pr ? pr.kvCachePerRequest : m.kvCacheTotal },
      { component: 'Activations', bytes: pr ? pr.activationsPeak   : m.activationsPeak }
    ]
    let cum = 0
    const parts = raw.map(p => {
      const x1 = cum
      cum += p.bytes
      return { ...p, x1, x2: cum }
    })

    return Plot.plot({
      width: containerWidth, height: 28,
      marginLeft: 0, marginRight: 0, marginTop: 0, marginBottom: 0,
      inset: 0, insetLeft: 0, insetRight: 0, insetTop: 0, insetBottom: 0,
      style: PLOT_STYLE,
      x: { domain: [0, capBytes], axis: null, insetLeft: 0, insetRight: 0 },
      y: { domain: [0, 1], axis: null, insetTop: 0, insetBottom: 0 },
      color: {
        domain: Object.keys(FILL),
        range: Object.values(FILL),
        legend: false
      },
      marks: [
        Plot.rect(parts, {
          x1: 'x1', x2: 'x2', y1: 0, y2: 1,
          fill: 'component', clip: true,
          insetLeft: 0, insetRight: 0, insetTop: 0, insetBottom: 0,
          tip: {
            format: { x1: false, x2: false, y1: false, y2: false, fill: false }
          },
          channels: {
            ' ': {
              value: (d: { component: string; bytes: number }) =>
                `${d.component}: ${gb(d.bytes)} GB`,
              label: ''
            }
          }
        })
      ]
    })
  })

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
      // Splice pattern <defs> into Plot's SVG so url(#...) fills resolve.
      const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs')
      defs.innerHTML = patternDefsSvg()
      chart.insertBefore(defs, chart.firstChild)
      container.appendChild(chart)
    }
  })
</script>

{#if $result}
  {@const m = $result.memory}
  {@const pr = m.perRank}
  {@const cap = m.hbmCapacityGB * GB}
  {@const barFits = pr ? pr.fits : m.fits}
  {@const barTotal = pr ? pr.total : m.total}
  {@const barHeadroom = pr ? pr.headroom : m.headroom}
  <section class="memory-panel">
    <h3>Memory budget — {gb(cap)} GB per GPU</h3>
    <div bind:this={container} class="bar-chart" class:oom={!barFits}></div>
    {#if pr && $multiDevice}
      <div class="cluster-total">
        {$multiDevice.system.accelerator.count} × {(pr.total / 1e9).toFixed(1)} GB per GPU
        = {(m.total / 1e9).toFixed(0)} GB cluster total
      </div>
    {/if}
    <table>
      <tbody>
        <tr>
          <td>
            <svg class="row-swatch" viewBox="0 0 14 10" aria-hidden="true">
              <defs>{@html patternDefsSvg()}</defs>
              <rect width="14" height="10" fill={FILL['Weights']}/>
            </svg>
            Weights
          </td>
          <td>{gb(pr ? pr.weights : m.weights)} GB</td>
        </tr>
        <tr>
          <td>
            <svg class="row-swatch" viewBox="0 0 14 10" aria-hidden="true">
              <defs>{@html patternDefsSvg()}</defs>
              <rect width="14" height="10" fill={FILL['KV cache']}/>
            </svg>
            KV cache {pr ? '(per request)' : '(total)'}
          </td>
          <td>{gb(pr ? pr.kvCachePerRequest : m.kvCacheTotal)} GB</td>
        </tr>
        <tr>
          <td>
            <svg class="row-swatch" viewBox="0 0 14 10" aria-hidden="true">
              <defs>{@html patternDefsSvg()}</defs>
              <rect width="14" height="10" fill={FILL['Activations']}/>
            </svg>
            Activations (~)
          </td>
          <td>{gb(pr ? pr.activationsPeak : m.activationsPeak)} GB</td>
        </tr>
        <tr class="total"><td>Total {pr ? 'per GPU' : ''}</td><td>{gb(barTotal)} GB</td></tr>
        <tr>
          <td>Headroom</td>
          <td class="headroom-value">
            {gb(barHeadroom)} GB
            <span class="status-badge" class:fits={barFits} class:oom={!barFits}>
              {barFits ? '✓ fits' : '✗ OOM'}
            </span>
          </td>
        </tr>
      </tbody>
    </table>
    <p class="caveat">~ activations estimate assumes FlashAttention-style kernels</p>
  </section>
{/if}

<style>
  .memory-panel { display: block; margin-top: 1rem; }
  .memory-panel > * + * { margin-top: 0.5rem; }
  .bar-chart {
    width: 100%; box-sizing: border-box;
    border: 2px solid #888; background: #f0f0f0;
  }
  .bar-chart.oom { border-color: #c33; }
  .bar-chart :global(svg) { display: block; overflow: visible; }
  .row-swatch {
    width: 14px; height: 10px; display: inline-block;
    margin-right: 0.5rem; vertical-align: middle;
  }
  table {
    font-variant-numeric: tabular-nums; border-collapse: collapse;
    margin: 0 auto;  /* center within block-layout panel */
  }
  td:first-child { padding-right: 2.5rem; }
  td:last-child { text-align: right; padding-left: 1rem; }
  /* Headroom row: number stays in column, status badge dangles outside the
     table's right edge via absolute positioning so it doesn't widen the table. */
  td.headroom-value { position: relative; }
  .status-badge {
    position: absolute;
    left: calc(100% + 0.5rem); top: 50%;
    transform: translateY(-50%);
    white-space: nowrap;
  }
  tr.total td { border-top: 1px solid #ccc; padding-top: 0.3rem; }
  tr.total { font-weight: bold; }
  /* Scope badge styling to .status-badge so the rules don't leak onto the
     .bar-chart.oom element (which shares the .oom class). That leakage was
     adding 0.15rem/0.5rem padding to the bar-chart and shrinking it. */
  .status-badge.fits {
    color: #1d6b45; background: #e6f5ec;
    padding: 0.15rem 0.5rem; border-radius: 0.2rem; font-weight: 600;
  }
  .status-badge.oom {
    color: #c33; background: #fde8e8;
    padding: 0.15rem 0.5rem; border-radius: 0.2rem; font-weight: 600;
  }
  .caveat { font-size: 0.8rem; color: #666; font-style: italic; }
  .cluster-total { font-size: 0.85rem; color: #555; margin-top: 0.3rem; }

  @media (max-width: 640px) {
    /* Tighter spacing and smaller font on narrow screens. The dangling
       status badge can wrap below the headroom number rather than
       overflow the table to the right. */
    td:first-child { padding-right: 1rem; }
    td:last-child { padding-left: 0.5rem; }
    .status-badge {
      position: static; transform: none;
      display: inline-block; margin-left: 0.4rem;
    }
  }
</style>

<script lang="ts">
  import { result } from './stores'

  const GB = 1024 ** 3
  function gb(bytes: number): string { return (bytes / GB).toFixed(2) }

  // Weights = solid steel blue. KV and Activations get textured fills using
  // CSS gradients (no SVG/Plot involved, no inset surprises, no ResizeObserver).
  type Component = 'Weights' | 'KV cache' | 'Activations'

  const STYLES: Record<Component, string> = {
    Weights:
      'background: #4682b4;',
    'KV cache':
      'background-color: #dcd0f5;' +
      'background-image: radial-gradient(circle, #7c5fc7 1.2px, transparent 1.5px);' +
      'background-size: 6px 6px;',
    Activations:
      'background: repeating-linear-gradient(-45deg, #c6e6e1 0 3px, #1d8a7e 3px 5px);'
  }

  function pct(part: number, whole: number): number {
    return Math.max(0, (part / whole) * 100)
  }
</script>

{#if $result}
  {@const m = $result.memory}
  {@const cap = m.hbmCapacityGB * GB}
  {@const parts = [
    { component: 'Weights' as Component,     bytes: m.weights },
    { component: 'KV cache' as Component,    bytes: m.kvCacheTotal },
    { component: 'Activations' as Component, bytes: m.activationsPeak }
  ]}
  <section class="memory-panel">
    <h3>Memory budget — {gb(cap)} GB</h3>
    <div class="bar-chart" class:oom={!m.fits}>
      {#each parts as p}
        <div
          class="seg"
          style="width: {pct(p.bytes, cap)}%; {STYLES[p.component]}"
          title="{p.component}: {gb(p.bytes)} GB"
        ></div>
      {/each}
    </div>
    <table>
      <tbody>
        <tr>
          <td>
            <span class="row-swatch" style={STYLES['Weights']}></span>
            Weights
          </td>
          <td>{gb(m.weights)} GB</td>
        </tr>
        <tr>
          <td>
            <span class="row-swatch" style={STYLES['KV cache']}></span>
            KV cache (total)
          </td>
          <td>{gb(m.kvCacheTotal)} GB</td>
        </tr>
        <tr>
          <td>
            <span class="row-swatch" style={STYLES['Activations']}></span>
            Activations (~)
          </td>
          <td>{gb(m.activationsPeak)} GB</td>
        </tr>
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
  /* HTML flex container — bars are sized by % width and stay anchored to the
     left edge. No SVG, no Plot, no measurements involved. Overflow on OOM
     gets clipped by overflow:hidden. */
  .bar-chart {
    width: 100%; height: 28px; box-sizing: border-box;
    display: flex; flex-direction: row;
    border: 2px solid #888; background: #f0f0f0;
    overflow: hidden;
  }
  .bar-chart.oom { border-color: #c33; }
  .seg { height: 100%; flex-shrink: 0; }

  .row-swatch {
    width: 14px; height: 10px; display: inline-block;
    margin-right: 0.5rem; vertical-align: middle;
    border: 1px solid rgba(0, 0, 0, 0.1);
  }
  table {
    font-variant-numeric: tabular-nums; border-collapse: collapse;
    align-self: center;
  }
  td:first-child { padding-right: 2.5rem; }
  td:last-child { text-align: right; padding-left: 1rem; }
  td.headroom-value { position: relative; }
  .status-badge {
    position: absolute;
    left: calc(100% + 0.5rem); top: 50%;
    transform: translateY(-50%);
    white-space: nowrap;
  }
  tr.total td { border-top: 1px solid #ccc; padding-top: 0.3rem; }
  tr.total { font-weight: bold; }
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

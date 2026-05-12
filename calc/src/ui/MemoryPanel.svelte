<script lang="ts">
  import { result } from './stores'

  const GB = 1024 ** 3
  function gb(bytes: number): string { return (bytes / GB).toFixed(2) }
  function pct(part: number, whole: number): number {
    return Math.max(0, Math.min(100, (part / whole) * 100))
  }
</script>

{#if $result}
  {@const m = $result.memory}
  {@const cap = m.hbmCapacityGB * GB}
  <section class="memory-panel">
    <h3>Memory budget — {gb(cap)} GB</h3>
    <div class="bar" class:oom={!m.fits}>
      <div class="seg weights" style="width: {pct(m.weights, cap)}%"></div>
      <div class="seg kv" style="width: {pct(m.kvCacheTotal, cap)}%"></div>
      <div class="seg act" style="width: {pct(m.activationsPeak, cap)}%"></div>
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
  .bar { display: flex; height: 1.5rem; border: 1px solid #888; background: #f0f0f0; }
  .bar.oom { border-color: #c33; }
  .seg.weights { background: #4a90e2; }
  .seg.kv      { background: #7ac74a; }
  .seg.act     { background: #e2a04a; }
  table { font-variant-numeric: tabular-nums; }
  td:first-child { padding-right: 1rem; }
  tr.total { font-weight: bold; }
  .oom { color: #c33; font-weight: bold; }
  .caveat { font-size: 0.8rem; color: #666; font-style: italic; }
</style>

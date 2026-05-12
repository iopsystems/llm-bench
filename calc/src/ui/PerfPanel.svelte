<script lang="ts">
  import { result } from './stores'

  function ms(s: number): string { return (s * 1000).toFixed(2) + ' ms' }
  function rate(tps: number): string { return tps.toFixed(1) + ' tok/s' }
</script>

{#if $result}
  <section class="perf-panel">
    <h3>Performance</h3>
    <table>
      <thead>
        <tr>
          <th>Operating point</th>
          <th>TTFT</th>
          <th>Prefill regime</th>
          <th>Decode time / tok</th>
          <th>Decode regime</th>
          <th>Input tok/s</th>
          <th>Output tok/s (aggregate)</th>
        </tr>
      </thead>
      <tbody>
        {#each Object.entries($result.perf) as [id, p]}
          <tr>
            <td>{id}</td>
            <td>{ms(p.ttftS)}</td>
            <td><span class="regime {p.prefill.regime}">{p.prefill.regime}</span></td>
            <td>{ms(p.decode.timePerTokenS)}</td>
            <td><span class="regime {p.decode.regime}">{p.decode.regime}</span></td>
            <td>{rate(p.inputTokenRate)}</td>
            <td>{rate(p.outputTokenRate)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </section>
{/if}

<style>
  .perf-panel { margin-top: 1rem; }
  table { font-variant-numeric: tabular-nums; border-collapse: collapse; }
  th, td { padding: 0.25rem 0.75rem; text-align: left; border-bottom: 1px solid #eee; }
  .regime { padding: 0.1rem 0.4rem; border-radius: 0.2rem; font-size: 0.85rem; }
  .regime.compute { background: #fde6c8; color: #8a4400; }
  .regime.memory  { background: #c8dcfd; color: #003a8c; }
</style>

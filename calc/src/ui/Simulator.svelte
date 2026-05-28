<script lang="ts">
  import InputPanel from './InputPanel.svelte'
  import SimulatorGantt from './SimulatorGantt.svelte'
  import { simResult, simError, workload, disaggFirstTokenOnPrefill } from './stores'
  import type { GanttInput } from './simulatorGantt'

  // Same formatting helpers as PerfPanel; copied here to keep this file
  // self-contained for v1 (extract into a shared module when a third view
  // wants them).
  function sig3(n: number): string {
    if (n === 0) return '0'
    return parseFloat(n.toPrecision(3)).toString()
  }
  function ms(s: number): string {
    if (s >= 1)     return `${sig3(s)} s`
    if (s >= 1e-3)  return `${sig3(s * 1e3)} ms`
    if (s >= 1e-6)  return `${sig3(s * 1e6)} µs`
    return `${sig3(s * 1e9)} ns`
  }

  // The simulator follows the same op-point the calc tab is showing. Since
  // op-point isn't currently URL state, "the same" reduces to "show every
  // op-point" — pick the first key in perf for the cards/gantt and let
  // additional op-points appear below as a small comparison list.
  $: opIds = $simResult ? Object.keys($simResult.perf) : []
  $: primary = opIds[0]
  $: tier = $simResult && primary ? $simResult.perf[primary] : null

  $: ganttInput = tier ? ({
    prefillS: tier.prefill.timeS,
    kvTransferS: tier.kvTransferS,
    tpotS: tier.decode.timePerTokenS,
    outputTokens: $workload.outputTokens,
    firstTokenOnPrefill: $disaggFirstTokenOnPrefill,
    ttftS: tier.ttftS,
    prefillRegime: tier.prefill.regime,
    decodeRegime: tier.decode.regime,
  } satisfies GanttInput) : null

  $: totalS = tier ? tier.ttftS + tier.decode.timePerTokenS * ($workload.outputTokens - 1) : 0
</script>

<section class="simulator">
  <InputPanel hideConcurrency={true} />

  {#if $simError}
    <div class="error">⚠ {$simError}</div>
  {:else if tier && ganttInput}
    <h3 class="config-header">Single request, monolithic</h3>
    <div class="kpis">
      <div class="kpi">
        <div class="label">TTFT</div>
        <div class="value">{ms(tier.ttftS)}</div>
        <div class="badge regime-{tier.prefill.regime}">{tier.prefill.regime}-bound prefill</div>
      </div>
      <div class="kpi">
        <div class="label">TPOT</div>
        <div class="value">{ms(tier.decode.timePerTokenS)}</div>
        <div class="badge regime-{tier.decode.regime}">{tier.decode.regime}-bound decode</div>
        <div class="caption">{sig3(1 / tier.decode.timePerTokenS)} tok/s</div>
      </div>
      <div class="kpi">
        <div class="label">Total latency</div>
        <div class="value">{ms(totalS)}</div>
        <div class="caption">{$workload.outputTokens} output tokens</div>
      </div>
    </div>

    <div class="gantt-wrap">
      <h4>Timeline ({primary})</h4>
      <SimulatorGantt input={ganttInput} />
    </div>

    {#if opIds.length > 1}
      <details class="other-ops">
        <summary>Other operating points</summary>
        <table>
          <thead><tr><th>Op</th><th>TTFT</th><th>TPOT</th><th>Total</th></tr></thead>
          <tbody>
            {#each opIds.slice(1) as id}
              {@const t = $simResult!.perf[id]}
              <tr>
                <td>{id}</td>
                <td>{ms(t.ttftS)}</td>
                <td>{ms(t.decode.timePerTokenS)}</td>
                <td>{ms(t.ttftS + t.decode.timePerTokenS * ($workload.outputTokens - 1))}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </details>
    {/if}
  {/if}
</section>

<style>
  .simulator { display: flex; flex-direction: column; gap: 1rem; }
  .error {
    padding: 0.5rem 0.75rem;
    background: #fde6e6; color: #8a1f1f;
    border: 1px solid #f0b0b0; border-radius: 0.25rem;
    font-size: 0.9rem;
  }
  .config-header {
    margin: 0.5rem 0 -0.25rem; font-size: 1rem; font-weight: 600; color: #333;
  }
  .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
  .kpi {
    border: 1px solid #d4d4d4; border-radius: 0.4rem; padding: 0.8rem 1rem;
    background: #fff;
  }
  .kpi .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #888; }
  .kpi .value { font-size: 1.75rem; font-weight: 700; line-height: 1.1; margin-top: 0.2rem; }
  .kpi .badge {
    display: inline-block; margin-top: 0.4rem; padding: 0.1rem 0.45rem;
    font-size: 0.75rem; border-radius: 0.2rem; color: #fff;
  }
  .badge.regime-compute { background: #2b6cb0; }
  .badge.regime-memory  { background: #c05621; }
  .badge.regime-comms   { background: #6b46c1; }
  .kpi .caption { font-size: 0.78rem; color: #666; margin-top: 0.3rem; }
  .gantt-wrap h4 { margin: 0 0 0.4rem; font-size: 0.85rem; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .other-ops { font-size: 0.85rem; }
  .other-ops table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  .other-ops th, .other-ops td { text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid #eee; }
  @media (max-width: 640px) {
    .kpis { grid-template-columns: 1fr; }
  }
</style>

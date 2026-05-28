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
  function rate(tps: number): string {
    if (tps >= 1e9) return `${sig3(tps / 1e9)} G tok/s`
    if (tps >= 1e6) return `${sig3(tps / 1e6)} M tok/s`
    if (tps >= 1e3) return `${sig3(tps / 1e3)} k tok/s`
    return `${sig3(tps)} tok/s`
  }

  // Gate results on memory fit: per-rank when parallelism shards the model
  // across devices, top-level otherwise. The calc tab shows perf anyway
  // ("what would it look like if it fit"); the simulator's framing is
  // user-experience-of-this-request, which is moot if the model can't load.
  $: memory = $simResult?.memory
  $: fits = memory ? (memory.perRank?.fits ?? memory.fits) : false

  // Build a row per operating point and sort by Total latency ascending
  // (fast → slow, i.e. peak → achievable when both are present). Each row
  // carries everything the cards + gantt need.
  interface OpRow {
    id: string
    ttftS: number
    tpotS: number
    totalS: number
    inputTokenRate: number
    prefillRegime: 'compute' | 'memory' | 'comms'
    decodeRegime: 'compute' | 'memory' | 'comms'
    gantt: GanttInput
  }
  $: rows = ($simResult ? Object.entries($simResult.perf).map(([id, t]): OpRow => ({
    id,
    ttftS: t.ttftS,
    tpotS: t.decode.timePerTokenS,
    totalS: t.ttftS + t.decode.timePerTokenS * ($workload.outputTokens - 1),
    inputTokenRate: t.inputTokenRate,
    prefillRegime: t.prefill.regime,
    decodeRegime: t.decode.regime,
    gantt: {
      prefillS: t.prefill.timeS,
      kvTransferS: t.kvTransferS,
      tpotS: t.decode.timePerTokenS,
      outputTokens: $workload.outputTokens,
      firstTokenOnPrefill: $disaggFirstTokenOnPrefill,
      ttftS: t.ttftS,
      prefillRegime: t.prefill.regime,
      decodeRegime: t.decode.regime,
    },
  })) : []).sort((a, b) => a.totalS - b.totalS)
</script>

<section class="simulator">
  <InputPanel hideConcurrency={true} />

  {#if $simError}
    <div class="error">⚠ {$simError}</div>
  {:else if memory && !fits}
    <div class="oom">
      <strong>✗ Out of memory.</strong>
      Model + KV cache + activations exceed HBM capacity on the selected
      configuration. Pick a larger SKU, add parallelism (TP/PP), or trim the
      workload (prompt/output tokens). See the Calculator tab's Memory panel
      for the breakdown.
    </div>
  {:else if rows.length > 0}
    <h3 class="config-header">Single request, monolithic</h3>
    <div class="kpis">
      <div class="kpi">
        <div class="label">TTFT</div>
        {#each rows as row, i}
          <div class="op" class:secondary={i > 0}>
            {#if rows.length > 1}<div class="op-name">{row.id}</div>{/if}
            <div class="value">{ms(row.ttftS)}</div>
            <div class="badge regime-{row.prefillRegime}">{row.prefillRegime}-bound prefill</div>
            <div class="caption">{rate(row.inputTokenRate)} input</div>
          </div>
        {/each}
      </div>
      <div class="kpi">
        <div class="label">TPOT</div>
        {#each rows as row, i}
          <div class="op" class:secondary={i > 0}>
            {#if rows.length > 1}<div class="op-name">{row.id}</div>{/if}
            <div class="value">{ms(row.tpotS)}</div>
            <div class="badge regime-{row.decodeRegime}">{row.decodeRegime}-bound decode</div>
            <div class="caption">{rate(1 / row.tpotS)} output</div>
          </div>
        {/each}
      </div>
      <div class="kpi">
        <div class="label">Total latency</div>
        {#each rows as row, i}
          <div class="op" class:secondary={i > 0}>
            {#if rows.length > 1}<div class="op-name">{row.id}</div>{/if}
            <div class="value">{ms(row.totalS)}</div>
            <div class="caption">{$workload.outputTokens} output tokens</div>
          </div>
        {/each}
      </div>
    </div>

    {#each rows as row}
      <div class="gantt-wrap">
        <h4>Timeline{rows.length > 1 ? ` (${row.id})` : ''}</h4>
        <SimulatorGantt input={row.gantt} />
      </div>
    {/each}
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
  .oom {
    padding: 0.7rem 0.9rem;
    background: #fff7ec; color: #8a3f00;
    border: 1px solid #f0c890; border-radius: 0.3rem;
    font-size: 0.9rem; line-height: 1.4;
  }
  .oom strong { color: #b85b00; margin-right: 0.25rem; }
  .config-header {
    margin: 0.5rem 0 -0.25rem; font-size: 1rem; font-weight: 600; color: #333;
  }
  .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; align-items: start; }
  .kpi {
    border: 1px solid #d4d4d4; border-radius: 0.4rem; padding: 0.8rem 1rem;
    background: #fff;
  }
  .kpi .label { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.02em; color: #888; }
  /* Each op-point sub-block inside a card. The secondary modifier separates
     subsequent rows with a hairline so the eye can scan top-down within one card. */
  .op { padding-top: 0.2rem; }
  .op.secondary {
    margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px solid #eee;
  }
  .op-name {
    font-size: 0.85rem; font-style: italic; color: #555;
    margin-bottom: 0.1rem;
  }
  .op .value { font-size: 1.5rem; font-weight: 700; line-height: 1.1; margin-top: 0.1rem; }
  .op .badge {
    display: inline-block; margin-top: 0.35rem; padding: 0.1rem 0.45rem;
    font-size: 0.75rem; border-radius: 0.2rem; color: #fff;
  }
  /* Regime palette matches the Calculator tab: compute=warm/orange,
     memory=cool/blue. (Calc uses pastel-on-dark-text; here we use saturated
     fills + white text for the badge form, but the warm/cool mapping is the
     same so users don't see compute and memory swap colors across tabs.) */
  .badge.regime-compute { background: #c05621; }
  .badge.regime-memory  { background: #2b6cb0; }
  .badge.regime-comms   { background: #6b46c1; }
  .op .caption { font-size: 0.78rem; color: #666; margin-top: 0.3rem; }
  .gantt-wrap h4 { margin: 0 0 0.4rem; font-size: 0.85rem; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  @media (max-width: 640px) {
    .kpis { grid-template-columns: 1fr; }
  }
</style>

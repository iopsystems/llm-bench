<script lang="ts">
  import InputPanel from './InputPanel.svelte'
  import DisaggInputPanel from './DisaggInputPanel.svelte'
  import SimulatorGantt from './SimulatorGantt.svelte'
  import {
    simResultMonolithic, simResultDisagg, simError, simErrorDisagg,
    workload, disaggFirstTokenOnPrefill, disaggKvTransferFabricId
  } from './stores'
  import type { GanttInput } from './simulatorGantt'
  import type { CalcResult } from '../engine/types'

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

  // Monolithic block: today's combined-total check (mirrors prefill side).
  $: monolithicMemory = $simResultMonolithic?.memory
  $: monolithicFits = monolithicMemory
    ? (monolithicMemory.perRank?.fits ?? monolithicMemory.fits)
    : false

  // Disagg block: two-sided per-cluster check. Heterogeneous SKUs mean each
  // side has its own capacity, so we report which side(s) bust.
  $: disaggMemory = $simResultDisagg?.memory
  $: disaggPrefillFits = disaggMemory
    ? (disaggMemory.prefillSide.perRank?.fits ?? disaggMemory.prefillSide.fits)
    : true
  $: disaggDecodeFits = disaggMemory
    ? (disaggMemory.decodeSide.perRank?.fits ?? disaggMemory.decodeSide.fits)
    : true
  $: disaggFits = disaggPrefillFits && disaggDecodeFits
  $: disaggFailingSides =
    !disaggPrefillFits && !disaggDecodeFits ? 'both' :
    !disaggPrefillFits ? 'prefill' :
    !disaggDecodeFits  ? 'decode'  : null

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

  function rowsFrom(result: CalcResult | null, firstTokenOnPrefill: boolean, outputTokens: number): OpRow[] {
    if (!result) return []
    return Object.entries(result.perf).map(([id, t]): OpRow => {
      // Stutter (case B disagg-overlap with slow fabric): kvTransferS > tpotS
      // means token #2 waits for KV to arrive after token #1 is emitted, so
      // total latency extends by (kvTransferS - tpotS). Keeps the KPI card's
      // Total in sync with the gantt's timeline end (see simulatorGantt.ts).
      const isOverlap = t.kvTransferS > 0 && firstTokenOnPrefill
      const stutterS = isOverlap ? Math.max(0, t.kvTransferS - t.decode.timePerTokenS) : 0
      return {
        id,
        ttftS: t.ttftS,
        tpotS: t.decode.timePerTokenS,
        totalS: t.ttftS + t.decode.timePerTokenS * (outputTokens - 1) + stutterS,
        inputTokenRate: t.inputTokenRate,
        prefillRegime: t.prefill.regime,
        decodeRegime: t.decode.regime,
        gantt: {
          prefillS: t.prefill.timeS,
          kvTransferS: t.kvTransferS,
          tpotS: t.decode.timePerTokenS,
          outputTokens,
          firstTokenOnPrefill,
          ttftS: t.ttftS,
          prefillRegime: t.prefill.regime,
          decodeRegime: t.decode.regime,
        },
      }
    }).sort((a, b) => a.totalS - b.totalS)
  }

  $: rowsMonolithic = rowsFrom($simResultMonolithic, $disaggFirstTokenOnPrefill, $workload.outputTokens)
  $: rowsDisagg     = rowsFrom($simResultDisagg,     $disaggFirstTokenOnPrefill, $workload.outputTokens)
</script>

{#snippet resultBlock(rows: OpRow[])}
  <div class="kpis" style:--row-count={rows.length}>
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
    <div class="kpi">
      <div class="label">Throughput</div>
      {#each rows as row, i}
        <div class="op" class:secondary={i > 0}>
          {#if rows.length > 1}<div class="op-name">{row.id}</div>{/if}
          <div class="tp-row"><span class="tp-label">Input</span><span class="tp-value">{rate(row.inputTokenRate)}</span></div>
          <div class="tp-row"><span class="tp-label">Output</span><span class="tp-value">{rate(1 / row.tpotS)}</span></div>
          <div class="tp-row"><span class="tp-label">Req</span><span class="tp-value">{sig3(1 / row.totalS)} req/s</span></div>
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
{/snippet}

<section class="simulator">
  <InputPanel hideConcurrency={true} hideDisagg={true} />

  {#if $simError}
    <div class="error">⚠ {$simError}</div>
  {:else if monolithicMemory && !monolithicFits}
    <div class="oom">
      <strong>✗ Out of memory.</strong>
      Model + KV cache + activations exceed HBM capacity on the selected
      configuration. Pick a larger SKU, add parallelism (TP/PP), or trim the
      workload (prompt/output tokens). See the Calculator tab's Memory panel
      for the breakdown.
    </div>
  {:else if rowsMonolithic.length > 0}
    <h3 class="config-header">Single request, monolithic</h3>
    {@render resultBlock(rowsMonolithic)}

    {#if $disaggKvTransferFabricId}
      <h3 class="config-header">Single request, PD-disagg</h3>
      <DisaggInputPanel />
      {#if $simErrorDisagg}
        <div class="error">⚠ {$simErrorDisagg}</div>
      {:else if disaggMemory && !disaggFits}
        <div class="oom">
          <strong>✗ Out of memory on {disaggFailingSides} cluster{disaggFailingSides === 'both' ? 's' : ''}.</strong>
          {#if !disaggPrefillFits}
            Prefill side: weights + prefill activations exceed HBM. Try a larger prefill SKU
            or trim promptTokens (prefill activations scale with prompt × hidden).
          {/if}
          {#if !disaggDecodeFits}
            Decode side: weights + KV cache exceed HBM. Try a larger decode SKU,
            add parallelism on the decode cluster, or reduce maxContext-bound KV growth.
          {/if}
        </div>
      {:else if rowsDisagg.length > 0}
        {@render resultBlock(rowsDisagg)}
      {/if}
    {:else if !$disaggKvTransferFabricId}
      <!-- Inline affordance for enabling disagg without going up to the
           shared inputs above. Lives in its own placeholder block. -->
      <div class="disagg-empty">
        <DisaggInputPanel />
        <p>Pick a KV transfer fabric above to add a PD-disagg comparison block.</p>
      </div>
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
  .disagg-empty p {
    margin: 0.25rem 0 0; font-size: 0.85rem; color: #666; font-style: italic;
  }
  /* Grid rows: one for the label, one per op-point. Each .kpi uses subgrid
     so its children land on the same row tracks across all 4 cards — keeps
     the .op.secondary border-tops (op-point dividers) horizontally aligned. */
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-template-rows: auto repeat(var(--row-count, 1), auto);
    gap: 0.75rem;
  }
  .kpi {
    display: grid;
    grid-template-rows: subgrid;
    grid-row: span calc(var(--row-count, 1) + 1);
    border: 1px solid #d4d4d4; border-radius: 0.4rem; padding: 0.8rem 1rem;
    background: #fff;
  }
  .kpi .label { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.02em; color: #888; }
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
  .badge.regime-compute { background: #c05621; }
  .badge.regime-memory  { background: #2b6cb0; }
  .badge.regime-comms   { background: #6b46c1; }
  .op .caption { font-size: 0.78rem; color: #666; margin-top: 0.3rem; }
  /* Throughput card rows: label/value pairs, two-column-ish alignment.
     Smaller font than the headline values of the latency cards so the
     three stacked metrics don't visually overpower the rest of the row. */
  .tp-row {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-top: 0.25rem;
  }
  .tp-label { font-size: 0.85rem; color: #666; }
  .tp-value { font-size: 0.95rem; font-weight: 700; color: #222; }
  .gantt-wrap h4 { margin: 0 0 0.4rem; font-size: 0.85rem; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  @media (max-width: 900px) {
    .kpis { grid-template-columns: repeat(2, 1fr); }
  }
  @media (max-width: 640px) {
    .kpis { grid-template-columns: 1fr; }
  }
</style>

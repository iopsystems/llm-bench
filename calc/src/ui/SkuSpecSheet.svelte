<!-- calc/src/ui/SkuSpecSheet.svelte -->
<script lang="ts">
  import type { AcceleratorSpec, MultiAcceleratorSystem } from '../engine/types'
  import { skuMetrics } from './catalogMetrics'
  export let sku: AcceleratorSpec | MultiAcceleratorSystem
  $: isSystem = 'aggregate' in sku
  $: metrics = skuMetrics(sku)

  // Pivot the flat peakTable into dtype-rows × variant-columns.
  const DTYPE_ORDER = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4']
  $: pivotVariants = metrics.kind === 'accelerator'
    ? metrics.peakTable.reduce<{ id: string; label: string; hbmCapacityGB: number }[]>(
        (acc, r) => acc.some(v => v.id === r.variantId)
          ? acc
          : [...acc, { id: r.variantId, label: r.variantLabel, hbmCapacityGB: r.hbmCapacityGB }],
        [])
    : []
  $: pivotDtypes = metrics.kind === 'accelerator'
    ? DTYPE_ORDER.filter(dt => metrics.peakTable.some(r => r.dtype === dt))
    : []
  function cell(dt: string, variantId: string) {
    if (metrics.kind !== 'accelerator') return undefined
    return metrics.peakTable.find(r => r.dtype === dt && r.variantId === variantId)
  }
</script>

<article class="sheet">
  <h2><slot />{sku.name}</h2>
  <div class="rule-thick"></div>
  <dl>
    <dt>Vendor</dt><dd>{sku.vendor}</dd>
    {#if 'family' in sku && sku.family}<dt>Family</dt><dd>{sku.family}</dd>{/if}
    <dt>Released</dt><dd>{sku.releaseDate}</dd>
  </dl>

  {#if isSystem && metrics.kind === 'system'}
    {@const s = sku as MultiAcceleratorSystem}
    <div class="rule"></div>
    <h3>System composition</h3>
    <dl>
      <dt>Accelerators</dt><dd>{s.accelerator.count}× {s.accelerator.id} ({s.accelerator.variantId})</dd>
      <dt>Interconnect</dt><dd>{s.interconnectId}</dd>
      <dt>Form factor</dt><dd>{s.formFactor}</dd>
      <dt>Total HBM</dt><dd>{metrics.totalHbmGB} GB</dd>
      <dt>Fabric (bidir)</dt><dd>{metrics.fabricBidirectionalTBs} TB/s</dd>
    </dl>
    {#if s.availability?.onPrem || s.availability?.clouds?.length}
      <div class="rule"></div>
      <h3>Availability</h3>
      <dl>
        <dt>On-prem</dt><dd>{s.availability?.onPrem ? 'Yes' : '—'}</dd>
        {#if s.availability?.clouds?.length}
          <dt>Clouds</dt><dd>{s.availability.clouds.join(', ')}</dd>
        {/if}
      </dl>
    {/if}
  {:else if metrics.kind === 'accelerator'}
    <div class="rule"></div>
    <h3>Peak arithmetic</h3>
    <table>
      <thead>
        <tr>
          <th rowspan="2">dtype</th>
          {#each pivotVariants as v}
            <th colspan="2" class="vhead">{v.label}</th>
          {/each}
        </tr>
        <tr>
          {#each pivotVariants as _v}
            <th class="num sub2">TFLOPS</th><th class="num sub2">FLOP/byte</th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each pivotDtypes as dt}
          <tr>
            <td>{dt}</td>
            {#each pivotVariants as v}
              {@const c = cell(dt, v.id)}
              <td class="num">{c ? c.tflops.toLocaleString() : '—'}</td>
              <td class="num">{c ? c.ridge.toFixed(0) : '—'}</td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>

    <div class="rule"></div>
    <h3>dtype support</h3>
    <table>
      <thead>
        <tr><th>dtype</th><th>support</th><th>throughput implication</th></tr>
      </thead>
      <tbody>
        {#each metrics.dtypeSupport as d}
          <tr>
            <td>{d.dtype}</td>
            <td><span class="sup {d.support}">{d.support}</span></td>
            <td class="impl">{d.note}</td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#if metrics.variants.some(v => v.efficiencyByDtype || v.operatingPoints.some(o => o.asOf || o.notes || o.sources))}
      <div class="rule"></div>
      <h3>Measured / provenance</h3>
      {#each metrics.variants as v}
        {#if v.efficiencyByDtype || v.operatingPoints.some(o => o.asOf || o.notes || o.sources)}
          <p class="vnote"><strong>{v.label}</strong></p>
          {#if v.efficiencyByDtype}
            <p class="sub">Achievable vs peak:
              {#each Object.entries(v.efficiencyByDtype) as [dt, e]}
                <span class="chip">{dt} {(e! * 100).toFixed(0)}%</span>
              {/each}
            </p>
          {/if}
          {#each v.operatingPoints as op}
            {#if op.asOf || op.notes || op.sources}
              <p class="sub">{op.label}: {[op.asOf, op.sources?.join(', '), op.notes].filter(Boolean).join(' · ')}</p>
            {/if}
          {/each}
        {/if}
      {/each}
    {/if}
  {/if}
</article>

<style>
  /* Nutrition-label aesthetic: hard black frame, heavy title rule, thinner
     section rules, dense tabular body. */
  .sheet {
    max-width: 720px; border: 2px solid #111; border-radius: 4px;
    padding: 0.9rem 1.1rem; background: #fff;
  }
  h2 {
    margin: 0 0 0.4rem; font-size: 1.25rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  h3 {
    margin: 0.6rem 0 0.4rem; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #333;
  }
  .rule-thick { border-bottom: 6px solid #111; margin: 0.3rem 0 0.6rem; }
  .rule { border-bottom: 1px solid #111; margin: 0.7rem 0 0; }
  .ref { font-weight: 400; color: #888; font-size: 0.8rem; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; margin: 0; }
  dt { color: #555; }
  dd { margin: 0; font-variant-numeric: tabular-nums; }
  table { border-collapse: collapse; font-size: 0.85rem; width: 100%; }
  th, td {
    text-align: left; padding: 0.22rem 0.6rem;
    border-bottom: 1px solid #e2e2e2;
  }
  th { border-bottom: 1px solid #111; font-size: 0.78rem; }
  .vhead { text-align: center; border-left: 1px solid #e2e2e2; }
  .sub2 { font-weight: 400; color: #666; border-bottom: 1px solid #111; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .chip { display: inline-block; margin-right: 0.5rem; color: #444; }
  .vnote { margin: 0.5rem 0 0.15rem; font-size: 0.85rem; }
  .sub { font-size: 0.75rem; color: #777; margin: 0.1rem 0 0; }
  .impl { color: #555; }
  .sup {
    display: inline-block; font-size: 0.72rem; padding: 0.05rem 0.4rem;
    border-radius: 0.25rem; text-transform: uppercase; letter-spacing: 0.03em;
  }
  .sup.native { background: #d8f0e4; color: #1b6b4a; }
  .sup.conversion { background: #fbeccd; color: #8a5a12; }
  .sup.software { background: #ececec; color: #777; }
</style>

<!-- calc/src/ui/SkuSpecSheet.svelte -->
<script lang="ts">
  import type { AcceleratorSpec, MultiAcceleratorSystem } from '../engine/types'
  import { skuMetrics } from './catalogMetrics'
  export let sku: AcceleratorSpec | MultiAcceleratorSystem
  $: isSystem = 'aggregate' in sku
  $: metrics = skuMetrics(sku)
</script>

<article class="sheet">
  <h2>{sku.name}</h2>
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
      {#if s.availability?.clouds?.length}
        <dt>Clouds</dt><dd>{s.availability.clouds.join(', ')}</dd>
      {/if}
    </dl>
  {:else if metrics.kind === 'accelerator'}
    <div class="rule"></div>
    <h3>Peak arithmetic</h3>
    <table>
      <thead>
        <tr><th>Variant</th><th>dtype</th><th class="num">peak TFLOPS</th><th class="num">peak FLOP/byte</th></tr>
      </thead>
      <tbody>
        {#each metrics.peakTable as row}
          <tr>
            <td>{row.variantLabel} <span class="ref">{row.hbmCapacityGB} GB</span></td>
            <td>{row.dtype}</td>
            <td class="num">{row.tflops.toLocaleString()}</td>
            <td class="num">{row.ridge.toFixed(0)}</td>
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
  h2 { margin: 0 0 0.4rem; font-size: 1.25rem; }
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
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .chip { display: inline-block; margin-right: 0.5rem; color: #444; }
  .vnote { margin: 0.5rem 0 0.15rem; font-size: 0.85rem; }
  .sub { font-size: 0.75rem; color: #777; margin: 0.1rem 0 0; }
</style>

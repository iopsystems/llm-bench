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
  <dl>
    <dt>Vendor</dt><dd>{sku.vendor}</dd>
    {#if 'family' in sku && sku.family}<dt>Family</dt><dd>{sku.family}</dd>{/if}
    <dt>Released</dt><dd>{sku.releaseDate}</dd>
  </dl>

  {#if isSystem && metrics.kind === 'system'}
    {@const s = sku as MultiAcceleratorSystem}
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
    {#each metrics.variants as v}
      <h3>{v.label} <span class="ref">({v.hbmCapacityGB} GB HBM)</span></h3>
      <table>
        <thead><tr><th>Operating point</th><th>Ridge (FLOP/byte) by dtype</th></tr></thead>
        <tbody>
          {#each v.operatingPoints as op}
            <tr>
              <td>{op.label}</td>
              <td>
                {#each Object.entries(op.ridgeByDtype) as [dt, r]}
                  <span class="chip">{dt}: {r!.toFixed(0)}</span>
                {/each}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/each}
  {/if}
</article>

<style>
  .sheet { max-width: 720px; }
  h2 { margin: 0 0 0.75rem; }
  h3 { margin: 1.25rem 0 0.4rem; font-size: 0.95rem; }
  .ref { font-weight: 400; color: #888; font-size: 0.8rem; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1rem; margin: 0; }
  dt { color: #666; }
  dd { margin: 0; font-variant-numeric: tabular-nums; }
  table { border-collapse: collapse; font-size: 0.85rem; margin-top: 0.25rem; }
  th, td { text-align: left; padding: 0.25rem 0.6rem; border-bottom: 1px solid #eee; }
  .chip { display: inline-block; margin-right: 0.5rem; color: #444; }
</style>

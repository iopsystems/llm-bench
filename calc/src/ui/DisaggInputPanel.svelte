<script lang="ts">
  import {
    acceleratorId, variantId, systemId,
    disaggKvTransferFabricId, disaggFirstTokenOnPrefill,
    heterogeneous, decodeAcceleratorId, decodeVariantId, decodeSystemId,
  } from './stores'
  import { groupedDisaggFabrics, formatFabricLabel } from './disaggFabrics'
  import { ACCELERATORS } from '../data'
  import { SYSTEMS } from '../data/systems'
  import { orderSkus } from './catalogOrder'
  import ParallelismPicker from './ParallelismPicker.svelte'

  // V2: when $heterogeneous is on, the decode-side combo dropdown lets the
  // user pick a different accelerator/variant/system for the decode cluster.
  // ParallelismPicker (side='decode') hosts the decode parallelism selector.
  $: groups = groupedDisaggFabrics($acceleratorId)
  $: skuGroups = orderSkus(ACCELERATORS, SYSTEMS)

  $: decodeComboValue = $decodeSystemId
    ? `sys:${$decodeSystemId}`
    : `chip:${$decodeAcceleratorId || $acceleratorId}`

  $: decodeAcceleratorObj = ACCELERATORS.find(a => a.id === ($decodeAcceleratorId || $acceleratorId))
  $: decodeVariants = decodeAcceleratorObj?.variants ?? []

  function onDecodeComboChange(e: Event) {
    const v = (e.target as HTMLSelectElement).value
    if (v.startsWith('sys:')) {
      decodeSystemId.set(v.slice(4))
    } else {
      decodeSystemId.set('')
      decodeAcceleratorId.set(v.slice(5))
    }
  }

  // Pre-populate decode-side stores from prefill on first toggle-on, so the
  // user starts symmetric and transitions by changing one knob.
  function onHetToggle(e: Event) {
    const on = (e.target as HTMLInputElement).checked
    heterogeneous.set(on)
    if (on && !$decodeAcceleratorId && !$decodeSystemId) {
      decodeAcceleratorId.set($acceleratorId)
      decodeVariantId.set($variantId)
      decodeSystemId.set($systemId)
    }
  }
</script>

<div class="disagg-inputs">
  <label>
    KV transfer fabric
    <select bind:value={$disaggKvTransferFabricId}>
      <option value="">— off (monolithic only) —</option>
      {#if groups.scaleUp.length > 0}
        <optgroup label="Intra-domain (scale-up)">
          {#each groups.scaleUp as f}
            <option value={f.id}>{formatFabricLabel(f)}</option>
          {/each}
        </optgroup>
      {/if}
      <optgroup label="Cross-rack (scale-out)">
        {#each groups.scaleOut as f}
          <option value={f.id}>{formatFabricLabel(f)}</option>
        {/each}
      </optgroup>
    </select>
  </label>
  {#if $disaggKvTransferFabricId}
    <label class="inline">
      <input type="checkbox" bind:checked={$disaggFirstTokenOnPrefill} />
      <span>1st token on prefill (hide transfer in TTFT)</span>
    </label>
    <label class="inline">
      <input type="checkbox" checked={$heterogeneous} on:change={onHetToggle} />
      <span>Use different hardware for decode cluster</span>
    </label>
  {/if}
</div>

{#if $heterogeneous && $disaggKvTransferFabricId}
  <div class="decode-cluster">
    <div class="section-label">Decode cluster</div>
    <div class="row">
      <label>
        Accelerator
        <select value={decodeComboValue} on:change={onDecodeComboChange}>
          {#each skuGroups as g}
            <optgroup label={g.publisher}>
              {#each g.entries as e}
                {#if e.kind === 'single'}
                  <option value={`chip:${e.id}`}>{e.name}</option>
                {:else}
                  <option value={`sys:${e.id}`}>{e.name} ({e.count}×)</option>
                {/if}
              {/each}
            </optgroup>
          {/each}
        </select>
      </label>
      {#if !$decodeSystemId}
        <label>
          Variant
          <select bind:value={$decodeVariantId}>
            {#each decodeVariants as v}
              <option value={v.id}>{v.label}</option>
            {/each}
          </select>
        </label>
      {/if}
      <ParallelismPicker side="decode" />
    </div>
  </div>
{/if}

<style>
  .disagg-inputs {
    display: flex; flex-direction: row; flex-wrap: wrap;
    gap: 0.75rem; align-items: flex-end;
    padding: 0.6rem 0.9rem;
    background: #fafafa;
    border: 1px solid #e0e0e0; border-radius: 0.3rem;
    margin-bottom: 0.5rem;
  }
  .decode-cluster {
    padding: 0.6rem 0.9rem;
    background: #fafafa;
    border: 1px solid #e0e0e0; border-radius: 0.3rem;
    margin-bottom: 0.75rem;
  }
  .section-label {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: #555; font-weight: 600; margin-bottom: 0.4rem;
  }
  .row { display: flex; flex-direction: row; flex-wrap: wrap; gap: 0.75rem; align-items: flex-end; }
  label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.9rem; }
  label.inline { flex-direction: row; align-items: center; gap: 0.4rem; font-size: 0.85rem; }
  label.inline input[type=checkbox] { width: auto; }
  select { font-size: 1rem; padding: 0.25rem; min-width: 200px; }
</style>

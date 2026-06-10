<script lang="ts">
  import {
    acceleratorId, variantId, systemId, parallelismOverride,
    // DisaggInputPanel renders inside the Sim tab; bind to the Sim-tab disagg
    // stores so changes here don't bleed into Calc's disagg state.
    simDisaggKvTransferFabricId as disaggKvTransferFabricId,
    simDisaggFirstTokenOnPrefill as disaggFirstTokenOnPrefill,
    heterogeneous,
    prefillAcceleratorId, prefillVariantId, prefillSystemId, prefillParallelismOverride,
    decodeAcceleratorId, decodeVariantId, decodeSystemId, decodeParallelismOverride,
  } from './stores'
  import { groupedDisaggFabrics, formatFabricLabel } from './disaggFabrics'
  import { ACCELERATORS } from '../data'
  import { SYSTEMS } from '../data/systems'
  import { orderSkus } from './catalogOrder'
  import ParallelismPicker from './ParallelismPicker.svelte'

  // V2 heterogeneous: when $heterogeneous is on, both clusters get their own
  // overrides stores (prefill = a1/v1/s1/p1, decode = a2/v2/s2/p2) so the
  // disagg block is fully decoupled from the monolithic block above. The
  // toggle-on handler seeds both from shared so the user starts symmetric and
  // changes one cluster at a time.
  $: groups = groupedDisaggFabrics($acceleratorId)
  $: skuGroups = orderSkus(ACCELERATORS, SYSTEMS)

  // Prefill cluster (= prefill-override stores). Once het is on the seed
  // handler / URL apply guarantees these are populated, so no fallback to
  // shared — that fallback is what made the menu reactively follow the
  // monolithic block.
  $: prefillComboValue = $prefillSystemId
    ? `sys:${$prefillSystemId}`
    : `chip:${$prefillAcceleratorId}`
  $: prefillAcceleratorObj = ACCELERATORS.find(a => a.id === $prefillAcceleratorId)
  $: prefillVariants = prefillAcceleratorObj?.variants ?? []

  function onPrefillComboChange(e: Event) {
    const v = (e.target as HTMLSelectElement).value
    if (v.startsWith('sys:')) {
      prefillSystemId.set(v.slice(4))
    } else {
      prefillSystemId.set('')
      const id = v.slice(5)
      prefillAcceleratorId.set(id)
      const found = ACCELERATORS.find(a => a.id === id)
      if (found) prefillVariantId.set(found.variants[0].id)
    }
  }

  // Decode cluster (= decode-override stores). Same invariant as prefill.
  $: decodeComboValue = $decodeSystemId
    ? `sys:${$decodeSystemId}`
    : `chip:${$decodeAcceleratorId}`
  $: decodeAcceleratorObj = ACCELERATORS.find(a => a.id === $decodeAcceleratorId)
  $: decodeVariants = decodeAcceleratorObj?.variants ?? []

  function onDecodeComboChange(e: Event) {
    const v = (e.target as HTMLSelectElement).value
    if (v.startsWith('sys:')) {
      decodeSystemId.set(v.slice(4))
    } else {
      decodeSystemId.set('')
      const id = v.slice(5)
      decodeAcceleratorId.set(id)
      const found = ACCELERATORS.find(a => a.id === id)
      if (found) decodeVariantId.set(found.variants[0].id)
    }
  }

  // Seed both clusters from shared on first toggle-on so the disagg block
  // starts symmetric and visibly matches the monolithic block above; user
  // then transitions by changing one knob at a time. Only seed when the
  // override store is empty — preserves prior state on re-toggle.
  function onHetToggle(e: Event) {
    const on = (e.target as HTMLInputElement).checked
    heterogeneous.set(on)
    if (!on) return
    if (!$prefillAcceleratorId && !$prefillSystemId) {
      prefillAcceleratorId.set($acceleratorId)
      prefillVariantId.set($variantId)
      prefillSystemId.set($systemId)
      prefillParallelismOverride.set($parallelismOverride)
    }
    if (!$decodeAcceleratorId && !$decodeSystemId) {
      decodeAcceleratorId.set($acceleratorId)
      decodeVariantId.set($variantId)
      decodeSystemId.set($systemId)
      decodeParallelismOverride.set($parallelismOverride)
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
      <span>Use different hardware for prefill and decode clusters</span>
    </label>
  {/if}
</div>

{#if $heterogeneous && $disaggKvTransferFabricId}
  <div class="cluster-pair">
    <div class="cluster">
      <div class="section-label">Prefill cluster</div>
      <div class="row">
        <label>
          Accelerator
          <select value={prefillComboValue} on:change={onPrefillComboChange}>
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
        {#if !$prefillSystemId}
          <label>
            Variant
            <select bind:value={$prefillVariantId}>
              {#each prefillVariants as v}
                <option value={v.id}>{v.label}</option>
              {/each}
            </select>
          </label>
        {/if}
        <ParallelismPicker side="prefill" />
      </div>
    </div>

    <div class="cluster">
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
  .cluster-pair {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;
    margin-bottom: 0.75rem;
  }
  .cluster {
    padding: 0.6rem 0.9rem;
    background: #fafafa;
    border: 1px solid #e0e0e0; border-radius: 0.3rem;
  }
  .section-label {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: #555; font-weight: 600; margin-bottom: 0.4rem;
  }
  .row { display: flex; flex-direction: row; flex-wrap: wrap; gap: 0.75rem; align-items: flex-end; }
  label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.9rem; }
  label.inline { flex-direction: row; align-items: center; gap: 0.4rem; font-size: 0.85rem; }
  label.inline input[type=checkbox] { width: auto; }
  select { font-size: 1rem; padding: 0.25rem; min-width: 180px; }
  @media (max-width: 800px) {
    .cluster-pair { grid-template-columns: 1fr; }
  }
</style>

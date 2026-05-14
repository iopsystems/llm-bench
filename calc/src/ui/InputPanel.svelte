<script lang="ts">
  import { ACCELERATORS, MODELS } from '../data'
  import { SYSTEMS } from '../data/systems'
  import { INTERCONNECTS } from '../data/interconnects'
  import { acceleratorId, variantId, systemId, modelId, quant, workload, disaggKvTransferFabricId, disaggFirstTokenOnPrefill } from './stores'
  import type { Dtype } from '../engine/types'
  import ParallelismPicker from './ParallelismPicker.svelte'

  // Disagg fabric options — scale-out fabrics (IB, EFA) are the realistic ones.
  // Filter to those entries in INTERCONNECTS.
  const disaggFabrics = INTERCONNECTS.filter(i => i.scale === 'scale-out')

  const DTYPES: Dtype[] = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4']

  // Combined catalog entries — each is either a single accelerator or a named system.
  type Entry =
    | { kind: 'single'; id: string; vendor: string; name: string }
    | { kind: 'system'; id: string; vendor: string; name: string; count: number }

  const entries: Entry[] = [
    ...ACCELERATORS.map(a => ({ kind: 'single' as const, id: a.id, vendor: a.vendor, name: a.name })),
    ...SYSTEMS.map(s => ({
      kind: 'system' as const, id: s.id, vendor: s.vendor,
      name: s.name, count: s.accelerator.count
    }))
  ]
  const vendors = Array.from(new Set(entries.map(e => e.vendor)))

  // Single combined value for the dropdown, prefixed to distinguish kind.
  $: comboValue = $systemId ? `sys:${$systemId}` : `chip:${$acceleratorId}`

  function onComboChange(e: Event) {
    const v = (e.target as HTMLSelectElement).value
    if (v.startsWith('sys:')) {
      systemId.set(v.slice(4))
    } else {
      systemId.set('')
      acceleratorId.set(v.slice(5))
    }
  }

  $: accelerator = ACCELERATORS.find(a => a.id === $acceleratorId)
  $: variants = accelerator?.variants ?? []
  $: if (accelerator && !variants.find(v => v.id === $variantId)) {
       variantId.set(variants[0]?.id ?? '')
     }
</script>

<section class="input-panel">
  <fieldset class="island">
    <legend>Hardware</legend>
    <div class="row">
      <label>
        Accelerator
        <select value={comboValue} on:change={onComboChange}>
          {#each vendors as v}
            <optgroup label={v}>
              {#each entries.filter(e => e.vendor === v) as e}
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
      {#if !$systemId}
        <label>
          Variant
          <select bind:value={$variantId}>
            {#each variants as v}
              <option value={v.id}>{v.label}</option>
            {/each}
          </select>
        </label>
      {/if}
      <ParallelismPicker />
      {#if $systemId}
        <label>
          Disagg KV transfer
          <select bind:value={$disaggKvTransferFabricId}>
            <option value="">— integrated —</option>
            {#each disaggFabrics as f}
              <option value={f.id}>{f.name}</option>
            {/each}
          </select>
        </label>
        {#if $disaggKvTransferFabricId}
          <label class="inline">
            <input type="checkbox" bind:checked={$disaggFirstTokenOnPrefill} />
            <span>1st token on prefill (hide transfer in TTFT)</span>
          </label>
        {/if}
      {/if}
    </div>
  </fieldset>

  <fieldset class="island">
    <legend>Model</legend>
    <div class="row">
      <label>
        Model
        <select bind:value={$modelId}>
          {#each MODELS as m}
            <option value={m.id}>{m.name}</option>
          {/each}
        </select>
      </label>
      <label>
        Weights
        <select bind:value={$quant.weights}>
          {#each DTYPES as d}<option value={d}>{d}</option>{/each}
        </select>
      </label>
      <label>
        KV
        <select bind:value={$quant.kv}>
          {#each DTYPES as d}<option value={d}>{d}</option>{/each}
        </select>
      </label>
      <label>
        Activations
        <select bind:value={$quant.activations}>
          {#each DTYPES as d}<option value={d}>{d}</option>{/each}
        </select>
      </label>
    </div>
  </fieldset>

  <fieldset class="island">
    <legend>Workload</legend>
    <div class="row">
      <label>
        Prompt tokens
        <input type="number" min="1" bind:value={$workload.promptTokens} />
      </label>
      <label>
        Output tokens
        <input type="number" min="1" bind:value={$workload.outputTokens} />
      </label>
      <label>
        Concurrency
        <input type="number" min="1" bind:value={$workload.concurrency} />
      </label>
    </div>
  </fieldset>
</section>

<style>
  .input-panel {
    display: flex; flex-direction: row; flex-wrap: wrap; gap: 0.75rem;
    align-items: stretch;
  }
  .island {
    flex: 1 1 220px; min-width: 0;
    border: 1px solid #d4d4d4; border-radius: 0.4rem;
    padding: 0.4rem 0.9rem 0.7rem; margin: 0; background: #fff;
  }
  .island legend {
    padding: 0 0.4rem; font-size: 0.85rem; font-weight: 600;
    color: #555; text-transform: uppercase; letter-spacing: 0.04em;
  }
  /* Inside each island, fields stack vertically so islands stay narrow
     enough to sit three-across on a typical viewport. */
  .row { display: flex; flex-direction: column; gap: 0.5rem; }
  label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.9rem; }
  label.inline { flex-direction: row; align-items: center; gap: 0.4rem; font-size: 0.85rem; }
  label.inline input[type=checkbox] { width: auto; }
  select, input { font-size: 1rem; padding: 0.25rem; width: 100%; box-sizing: border-box; }
</style>

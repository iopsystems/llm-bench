<script lang="ts">
  import { GPUS, MODELS } from '../data'
  import { gpuId, variantId, modelId, quant, workload } from './stores'
  import type { Dtype } from '../engine/types'

  const DTYPES: Dtype[] = ['fp32', 'fp16', 'bf16', 'fp8', 'int8', 'int4']

  $: gpu = GPUS.find(g => g.id === $gpuId)
  $: variants = gpu?.variants ?? []
  // Reset variant if it falls outside the new GPU's list.
  $: if (gpu && !variants.find(v => v.id === $variantId)) {
       variantId.set(variants[0]?.id ?? '')
     }
</script>

<section class="input-panel">
  <div class="row">
    <label>
      GPU
      <select bind:value={$gpuId}>
        {#each GPUS as g}
          <option value={g.id}>{g.name}</option>
        {/each}
      </select>
    </label>

    <label>
      Variant
      <select bind:value={$variantId}>
        {#each variants as v}
          <option value={v.id}>{v.label}</option>
        {/each}
      </select>
    </label>

    <label>
      Model
      <select bind:value={$modelId}>
        {#each MODELS as m}
          <option value={m.id}>{m.name}</option>
        {/each}
      </select>
    </label>
  </div>

  <div class="row">
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
</section>

<style>
  .input-panel { display: flex; flex-direction: column; gap: 0.5rem; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; }
  label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.9rem; }
  select, input { font-size: 1rem; padding: 0.25rem; }
</style>

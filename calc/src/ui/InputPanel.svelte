<script lang="ts">
  import { ACCELERATORS, MODELS } from '../data'
  import { SYSTEMS } from '../data/systems'
  import { acceleratorId, variantId, systemId, modelId, quant, workload, disaggKvTransferFabricId, disaggFirstTokenOnPrefill } from './stores'
  import type { Dtype } from '../engine/types'
  import ParallelismPicker from './ParallelismPicker.svelte'
  import { parseTokenCount, formatTokenCount } from './parseTokens'
  import { orderModels, orderSkus } from './catalogOrder'
  import { groupedDisaggFabrics, formatFabricLabel } from './disaggFabrics'

  export let hideConcurrency = false
  export let hideDisagg = false

  const DTYPES: Dtype[] = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4']

  // Width in bits per dtype, for the weights-only "no upcast / no sideways"
  // selectability rule: a weight option is disabled if its width is >= the
  // model's native dtype AND it isn't the native itself.
  const DTYPE_WIDTH: Record<Dtype, number> = {
    fp32: 32, fp16: 16, bf16: 16, fp8: 8, int8: 8, fp4: 4, int4: 4,
  }
  function isWeightDisabled(d: Dtype, native: Dtype | undefined): boolean {
    if (!native || d === native) return false
    return DTYPE_WIDTH[d] >= DTYPE_WIDTH[native]
  }

  // Picker ordering lives in catalogOrder.ts: publisher groups (newest-shipping
  // publisher first), then within a group newer/larger first. SKU groups also
  // put single accelerators ahead of multi-accelerator systems.
  const skuGroups = orderSkus(ACCELERATORS, SYSTEMS)
  const modelGroups = orderModels(MODELS)

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

  // Disagg fabric options grouped by scale and filtered by accelerator family.
  $: disaggGroups = groupedDisaggFabrics($acceleratorId)
  $: if (accelerator && !variants.find(v => v.id === $variantId)) {
       variantId.set(variants[0]?.id ?? '')
     }

  // Soft warning when promptTokens exceeds the model's trained context window.
  // Calc still runs and extrapolates linearly; the badge surfaces the caveat.
  $: selectedModel = MODELS.find(m => m.id === $modelId)
  $: contextWarning = selectedModel && $workload.promptTokens > selectedModel.maxContext
    ? `> trained ceiling ${formatTokenCount(selectedModel.maxContext)}`
    : null

  // Local string state for unit-aware token inputs ("40k", "1M", etc.).
  // Push to the store only on a successful parse, so a partially-typed or
  // invalid value never propagates NaN downstream and blanks the chart.
  let promptInput = formatTokenCount($workload.promptTokens)
  let outputInput = formatTokenCount($workload.outputTokens)
  let concurrencyInput = String($workload.concurrency)
  let promptInvalid = false
  let outputInvalid = false
  let concurrencyInvalid = false

  function onPromptInput(e: Event) {
    const v = (e.target as HTMLInputElement).value
    promptInput = v
    const n = parseTokenCount(v)
    if (n === null) { promptInvalid = true; return }
    promptInvalid = false
    workload.update(w => ({ ...w, promptTokens: n }))
  }

  function onOutputInput(e: Event) {
    const v = (e.target as HTMLInputElement).value
    outputInput = v
    const n = parseTokenCount(v)
    if (n === null) { outputInvalid = true; return }
    outputInvalid = false
    workload.update(w => ({ ...w, outputTokens: n }))
  }

  function onConcurrencyInput(e: Event) {
    const v = (e.target as HTMLInputElement).value
    concurrencyInput = v
    const n = parseTokenCount(v)
    if (n === null) { concurrencyInvalid = true; return }
    concurrencyInvalid = false
    workload.update(w => ({ ...w, concurrency: n }))
  }
</script>

<section class="input-panel">
  <fieldset class="island">
    <legend>Hardware</legend>
    <div class="row">
      <label>
        Accelerator
        <select value={comboValue} on:change={onComboChange}>
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
      {#if !hideDisagg}
        <label>
          Disagg KV transfer
          <select bind:value={$disaggKvTransferFabricId}>
            <option value="">— integrated —</option>
            {#if disaggGroups.scaleUp.length > 0}
              <optgroup label="Intra-domain (scale-up)">
                {#each disaggGroups.scaleUp as f}
                  <option value={f.id}>{formatFabricLabel(f)}</option>
                {/each}
              </optgroup>
            {/if}
            <optgroup label="Cross-rack (scale-out)">
              {#each disaggGroups.scaleOut as f}
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
          {#each modelGroups as g}
            <optgroup label={g.publisher}>
              {#each g.models as m}
                <option value={m.id}>{m.name}</option>
              {/each}
            </optgroup>
          {/each}
        </select>
      </label>
      <label>
        Weights
        <select bind:value={$quant.weights}>
          {#each DTYPES as d}<option value={d} class:native={d === selectedModel?.nativeDtype} disabled={isWeightDisabled(d, selectedModel?.nativeDtype)}>{d}{d === selectedModel?.nativeDtype ? ' — native' : ''}</option>{/each}
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
          {#each DTYPES as d}<option value={d} class:native={d === selectedModel?.nativeDtype}>{d}{d === selectedModel?.nativeDtype ? ' — native' : ''}</option>{/each}
        </select>
      </label>
    </div>
  </fieldset>

  <fieldset class="island">
    <legend>Workload</legend>
    <div class="row">
      <label>
        Prompt tokens
        <input
          type="text"
          inputmode="numeric"
          value={promptInput}
          on:input={onPromptInput}
          class:invalid={promptInvalid}
          title="Positive integer (≥1). Accepts plain integers or k/M suffixes (1024-based), e.g. 40k, 1M"
        />
        {#if promptInvalid}
          <span class="warn">⚠ invalid — use a positive integer (e.g. 8192, 40k, 1M)</span>
        {:else if contextWarning}
          <span class="warn" title="Model trained at max_position_embeddings={selectedModel?.maxContext}. The calc still runs but accuracy is extrapolated past this ceiling.">
            ⚠ {contextWarning}
          </span>
        {/if}
      </label>
      <label>
        Output tokens
        <input
          type="text"
          inputmode="numeric"
          value={outputInput}
          on:input={onOutputInput}
          class:invalid={outputInvalid}
          title="Positive integer (≥1). Accepts plain integers or k/M suffixes (1024-based)"
        />
        {#if outputInvalid}
          <span class="warn">⚠ invalid — use a positive integer (e.g. 512, 4k)</span>
        {/if}
      </label>
      {#if !hideConcurrency}
        <label>
          Concurrency
          <input
            type="text"
            inputmode="numeric"
            value={concurrencyInput}
            on:input={onConcurrencyInput}
            class:invalid={concurrencyInvalid}
            title="Positive integer (≥1)"
          />
          {#if concurrencyInvalid}
            <span class="warn">⚠ invalid — use a positive integer</span>
          {/if}
        </label>
      {/if}
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
  .warn { font-size: 0.78rem; color: #b85b00; margin-top: 0.15rem; }
  select, input { font-size: 1rem; padding: 0.25rem; width: 100%; box-sizing: border-box; }
  input.invalid { border-color: #b85b00; background: #fff7ec; }
  option.native { font-weight: 700; }
</style>

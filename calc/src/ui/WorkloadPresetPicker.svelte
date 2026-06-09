<script lang="ts">
  import { workload } from './stores'
  import { WORKLOAD_PRESETS, matchPreset } from '../data/workload-presets'

  $: selectedPresetId = matchPreset(
    { promptTokens: $workload.promptTokens, outputTokens: $workload.outputTokens },
    WORKLOAD_PRESETS
  )
  $: activePreset = WORKLOAD_PRESETS.find(p => p.id === selectedPresetId)

  $: codeGenPresets = WORKLOAD_PRESETS.filter(p => p.group === 'code-gen')
  $: otherPresets   = WORKLOAD_PRESETS.filter(p => p.group === 'other')

  // Default workload shape — kept in sync with the workload store's initial
  // value in stores.ts. Selecting "Custom" resets to these so the user has a
  // meaningful escape from any active preset (without it, the picker silently
  // snaps back to whatever preset still matches the values).
  const DEFAULT_PROMPT_TOKENS = 2048
  const DEFAULT_OUTPUT_TOKENS = 512

  function onPresetChange(e: Event) {
    const id = (e.target as HTMLSelectElement).value
    if (id === 'custom') {
      workload.update(w => ({
        ...w,
        promptTokens: DEFAULT_PROMPT_TOKENS,
        outputTokens: DEFAULT_OUTPUT_TOKENS,
      }))
      return
    }
    const preset = WORKLOAD_PRESETS.find(p => p.id === id)
    if (!preset) return
    workload.update(w => ({
      ...w,
      promptTokens: preset.promptTokens,
      outputTokens: preset.outputTokens,
    }))
  }

  function prettyHost(url: string): string {
    try {
      const u = new URL(url)
      return u.host + u.pathname.replace(/\/+$/, '')
    } catch {
      return url
    }
  }
</script>

<label class="preset-row">
  Benchmark preset
  <select value={selectedPresetId} on:change={onPresetChange}>
    <option value="custom">Custom</option>
    {#if codeGenPresets.length > 0}
      <optgroup label="Code-gen">
        {#each codeGenPresets as p}
          <option value={p.id} title={p.description}>{p.name}</option>
        {/each}
      </optgroup>
    {/if}
    {#if otherPresets.length > 0}
      <optgroup label="Other">
        {#each otherPresets as p}
          <option value={p.id} title={p.description}>{p.name}</option>
        {/each}
      </optgroup>
    {/if}
  </select>
</label>
{#if activePreset}
  <div class="preset-source">
    Source: <a href={activePreset.sourceUrl} target="_blank" rel="noopener">{prettyHost(activePreset.sourceUrl)}</a>
    &middot; as of {activePreset.sourceAccessedAt}
    &middot; tokenized with Llama-3 reference tokenizer (±10–20% on other tokenizers)
  </div>
{/if}

<style>
  .preset-row {
    display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.9rem;
  }
  .preset-row select {
    font-size: 1rem; padding: 0.25rem; min-width: 220px;
  }
  .preset-source {
    margin-top: 0.3rem;
    font-size: 0.78rem; color: #666;
    line-height: 1.4;
  }
  .preset-source a { color: #2b6cb0; }
</style>

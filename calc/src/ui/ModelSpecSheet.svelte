<!-- calc/src/ui/ModelSpecSheet.svelte -->
<script lang="ts">
  import type { ModelArch } from '../engine/types'
  import { modelMetrics } from './catalogMetrics'
  export let model: ModelArch
  $: m = modelMetrics(model)
  $: arch = model.architecture
  function kb(bytes: number): string {
    return bytes >= 1024 ? `${(bytes / 1024).toFixed(2)} KB` : `${bytes} B`
  }
  function paramsStr(n: number): string {
    return n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : `${(n / 1e6).toFixed(0)}M`
  }
</script>

<article class="sheet">
  <h2>{model.name}</h2>
  <dl>
    <dt>Publisher</dt><dd>{model.publisher}</dd>
    <dt>Family</dt><dd>{model.family}</dd>
    <dt>Released</dt><dd>{model.releaseDate}</dd>
    <dt>Parameters</dt>
    <dd>
      {paramsStr(model.paramCount)} total
      {#if arch.type === 'moe'}· {paramsStr(arch.activeParamCount)} active
        ({(m.moeActiveRatio! * 100).toFixed(1)}%){/if}
    </dd>
    <dt>Architecture</dt>
    <dd>
      {#if arch.type === 'moe'}MoE — {arch.numExperts} experts, {arch.numExpertsActive} active{#if arch.numSharedExperts}, {arch.numSharedExperts} shared{/if}
      {:else}Dense{/if}
    </dd>
    <dt>Layers</dt><dd>{model.layers}</dd>
    <dt>Hidden / Intermediate</dt><dd>{model.hiddenDim} / {model.intermediateDim}</dd>
    <dt>Heads (Q / KV)</dt><dd>{model.numHeads} / {model.numKvHeads} · head dim {model.headDim}</dd>
    <dt>GQA ratio</dt><dd>{m.gqaRatio.toFixed(1)}×</dd>
    <dt>Vocab</dt><dd>{model.vocabSize.toLocaleString()}</dd>
    <dt>Max context</dt><dd>{model.maxContext.toLocaleString()} tokens</dd>
    <dt>Attention</dt><dd>{m.attentionLabel}</dd>
    {#if model.numNextnLayers > 0}
      <dt>MTP depth</dt><dd>{model.numNextnLayers}</dd>
    {/if}
  </dl>
  <h3>Derived <span class="ref">(fp16 KV reference)</span></h3>
  <dl>
    <dt>KV / token / layer</dt><dd>{kb(m.kvBytesPerTokenPerLayer)}</dd>
    <dt>KV / token (model)</dt><dd>{kb(m.kvBytesPerToken)}</dd>
  </dl>
</article>

<style>
  .sheet { max-width: 640px; }
  h2 { margin: 0 0 0.75rem; }
  h3 { margin: 1.25rem 0 0.4rem; font-size: 0.95rem; }
  .ref { font-weight: 400; color: #888; font-size: 0.8rem; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1rem; margin: 0; }
  dt { color: #666; }
  dd { margin: 0; font-variant-numeric: tabular-nums; }
</style>

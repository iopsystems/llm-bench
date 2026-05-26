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
  <h2><slot />{model.name}</h2>
  <div class="rule-thick"></div>
  <dl>
    <dt>Publisher</dt><dd>{model.publisher}</dd>
    <dt>Family</dt><dd>{model.family}</dd>
    <dt>Released</dt><dd>{model.releaseDate}</dd>
  </dl>

  <div class="rule"></div>
  <h3>Design</h3>
  <dl>
    <dt>Architecture</dt>
    <dd>{arch.type === 'moe' ? 'Mixture of experts' : 'Dense'}</dd>
    <dt>Attention</dt><dd>{m.attentionLabel}</dd>
    <dt>Native precision</dt><dd>{model.nativeDtype}</dd>
  </dl>

  <div class="rule"></div>
  <h3>Scale</h3>
  <dl>
    <dt>Parameters</dt>
    <dd>
      {paramsStr(model.paramCount)} total
      {#if arch.type === 'moe'}· {paramsStr(arch.activeParamCount)} active
        ({(m.moeActiveRatio! * 100).toFixed(1)}%){/if}
    </dd>
    {#if arch.type === 'moe'}
      <dt>Experts</dt>
      <dd>{arch.numExperts} total · {arch.numExpertsActive} active{#if arch.numSharedExperts} · {arch.numSharedExperts} shared{/if}</dd>
    {/if}
  </dl>

  <div class="rule"></div>
  <h3>Dimensions</h3>
  <dl>
    <dt>Layers</dt><dd>{model.layers}</dd>
    <dt>Hidden / Intermediate</dt><dd>{model.hiddenDim} / {model.intermediateDim}</dd>
    <dt>Heads (Q / KV)</dt>
    <dd>{model.numHeads} / {model.numKvHeads} · head dim {model.headDim} · GQA {m.gqaRatio.toFixed(1)}×</dd>
  </dl>

  <div class="rule"></div>
  <h3>Tokenizer &amp; context</h3>
  <dl>
    <dt>Vocab</dt><dd>{model.vocabSize.toLocaleString()}</dd>
    <dt>Max context</dt><dd>{model.maxContext.toLocaleString()} tokens</dd>
    {#if model.numNextnLayers > 0}
      <dt>MTP depth</dt><dd>{model.numNextnLayers}</dd>
    {/if}
  </dl>

  <div class="rule"></div>
  <h3>Derived memory <span class="ref">(fp16 KV reference)</span></h3>
  <dl>
    <dt>KV / token / layer</dt><dd>{kb(m.kvBytesPerTokenPerLayer)}</dd>
    <dt>KV / token (model)</dt><dd>{kb(m.kvBytesPerToken)}</dd>
  </dl>
</article>

<style>
  /* Nutrition-label aesthetic — matches SkuSpecSheet. */
  .sheet {
    max-width: 640px; border: 2px solid #111; border-radius: 4px;
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
</style>

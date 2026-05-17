<!-- calc/src/ui/InfoPanel.svelte -->
<script lang="ts">
  import { ACCELERATORS, MODELS } from '../data'
  import { SYSTEMS } from '../data/systems'
  import { orderModels, orderSkus } from './catalogOrder'
  import { route, navigate } from './route'
  import { modelId, acceleratorId, systemId } from './stores'
  import ModelSpecSheet from './ModelSpecSheet.svelte'
  import SkuSpecSheet from './SkuSpecSheet.svelte'

  let section: 'models' | 'skus' = 'models'
  // The selected card is shown by default but collapsible so the browse
  // list below isn't pushed off-screen.
  let cardOpen = true

  const modelGroups = orderModels(MODELS)
  const skuGroups = orderSkus(ACCELERATORS, SYSTEMS)

  // Resolve the detail target from the route, if any.
  $: detail = $route.tab === 'info' && 'detail' in $route ? $route.detail : null
  $: detailModel = detail?.kind === 'model' ? MODELS.find(m => m.id === detail.id) : undefined
  $: detailSku = detail?.kind === 'sku'
    ? (ACCELERATORS.find(a => a.id === detail.id) ?? SYSTEMS.find(s => s.id === detail.id))
    : undefined

  // Pin the calculator's current selection at the top of the relevant list.
  $: pinnedModelId = $modelId
  $: pinnedSkuId = $systemId || $acceleratorId
  $: pinnedModel = MODELS.find(m => m.id === pinnedModelId)
  $: pinnedSku = ACCELERATORS.find(a => a.id === pinnedSkuId) ?? SYSTEMS.find(s => s.id === pinnedSkuId)
</script>

<section class="info">
  {#if detailModel}
    <button class="back" on:click={() => navigate({ tab: 'info' })}>← All models &amp; SKUs</button>
    <ModelSpecSheet model={detailModel} />
  {:else if detailSku}
    <button class="back" on:click={() => navigate({ tab: 'info' })}>← All models &amp; SKUs</button>
    <SkuSpecSheet sku={detailSku} />
  {:else}
    <div class="subtabs">
      <button class:active={section === 'models'} on:click={() => section = 'models'}>Models</button>
      <button class:active={section === 'skus'} on:click={() => section = 'skus'}>SKUs</button>
    </div>

    {#if section === 'models'}
      {#if pinnedModel}
        <div class="selected">
          <div class="selhead">
            <span class="pinlabel">Selected in calculator</span>
            <button class="collapse" on:click={() => cardOpen = !cardOpen}
              aria-expanded={cardOpen}>
              {cardOpen ? '▾ Collapse' : `▸ ${pinnedModel.name}`}
            </button>
          </div>
          {#if cardOpen}<ModelSpecSheet model={pinnedModel} />{/if}
        </div>
      {/if}
      <div class="groups">
        {#each modelGroups as g}
          <div class="group">
            <h3>{g.publisher}</h3>
            <ul>
              {#each g.models as m}
                <li>
                  <button class="entry" class:pinned={m.id === pinnedModelId}
                    on:click={() => navigate({ tab: 'info', detail: { kind: 'model', id: m.id } })}>
                    {m.name}{#if m.id === pinnedModelId} <span class="badge">selected</span>{/if}
                  </button>
                </li>
              {/each}
            </ul>
          </div>
        {/each}
      </div>
    {:else}
      {#if pinnedSku}
        <div class="selected">
          <div class="selhead">
            <span class="pinlabel">Selected in calculator</span>
            <button class="collapse" on:click={() => cardOpen = !cardOpen}
              aria-expanded={cardOpen}>
              {cardOpen ? '▾ Collapse' : `▸ ${pinnedSku.name}`}
            </button>
          </div>
          {#if cardOpen}<SkuSpecSheet sku={pinnedSku} />{/if}
        </div>
      {/if}
      <div class="groups">
        {#each skuGroups as g}
          <div class="group">
            <h3>{g.publisher}</h3>
            <ul>
              {#each g.entries as e}
                <li>
                  <button class="entry" class:pinned={e.id === pinnedSkuId}
                    on:click={() => navigate({ tab: 'info', detail: { kind: 'sku', id: e.id } })}>
                    {e.name}{#if e.kind === 'system'} ({e.count}×){/if}
                    {#if e.id === pinnedSkuId} <span class="badge">selected</span>{/if}
                  </button>
                </li>
              {/each}
            </ul>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</section>

<style>
  .info { max-width: 1100px; }
  .subtabs { display: flex; gap: 0.4rem; margin-bottom: 1.25rem; }
  .subtabs button {
    font: inherit; font-size: 0.95rem; font-weight: 600; padding: 0.45rem 1.1rem;
    border: 1px solid #c4c4c4; background: #fff; color: #444;
    cursor: pointer; border-radius: 0.35rem;
  }
  .subtabs button:hover { background: #f1f1f1; }
  .subtabs button.active { background: #333; color: #fff; border-color: #333; }
  /* Newspaper-style flow: variable-height publisher groups fill the screen
     width in columns, collapsing to one column on narrow viewports. */
  .groups { columns: 240px; column-gap: 2rem; }
  .group {
    break-inside: avoid; -webkit-column-break-inside: avoid;
    padding-bottom: 0.9rem;
  }
  h3 {
    margin: 0 0 0.3rem; font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.04em; color: #888;
  }
  ul { list-style: none; margin: 0; padding: 0; }
  li { margin: 0; }
  .entry {
    font: inherit; font-size: 0.95rem; width: 100%; text-align: left;
    background: none; border: none; padding: 0.3rem 0.4rem; cursor: pointer;
    color: #1a4f8a; border-radius: 0.25rem;
  }
  .entry:hover { background: #eef2f7; }
  .entry.pinned { font-weight: 600; }
  .badge {
    font-size: 0.7rem; color: #fff; background: #21a87a;
    padding: 0.05rem 0.35rem; border-radius: 0.25rem; vertical-align: middle;
  }
  .back {
    font: inherit; font-size: 0.85rem; background: none; border: none;
    color: #1a4f8a; cursor: pointer; padding: 0 0 0.75rem; display: block;
  }
  .selected { margin-bottom: 1.5rem; }
  .selhead {
    display: flex; align-items: center; justify-content: space-between;
    gap: 0.75rem; margin-bottom: 0.35rem;
  }
  .pinlabel {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.04em; color: #888;
  }
  .collapse {
    font: inherit; font-size: 0.8rem; background: #f1f1f1; color: #333;
    border: 1px solid #c8c8c8; border-radius: 0.3rem;
    padding: 0.2rem 0.6rem; cursor: pointer;
  }
  .collapse:hover { background: #e7e7e7; }
</style>

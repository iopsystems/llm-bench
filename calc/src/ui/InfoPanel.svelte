<!-- calc/src/ui/InfoPanel.svelte -->
<script lang="ts">
  import { ACCELERATORS, MODELS } from '../data'
  import { SYSTEMS } from '../data/systems'
  import { orderModels, orderSkus } from './catalogOrder'
  import { route, navigate } from './route'
  import { modelId, acceleratorId, systemId } from './stores'
  import ModelSpecSheet from './ModelSpecSheet.svelte'
  import SkuSpecSheet from './SkuSpecSheet.svelte'

  const modelGroups = orderModels(MODELS)
  const skuGroups = orderSkus(ACCELERATORS, SYSTEMS)

  let section: 'models' | 'skus' = 'models'
  let cardOpen = true

  // Route detail (deep link / click) takes precedence; otherwise the card
  // shows whatever the calculator currently has selected.
  $: routeDetail = $route.tab === 'info' && 'detail' in $route ? $route.detail : null
  $: pinnedModel = MODELS.find(m => m.id === $modelId)
  $: pinnedSku = ACCELERATORS.find(a => a.id === ($systemId || $acceleratorId))
    ?? SYSTEMS.find(s => s.id === ($systemId || $acceleratorId))
  $: activeModel = (routeDetail?.kind === 'model'
    ? MODELS.find(m => m.id === routeDetail.id) : undefined) ?? pinnedModel
  $: activeSku = (routeDetail?.kind === 'sku'
    ? (ACCELERATORS.find(a => a.id === routeDetail.id) ?? SYSTEMS.find(s => s.id === routeDetail.id))
    : undefined) ?? pinnedSku
  // A detail route forces its section; otherwise the manual toggle wins.
  $: effSection = routeDetail ? (routeDetail.kind === 'model' ? 'models' : 'skus') : section

  function selectSection(s: 'models' | 'skus') {
    section = s
    navigate({ tab: 'info' })  // clear any detail so the toggle takes effect
  }

  // SKU browse columns are explicitly bucketed by vendor (per product
  // decision): NVIDIA | cloud-only (Google·AWS·Cerebras) | Intel·rest.
  $: skuByPublisher = new Map(skuGroups.map(g => [g.publisher, g]))
  const pick = (m: typeof skuByPublisher, names: string[]) =>
    names.map(n => m.get(n)).filter((g): g is NonNullable<typeof g> => !!g)
  $: skuColumns = [
    pick(skuByPublisher, ['NVIDIA']),
    pick(skuByPublisher, ['Google', 'AWS', 'Cerebras']),
    [
      ...pick(skuByPublisher, ['Intel']),
      ...skuGroups.filter(g =>
        !['NVIDIA', 'Google', 'AWS', 'Cerebras', 'Intel'].includes(g.publisher)),
    ],
  ]
</script>

<section class="info">
  <div class="subtabs">
    <button class:active={effSection === 'models'} on:click={() => selectSection('models')}>Models</button>
    <button class:active={effSection === 'skus'} on:click={() => selectSection('skus')}>SKUs</button>
  </div>

  {#if effSection === 'models'}
    {#if activeModel}
      <div class="cardwrap">
        {#if cardOpen}
          <ModelSpecSheet model={activeModel}>
            <button class="cardtoggle" title="Collapse" aria-label="Collapse"
              aria-expanded="true" on:click={() => cardOpen = false}>−</button>
          </ModelSpecSheet>
        {:else}
          <div class="collapsed">
            <button class="cardtoggle" title="Expand" aria-label="Expand"
              aria-expanded="false" on:click={() => cardOpen = true}>+</button>
            {activeModel.name}
          </div>
        {/if}
      </div>
    {/if}
    <div class="groups">
      {#each modelGroups as g}
        <div class="group">
          <h3>{g.publisher}</h3>
          <ul>
            {#each g.models as m}
              <li>
                <button class="entry" class:pinned={m.id === activeModel?.id}
                  on:click={() => navigate({ tab: 'info', detail: { kind: 'model', id: m.id } })}>
                  {m.name}
                </button>
              </li>
            {/each}
          </ul>
        </div>
      {/each}
    </div>
  {:else}
    {#if activeSku}
      <div class="cardwrap">
        {#if cardOpen}
          <SkuSpecSheet sku={activeSku}>
            <button class="cardtoggle" title="Collapse" aria-label="Collapse"
              aria-expanded="true" on:click={() => cardOpen = false}>−</button>
          </SkuSpecSheet>
        {:else}
          <div class="collapsed">
            <button class="cardtoggle" title="Expand" aria-label="Expand"
              aria-expanded="false" on:click={() => cardOpen = true}>+</button>
            {activeSku.name}
          </div>
        {/if}
      </div>
    {/if}
    <div class="skucols">
      {#each skuColumns as col}
        <div class="skucol">
          {#each col as g}
            <div class="group">
              <h3>{g.publisher}</h3>
              <ul>
                {#each g.entries as e}
                  <li>
                    <button class="entry" class:pinned={e.id === activeSku?.id}
                      on:click={() => navigate({ tab: 'info', detail: { kind: 'sku', id: e.id } })}>
                      {e.name}{#if e.kind === 'system'} ({e.count}×){/if}
                    </button>
                  </li>
                {/each}
              </ul>
            </div>
          {/each}
        </div>
      {/each}
    </div>
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

  /* Card sits above the always-visible browse list, with a collapse icon
     pinned to its own top-right corner. */
  .cardwrap {
    display: inline-block; vertical-align: top; margin-bottom: 1.5rem;
  }
  /* The toggle is the first element on the card's title line (inside the
     border, via the sheet's title slot / the collapsed bar). Sized to sit
     on the 1.25rem title; same metrics in both states so it doesn't move. */
  :global(.cardtoggle) {
    flex: none; width: 1.6rem; height: 1.6rem; display: inline-flex;
    align-items: center; justify-content: center; font-size: 1.05rem;
    line-height: 1; background: #fff; color: #555;
    border: 1px solid #c8c8c8; border-radius: 0.25rem; cursor: pointer;
  }
  :global(.cardtoggle):hover { background: #efefef; color: #111; }
  /* Mirror the sheet card frame + title metrics so collapsing reads as
     hiding the body, with the title line (and the toggle) unmoved. */
  .collapsed {
    border: 2px solid #111; border-radius: 4px;
    padding: 0.9rem 1.1rem; font-size: 1.25rem; font-weight: 600;
    display: flex; align-items: center; gap: 0.5rem;
  }

  .groups { columns: 240px; column-gap: 2rem; }
  .skucols { display: flex; flex-wrap: wrap; gap: 2rem; align-items: flex-start; }
  .skucol { flex: 1 1 220px; min-width: 0; }
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
  .entry.pinned { font-weight: 700; }
</style>

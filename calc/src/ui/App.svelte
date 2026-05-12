<script lang="ts">
  import InputPanel from './InputPanel.svelte'
  import MemoryPanel from './MemoryPanel.svelte'
  import PerfPanel from './PerfPanel.svelte'
  import RooflinePanel from './RooflinePanel.svelte'
  import DerivationDrawer from './DerivationDrawer.svelte'
  import { error } from './stores'
</script>

<main>
  <header>
    <h1>LLM Performance Calculator</h1>
    <p>Roofline estimates for dense decoder-only transformers.</p>
  </header>
  <InputPanel />
  {#if $error}
    <div class="error">⚠ {$error}</div>
  {/if}
  <MemoryPanel />
  <PerfPanel />
  <RooflinePanel />
  <DerivationDrawer />
</main>

<style>
  /* Force the vertical scrollbar to always be present so the viewport width
     doesn't fluctuate when content tips past/under the viewport height —
     that fluctuation was shrinking the memory bar by 16px on OOM. */
  :global(html) { overflow-y: scroll; }
  :global(body) {
    margin: 0; font-family: system-ui, -apple-system, sans-serif;
    background: #fafafa; color: #222;
  }
  main { max-width: 960px; margin: 0 auto; padding: 1.5rem; }
  @media (max-width: 640px) {
    main { padding: 0.75rem; }
    h1 { font-size: 1.4rem; }
  }
  header { margin-bottom: 1.5rem; }
  h1 { margin: 0 0 0.25rem; }
  header p { margin: 0; color: #666; }
  .error {
    margin-top: 1rem; padding: 0.5rem 0.75rem;
    background: #fde6e6; color: #8a1f1f;
    border: 1px solid #f0b0b0; border-radius: 0.25rem;
    font-size: 0.9rem;
  }
</style>

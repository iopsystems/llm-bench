<script lang="ts">
  import InputPanel from './InputPanel.svelte'
  import MemoryPanel from './MemoryPanel.svelte'
  import PerfPanel from './PerfPanel.svelte'
  import DerivationDrawer from './DerivationDrawer.svelte'
</script>

<main>
  <header>
    <h1>LLM Performance Calculator</h1>
    <p>Roofline estimates for dense decoder-only transformers.</p>
  </header>
  <InputPanel />
  <MemoryPanel />
  <PerfPanel />
  <DerivationDrawer />
</main>

<style>
  :global(body) {
    margin: 0; font-family: system-ui, -apple-system, sans-serif;
    background: #fafafa; color: #222;
  }
  main { max-width: 960px; margin: 0 auto; padding: 1.5rem; }
  header { margin-bottom: 1.5rem; }
  h1 { margin: 0 0 0.25rem; }
  header p { margin: 0; color: #666; }
</style>

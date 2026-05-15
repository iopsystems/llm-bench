<script lang="ts">
  import InputPanel from './InputPanel.svelte'
  import MemoryPanel from './MemoryPanel.svelte'
  import PerfPanel from './PerfPanel.svelte'
  import RooflinePanel from './RooflinePanel.svelte'
  import DerivationDrawer from './DerivationDrawer.svelte'
  import { error } from './stores'
  import { buildShareUrl } from './share'

  let copied = false
  let copyTimer: ReturnType<typeof setTimeout> | null = null

  async function copyLink() {
    const url = buildShareUrl()
    try {
      await navigator.clipboard.writeText(url)
      copied = true
      if (copyTimer) clearTimeout(copyTimer)
      copyTimer = setTimeout(() => { copied = false }, 1500)
    } catch {
      // Clipboard API blocked (insecure context, permission denied). Fall back
      // to selecting the URL bar by no-op; the address bar already carries the
      // up-to-date hash so the user can copy it manually.
    }
  }
</script>

<main>
  <header>
    <div class="title-row">
      <h1>LLM Performance Calculator</h1>
      <button type="button" class="share" on:click={copyLink} title="Copy a shareable link that restores these inputs">
        {copied ? 'Copied' : 'Copy link'}
      </button>
    </div>
    <p>Roofline estimates for modern decoder-only LLMs.</p>
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
  .title-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 0.75rem; flex-wrap: wrap;
  }
  h1 { margin: 0 0 0.25rem; }
  header p { margin: 0; color: #666; }
  .share {
    font: inherit; font-size: 0.85rem;
    padding: 0.3rem 0.7rem; border-radius: 0.3rem;
    border: 1px solid #c8c8c8; background: #fff; color: #333;
    cursor: pointer;
  }
  .share:hover { background: #f1f1f1; }
  .share:active { background: #e6e6e6; }
  .error {
    margin-top: 1rem; padding: 0.5rem 0.75rem;
    background: #fde6e6; color: #8a1f1f;
    border: 1px solid #f0b0b0; border-radius: 0.25rem;
    font-size: 0.9rem;
  }
</style>

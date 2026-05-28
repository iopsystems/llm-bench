<!-- calc/src/ui/TabBar.svelte -->
<script lang="ts">
  import { route, navigate } from './route'
  // Compare / Cloud tabs are added by roadmap items #5 / #6.
  const tabs = [
    { id: 'calc' as const, label: 'Calculator' },
    { id: 'sim'  as const, label: 'Simulator' },
    { id: 'info' as const, label: 'Info' },
  ]
</script>

<nav class="tabbar">
  {#each tabs as t}
    <button
      type="button"
      class:active={$route.tab === t.id}
      on:click={() => navigate(t.id === 'info' ? { tab: 'info' } : { tab: t.id })}
    >{t.label}</button>
  {/each}
</nav>

<style>
  .tabbar { display: flex; gap: 0.25rem; margin-bottom: 1rem; border-bottom: 1px solid #d4d4d4; }
  button {
    font: inherit; font-size: 0.9rem; padding: 0.45rem 0.9rem;
    border: none; background: none; color: #555; cursor: pointer;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  button:hover { color: #222; }
  button.active { color: #111; border-bottom-color: #333; font-weight: 600; }
</style>

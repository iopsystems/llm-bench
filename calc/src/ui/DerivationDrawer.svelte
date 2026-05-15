<script lang="ts">
  import { result } from './stores'
  let open = $state(false)

  function fmt(value: number, unit: string): string {
    if (unit === 'bytes' && value >= 1024 ** 3) return (value / 1024 ** 3).toFixed(2) + ' GB'
    if (unit === 'bytes' && value >= 1024 ** 2) return (value / 1024 ** 2).toFixed(2) + ' MB'
    if (unit === 'bytes' && value >= 1024)      return (value / 1024).toFixed(2) + ' KB'
    if (unit === 's' && value < 0.001)          return (value * 1e6).toFixed(2) + ' µs'
    if (unit === 's' && value < 1)              return (value * 1000).toFixed(3) + ' ms'
    return value.toLocaleString() + ' ' + unit
  }
</script>

<button class="toggle" class:open onclick={() => open = !open}>
  {open ? '✕ Hide' : '☰ Show'} math
</button>

{#if open && $result}
  <aside class="drawer">
    <h3>Derivation</h3>
    <ol>
      {#each $result.derivation as step}
        <li>
          <div class="label">{step.label}</div>
          <code class="expr">{step.expression}</code>
          <div class="value">= {fmt(step.value, step.unit)}</div>
        </li>
      {/each}
    </ol>
  </aside>
{/if}

<style>
  .toggle {
    position: fixed; bottom: 1rem; right: 1rem; z-index: 11;
    background: #333; color: #fff; border: none; padding: 0.5rem 1rem;
    cursor: pointer; font-family: inherit;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    border-radius: 4px;
  }
  .drawer {
    position: fixed; top: 0; right: 0; bottom: 0; width: min(420px, 90vw);
    background: #fff; border-left: 1px solid #888;
    overflow-y: auto; padding: 1rem 1rem 4rem; z-index: 10;
    box-shadow: -4px 0 12px rgba(0,0,0,0.1);
  }
  /* On narrow viewports the chart hugs the bottom-right corner, so the
     floating button covers the data when closed. Flow it inline below
     the roofline (preceding sibling in App.svelte). But when the drawer
     is open, re-float it at top-right above the drawer so it remains
     reachable as the close affordance. */
  @media (max-width: 720px) {
    .toggle:not(.open) {
      position: static;
      display: block;
      margin: 1rem auto 0;
    }
    .toggle.open {
      top: 1rem; bottom: auto;
      z-index: 12;
    }
  }
  ol { list-style: decimal inside; padding-left: 0; }
  li { margin-bottom: 0.75rem; }
  .label { font-weight: 600; }
  .expr { display: block; font-size: 0.85rem; color: #555; margin: 0.1rem 0; }
  .value { font-variant-numeric: tabular-nums; }
</style>

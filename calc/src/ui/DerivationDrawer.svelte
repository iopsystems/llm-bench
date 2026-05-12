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

<button class="toggle" onclick={() => open = !open}>
  {open ? '✕' : '☰'} Show math
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
    position: fixed; top: 1rem; right: 1rem; z-index: 11;
    background: #333; color: #fff; border: none; padding: 0.5rem 1rem;
    cursor: pointer; font-family: inherit;
  }
  .drawer {
    position: fixed; top: 0; right: 0; bottom: 0; width: min(420px, 90vw);
    background: #fff; border-left: 1px solid #888;
    overflow-y: auto; padding: 3rem 1rem 1rem; z-index: 10;
    box-shadow: -4px 0 12px rgba(0,0,0,0.1);
  }
  ol { list-style: decimal inside; padding-left: 0; }
  li { margin-bottom: 0.75rem; }
  .label { font-weight: 600; }
  .expr { display: block; font-size: 0.85rem; color: #555; margin: 0.1rem 0; }
  .value { font-variant-numeric: tabular-nums; }
</style>

<script lang="ts">
  import { systemId, modelId, parallelismOverride } from './stores'
  import { SYSTEMS } from '../data/systems'
  import { MODELS } from '../data'
  import { defaultParallelism, type ParallelismConfig } from '../engine/parallelism'

  $: system = SYSTEMS.find(s => s.id === $systemId)
  $: model = MODELS.find(m => m.id === $modelId)
  $: defaults = system && model ? defaultParallelism(system, model) : null
  $: active = $parallelismOverride ?? defaults

  function candidates(sys: NonNullable<typeof system>, isMoE: boolean): ParallelismConfig[] {
    const N = sys.accelerator.count
    const opts: ParallelismConfig[] = []
    if (N <= 8) {
      opts.push({ parallelism: ['tp'], parallelismDegrees: { tp: N } })
    }
    if (N > 8 && N % 8 === 0) {
      opts.push({ parallelism: ['tp', 'pp'], parallelismDegrees: { tp: 8, pp: N / 8 } })
    }
    if (isMoE) {
      const last = opts[opts.length - 1]
      if (last) {
        opts.push({
          parallelism: [...last.parallelism, 'ep'],
          parallelismDegrees: { ...last.parallelismDegrees, ep: N }
        })
      }
    }
    return opts
  }

  $: opts = system && model ? candidates(system, model.architecture.type === 'moe') : []

  function describe(p: ParallelismConfig): string {
    return Object.entries(p.parallelismDegrees)
      .map(([k, v]) => `${k.toUpperCase()}=${v}`)
      .join(' × ')
  }

  function onChange(e: Event) {
    const v = (e.target as HTMLSelectElement).value
    if (v === 'default') {
      parallelismOverride.set(null)
    } else {
      parallelismOverride.set(JSON.parse(v))
    }
  }
</script>

{#if system && active && defaults}
  <label>
    Parallelism
    <select on:change={onChange}>
      <option value="default">Auto: {describe(defaults)}</option>
      {#each opts as o}
        <option value={JSON.stringify(o)}>{describe(o)}</option>
      {/each}
    </select>
  </label>
{/if}

<style>
  label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.9rem; }
  select { font-size: 1rem; padding: 0.25rem; width: 100%; box-sizing: border-box; }
</style>

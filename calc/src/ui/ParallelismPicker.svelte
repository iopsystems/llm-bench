<script lang="ts">
  import {
    systemId as sharedSystemId,
    parallelismOverride as sharedParallelism,
    prefillSystemId,
    prefillParallelismOverride,
    decodeSystemId,
    decodeParallelismOverride,
    modelId,
  } from './stores'
  import { SYSTEMS } from '../data/systems'
  import { MODELS } from '../data'
  import { defaultParallelism, type ParallelismConfig } from '../engine/parallelism'

  // side='shared' (default) — monolithic / Calc-tab stores (a/v/s/p).
  // side='prefill' — disagg prefill-cluster overrides (a1/v1/s1/p1).
  // side='decode'  — disagg decode-cluster overrides (a2/v2/s2/p2).
  export let side: 'shared' | 'prefill' | 'decode' = 'shared'

  $: activeSystemIdValue =
    side === 'decode'  ? $decodeSystemId :
    side === 'prefill' ? $prefillSystemId :
                         $sharedSystemId
  $: activeParallelismValue =
    side === 'decode'  ? $decodeParallelismOverride :
    side === 'prefill' ? $prefillParallelismOverride :
                         $sharedParallelism
  $: activeParallelismStore =
    side === 'decode'  ? decodeParallelismOverride :
    side === 'prefill' ? prefillParallelismOverride :
                         sharedParallelism

  $: system = SYSTEMS.find(s => s.id === activeSystemIdValue)
  $: model = MODELS.find(m => m.id === $modelId)
  $: defaults = system && model ? defaultParallelism(system, model) : null
  $: active = activeParallelismValue ?? defaults

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
      activeParallelismStore.set(null)
    } else {
      activeParallelismStore.set(JSON.parse(v))
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

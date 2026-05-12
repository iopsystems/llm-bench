<script lang="ts">
  import { result } from './stores'
  import { SOURCES } from '../data/sources'
  import type { PerfTier } from '../engine/types'

  function ms(s: number): string { return (s * 1000).toFixed(2) + ' ms' }
  function rate(tps: number): string { return tps.toFixed(1) + ' tok/s' }

  // Citations are scoped to the operating point that declares them. Numbering
  // is local per op point — `[1][2]` next to "achievable" refers to that row's
  // own references block, not anything global.
  function citationsFor(p: PerfTier): { key: string; n: number; title: string; url: string }[] {
    const out: { key: string; n: number; title: string; url: string }[] = []
    const keys = p.sources ?? []
    for (let i = 0; i < keys.length; i++) {
      const src = SOURCES[keys[i] as keyof typeof SOURCES]
      if (!src) continue
      out.push({ key: keys[i], n: i + 1, title: src.title, url: src.url })
    }
    return out
  }
</script>

{#if $result}
  <section class="perf-panel">
    <h3>Performance</h3>
    <table>
      <thead>
        <tr>
          <th>Operating point</th>
          <th>TTFT</th>
          <th>Prefill regime</th>
          <th>Decode time / tok</th>
          <th>Decode regime</th>
          <th>Input tok/s</th>
          <th>Output tok/s (aggregate)</th>
        </tr>
      </thead>
      <tbody>
        {#each Object.entries($result.perf) as [id, p]}
          {@const cites = citationsFor(p)}
          <tr>
            <td>
              {id}
              {#each cites as c}<sup class="cite"><a href="#ref-{id}-{c.key}">[{c.n}]</a></sup>{/each}
            </td>
            <td>{ms(p.ttftS)}</td>
            <td><span class="regime {p.prefill.regime}">{p.prefill.regime}</span></td>
            <td>{ms(p.decode.timePerTokenS)}</td>
            <td><span class="regime {p.decode.regime}">{p.decode.regime}</span></td>
            <td>{rate(p.inputTokenRate)}</td>
            <td>{rate(p.outputTokenRate)}</td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#each Object.entries($result.perf) as [id, p]}
      {@const cites = citationsFor(p)}
      {#if cites.length > 0}
        <div class="refs">
          <span class="refs-label">References — {id}</span>
          {#if p.asOf || p.notes}
            <div class="meta">
              {#if p.asOf}<span>as of {p.asOf}</span>{/if}
              {#if p.asOf && p.notes}<span class="sep">·</span>{/if}
              {#if p.notes}<span>{p.notes}</span>{/if}
            </div>
          {/if}
          <ol>
            {#each cites as c}
              <li id="ref-{id}-{c.key}" value={c.n}>
                <a href={c.url} target="_blank" rel="noopener noreferrer">{c.title}</a>
              </li>
            {/each}
          </ol>
        </div>
      {/if}
    {/each}
  </section>
{/if}

<style>
  .perf-panel { margin-top: 1rem; }
  table { font-variant-numeric: tabular-nums; border-collapse: collapse; }
  th, td { padding: 0.25rem 0.75rem; text-align: left; border-bottom: 1px solid #eee; }
  .regime { padding: 0.1rem 0.4rem; border-radius: 0.2rem; font-size: 0.85rem; }
  .regime.compute { background: #fde6c8; color: #8a4400; }
  .regime.memory  { background: #c8dcfd; color: #003a8c; }
  .cite a { text-decoration: none; color: #003a8c; }
  .cite a:hover { text-decoration: underline; }
  .refs { margin-top: 0.75rem; font-size: 0.85rem; color: #444; }
  .refs-label { font-weight: 600; color: #222; }
  .refs ol { margin: 0.25rem 0 0; padding-left: 1.5rem; }
  .refs li { margin: 0.1rem 0; }
  .refs a { color: #003a8c; }
  .meta { font-style: italic; color: #666; margin-top: 0.1rem; }
  .meta .sep { margin: 0 0.3rem; }
</style>

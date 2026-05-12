<script lang="ts">
  import { result } from './stores'
  import { SOURCES, type Source } from '../data/sources'
  import type { PerfTier } from '../engine/types'

  // 3 significant figures, no exponential notation.
  function sig3(n: number): string {
    if (n === 0) return '0'
    return parseFloat(n.toPrecision(3)).toString()
  }
  function ms(s: number): string {
    if (s >= 1)     return `${sig3(s)} s`
    if (s >= 1e-3)  return `${sig3(s * 1e3)} ms`
    if (s >= 1e-6)  return `${sig3(s * 1e6)} µs`
    return `${sig3(s * 1e9)} ns`
  }
  function rate(tps: number): string {
    if (tps >= 1e9) return `${sig3(tps / 1e9)} G tok/s`
    if (tps >= 1e6) return `${sig3(tps / 1e6)} M tok/s`
    if (tps >= 1e3) return `${sig3(tps / 1e3)} k tok/s`
    return `${sig3(tps)} tok/s`
  }
  function sameSet(a: string[] | undefined, b: string[] | undefined): boolean {
    if (!a || !b) return false
    if (a.length !== b.length) return false
    const s = new Set(a)
    return b.every(k => s.has(k))
  }

  // Citations are scoped to the operating point that declares them. We compute
  // a per-op-point numbered list (de-duped across the two axes) and produce
  // labeled "TFLOPS" / "Bandwidth" groups — or a single merged "Sources" group
  // when both axes share the same set of citations (the common case).
  function citationsFor(p: PerfTier) {
    const order: string[] = []
    const push = (key: string) => { if (!order.includes(key)) order.push(key) }
    for (const k of p.tflopsSources ?? []) push(k)
    for (const k of p.bandwidthSources ?? []) push(k)
    const refs = order
      .map(key => ({ key, src: SOURCES[key as keyof typeof SOURCES] as Source | undefined }))
      .filter((x): x is { key: string; src: Source } => !!x.src)
      .map((x, i) => ({ key: x.key, n: i + 1, title: x.src.title, url: x.src.url }))
    const numOf = (k: string) => refs.find(r => r.key === k)?.n
    const merged = sameSet(p.tflopsSources, p.bandwidthSources)
    const groups = merged
      ? [{ label: 'Sources', keys: p.tflopsSources ?? [] }]
      : [
          { label: 'TFLOPS', keys: p.tflopsSources ?? [] },
          { label: 'Bandwidth', keys: p.bandwidthSources ?? [] }
        ].filter(g => g.keys.length > 0)
    return { refs, numOf, groups, allMarks: refs.map(r => r.n) }
  }
</script>

{#if $result}
  <section class="perf-panel">
    <h3>Performance</h3>
    <table>
      <thead>
        <tr>
          <th>Operating assumption</th>
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
          {@const c = citationsFor(p)}
          <tr>
            <td>
              {id}
              {#each c.allMarks as n}<sup class="cite"><a href="#ref-{id}-{n}">[{n}]</a></sup>{/each}
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
      {@const c = citationsFor(p)}
      {#if c.refs.length > 0}
        <div class="refs">
          <span class="refs-label">References — {id}</span>
          {#if p.asOf || p.notes}
            <div class="meta">
              {#if p.asOf}<span>as of {p.asOf}</span>{/if}
              {#if p.asOf && p.notes}<span class="sep">·</span>{/if}
              {#if p.notes}<span>{p.notes}</span>{/if}
            </div>
          {/if}
          <div class="groups">
            {#each c.groups as g}
              <span class="group">
                <span class="group-label">{g.label}:</span>
                {#each g.keys as k}
                  {@const n = c.numOf(k)}
                  {#if n !== undefined}<span class="mark">[{n}]</span>{/if}
                {/each}
              </span>
            {/each}
          </div>
          <ol>
            {#each c.refs as r}
              <li id="ref-{id}-{r.n}" value={r.n}>
                <a href={r.url} target="_blank" rel="noopener noreferrer">{r.title}</a>
              </li>
            {/each}
          </ol>
        </div>
      {/if}
    {/each}
  </section>
{/if}

<style>
  .perf-panel { margin-top: 1rem; overflow-x: auto; }
  table {
    font-variant-numeric: tabular-nums; border-collapse: collapse;
    /* Auto layout sizes each column to its content; nowrap below prevents
       awkward wraps inside headers and values. */
  }
  th, td {
    padding: 0.25rem 0.75rem; text-align: left;
    border-bottom: 1px solid #eee; white-space: nowrap;
  }
  th { font-weight: 600; color: #333; }
  /* Numeric value columns: right-align so digits line up. */
  td:nth-child(2), td:nth-child(4), td:nth-child(6), td:nth-child(7) {
    text-align: right; font-variant-numeric: tabular-nums;
  }
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
  .groups { margin-top: 0.25rem; }
  .group { margin-right: 1rem; }
  .group-label { font-weight: 600; color: #222; }
  .mark { margin-left: 0.15rem; color: #003a8c; }
</style>

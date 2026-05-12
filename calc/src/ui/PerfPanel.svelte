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
          <th rowspan="2">Operating assumption</th>
          <th rowspan="2">TTFT</th>
          <th colspan="2" class="group-header">Bottleneck</th>
          <th rowspan="2">Decode time / tok</th>
          <th rowspan="2">Input tok/s</th>
          <th colspan="2" class="group-header">Output tok/s</th>
        </tr>
        <tr>
          <th class="sub-header">Prefill</th>
          <th class="sub-header">Decode</th>
          <th class="sub-header">per stream</th>
          <th class="sub-header">total</th>
        </tr>
      </thead>
      <tbody>
        {#each Object.entries($result.perf) as [id, p]}
          {@const c = citationsFor(p)}
          <tr>
            <td data-label="Operating assumption">
              {id}
              {#each c.allMarks as n}<sup class="cite"><a href="#ref-{id}-{n}">[{n}]</a></sup>{/each}
            </td>
            <td data-label="TTFT">{ms(p.ttftS)}</td>
            <td data-label="Prefill bottleneck"><span class="regime {p.prefill.regime}">{p.prefill.regime}</span></td>
            <td data-label="Decode bottleneck"><span class="regime {p.decode.regime}">{p.decode.regime}</span></td>
            <td data-label="Decode time / tok">{ms(p.decode.timePerTokenS)}</td>
            <td data-label="Input tok/s">{rate(p.inputTokenRate)}</td>
            <td data-label="Output per stream">{rate(1 / p.decode.timePerTokenS)}</td>
            <td data-label="Output total">{rate(p.outputTokenRate)}</td>
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
  .perf-panel { margin-top: 1rem; }
  table { font-variant-numeric: tabular-nums; border-collapse: collapse; }
  th, td {
    padding: 0.2rem 0.5rem; text-align: left;
    border-bottom: 1px solid #eee;
  }
  /* Values stay single-line so numbers don't break mid-unit. */
  td { white-space: nowrap; }
  /* Headers may wrap so the table doesn't need horizontal scrolling. */
  th {
    font-weight: 600; color: #333;
    white-space: normal; vertical-align: bottom;
    line-height: 1.2;
  }
  /* Group header "Bottleneck" sits centered over its two sub-columns. */
  th.group-header {
    text-align: center; border-bottom: 1px solid #ccc;
    padding-bottom: 0.15rem;
  }
  /* Sub-headers (Prefill / Decode) are smaller and less heavy than the
     primary header row. Force right-align — :not(:first-child) above
     wouldn't apply to "Prefill" since it's the first cell in its own row. */
  th.sub-header {
    font-weight: 500; font-size: 0.85em; color: #666;
    padding-top: 0.15rem; text-align: right;
  }
  /* Output columns (everything except the operating-assumption label):
     right-align so numbers and labels stack to a common edge. */
  th:not(:first-child), td:not(:first-child) {
    text-align: right;
  }
  .regime { padding: 0.1rem 0.4rem; border-radius: 0.2rem; font-size: 0.85rem; }
  .regime.compute { background: #fde6c8; color: #8a4400; }
  .regime.memory  { background: #c8dcfd; color: #003a8c; }
  /* On narrow viewports, transpose the table to one card per operating point.
     Each cell becomes a labeled key/value row using its data-label attribute. */
  @media (max-width: 640px) {
    table { display: block; }
    thead { display: none; }
    tbody { display: block; }
    tr {
      display: block;
      border: 1px solid #ddd; border-radius: 0.25rem;
      padding: 0.4rem 0.6rem; margin-bottom: 0.5rem;
      background: #fff;
    }
    td {
      display: flex; justify-content: space-between; align-items: center;
      padding: 0.2rem 0; border-bottom: none; text-align: right;
    }
    td::before {
      content: attr(data-label);
      font-weight: 600; color: #555; text-align: left;
      margin-right: 0.75rem;
    }
    /* First cell (operating assumption identifier) is the card title;
       drop the data-label prefix and emphasize the id text. */
    td:first-child {
      font-weight: 700; font-size: 1rem;
      padding-bottom: 0.35rem; margin-bottom: 0.35rem;
      border-bottom: 1px solid #eee;
      justify-content: flex-start;
    }
    td:first-child::before { content: none; }
  }
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

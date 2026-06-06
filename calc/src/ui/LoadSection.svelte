<script lang="ts">
  import { simInputDisagg, concurrencyOverride, nMaxDecode } from './stores'
  import { loadCurve } from '../engine/queueModel'
  import LoadCharts from './LoadCharts.svelte'

  // Sweep range: 1 to nMaxDecode, capped at 256 sample points so very large
  // nMax doesn't blow the chart with thousands of <path> nodes. Use a uniform
  // stride for now — log stride is overkill at this scale.
  $: nMax = $nMaxDecode
  $: ns = (() => {
    if (nMax <= 0) return []
    const cap = 256
    if (nMax <= cap) return Array.from({ length: nMax }, (_, i) => i + 1)
    const stride = Math.ceil(nMax / cap)
    const out: number[] = []
    for (let n = 1; n <= nMax; n += stride) out.push(n)
    if (out[out.length - 1] !== nMax) out.push(nMax)
    return out
  })()

  // `simInputDisagg` produces a new reference on every slider tick (the slider
  // writes `concurrencyOverride` → `input` recomputes → `simInputDisagg`
  // rebuilds). Its content is identical because `concurrency` is clamped to 1
  // inside, but Svelte refires this derivation anyway. At nMax ≤ 256 the
  // per-tick cost is ~256 cheap arithmetic `computeMemory` calls — sub-ms in
  // practice. If this ever shows up in profiling, factor out a `curveInput`
  // derived that ignores `concurrencyOverride`.
  $: points = ($simInputDisagg && ns.length > 0) ? loadCurve($simInputDisagg, ns) : []

  // Selected N: user's override (if set), else nMaxDecode (= run at the cap).
  // Clamp to nMaxDecode for display — the user's override might be larger
  // (legitimately, for the Calc-tab context); we don't mutate the store.
  $: rawSelected = $concurrencyOverride ?? nMax
  $: selectedN = nMax > 0 ? Math.max(1, Math.min(nMax, rawSelected)) : 1
  $: clamped = ($concurrencyOverride !== null) && ($concurrencyOverride > nMax)

  // When nMax > 256, `ns` is strided and `selectedN` likely won't be a sampled
  // point — fall back to the nearest sampled neighbor. Readout still shows the
  // slider's exact value; KPI shows the nearest sample. Drift is small.
  $: selectedPoint = points.find(p => p.n === selectedN)
    ?? (points.length > 0 ? points.reduce((acc, p) => (Math.abs(p.n - selectedN) < Math.abs(acc.n - selectedN) ? p : acc)) : null)

  function onSliderInput(e: Event) {
    const v = parseInt((e.target as HTMLInputElement).value, 10)
    if (Number.isFinite(v) && v >= 1) {
      concurrencyOverride.set(v)
    }
  }

  function fmt(v: number, unit: string): string {
    if (unit === 's' && v < 1) return `${(v * 1000).toPrecision(3)} ms`
    if (unit === 'tok/s' && v >= 1e3) return `${(v / 1e3).toPrecision(3)} k tok/s`
    return `${v.toPrecision(3)} ${unit}`
  }
</script>

{#if nMax > 0 && points.length > 0 && selectedPoint}
  <div class="load-section">
    <h3 class="section-header">Under load</h3>

    <div class="slider-row">
      <label class="slider-label">
        <span>N (in-flight decode batch)</span>
        <input
          type="range"
          min="1" max={nMax} step="1"
          value={selectedN}
          on:input={onSliderInput}
        />
      </label>
      <div class="readout">
        <strong>{selectedN}</strong> / {nMax}
        {#if clamped}
          <span class="clamped">(override {$concurrencyOverride} clamped to decode-cluster cap)</span>
        {/if}
      </div>
    </div>

    <div class="kpi-row primary">
      <div class="kpi">
        <div class="label">TTFT</div>
        <div class="value">{fmt(selectedPoint.ttftS, 's')}</div>
        <div class="caption">
          {#if selectedPoint.ttftMode === 'overlap'}
            prefill + first decode step on prefill cluster (KV streams in parallel)
          {:else if selectedPoint.ttftMode === 'sequential'}
            prefill + full KV transfer (no overlap)
          {:else}
            prefill only (no disagg overhead)
          {/if}
        </div>
      </div>
      <div class="kpi">
        <div class="label">TPOT</div>
        <div class="value">{fmt(selectedPoint.tpotS, 's')}</div>
        <div class="caption">at N = {selectedN}</div>
      </div>
      <div class="kpi">
        <div class="label">Total latency</div>
        <div class="metric"><span class="m-label">p50</span><span class="m-value">{fmt(selectedPoint.latencyP50S, 's')}</span></div>
        <div class="metric"><span class="m-label">p99</span><span class="m-value">{fmt(selectedPoint.latencyP99S, 's')}</span></div>
        <div class="caption">deterministic v1 — p50 = p99 (single workload)</div>
      </div>
      <div class="kpi">
        <div class="label">Throughput</div>
        <div class="metric"><span class="m-label">Input</span><span class="m-value">{fmt(selectedPoint.inputTokPerS, 'tok/s')}</span></div>
        <div class="metric"><span class="m-label">Output</span><span class="m-value">{fmt(selectedPoint.throughputTokS, 'tok/s')}</span></div>
        <div class="metric"><span class="m-label">Req</span><span class="m-value">{selectedPoint.throughputReqS.toPrecision(3)} req/s</span></div>
      </div>
    </div>

    <div class="kpi-row disagg">
      <div class="kpi">
        <div class="label">Prefill (per device)</div>
        <div class="value">{fmt(selectedPoint.prefillInputTokPerSPerDevice, 'tok/s')}</div>
        <div class="caption">
          × {selectedPoint.prefillDevices} = {fmt(selectedPoint.prefillInputTokPerSPerDevice * selectedPoint.prefillDevices, 'tok/s')} input
        </div>
      </div>
      <div class="kpi">
        <div class="label">Decode (per device)</div>
        <div class="value">{fmt(selectedPoint.decodeOutputTokPerSPerDevice, 'tok/s')}</div>
        <div class="caption">
          × {selectedPoint.decodeDevices} = {fmt(selectedPoint.decodeOutputTokPerSPerDevice * selectedPoint.decodeDevices, 'tok/s')} output
        </div>
      </div>
      <div class="kpi pd">
        <div class="label">P:D instance ratio</div>
        <div class="value">{selectedPoint.pdRatio.toPrecision(3)}</div>
        <div class="caption">
          {#if selectedPoint.pdRatio > 1}
            prefill-bound: need {selectedPoint.pdRatio.toPrecision(3)} prefill nodes per decode node
          {:else}
            decode-bound: {selectedPoint.pdRatio.toPrecision(3)} prefill nodes per decode node sustain the batch
          {/if}
        </div>
      </div>
    </div>

    <LoadCharts {points} {selectedN} {nMax} />
  </div>
{:else if nMax === 0}
  <div class="load-section">
    <h3 class="section-header">Under load</h3>
    <div class="oom-hint">
      Decode cluster can't fit any in-flight requests at this configuration
      (weights alone exceed HBM, or per-request KV overhead does after weights).
      Pick a larger decode SKU or add parallelism on the decode cluster.
    </div>
  </div>
{/if}

<style>
  .load-section { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.75rem; }
  .section-header { margin: 0; font-size: 1rem; font-weight: 600; color: #333; }
  .slider-row { display: flex; flex-direction: row; align-items: center; gap: 1rem; }
  .slider-label { display: flex; flex-direction: column; gap: 0.3rem; flex: 1; font-size: 0.85rem; color: #555; }
  .slider-label input[type=range] { width: 100%; }
  .readout { font-size: 0.95rem; color: #333; min-width: 8rem; }
  .readout strong { font-size: 1.2rem; }
  .clamped { display: block; font-size: 0.75rem; color: #8a3f00; font-style: italic; }
  .kpi-row.primary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; }
  .kpi-row.disagg  { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 0.5rem; }
  .kpi {
    padding: 0.6rem 0.9rem; background: #fff;
    border: 1px solid #d4d4d4; border-radius: 0.4rem;
  }
  .kpi .label {
    font-size: 0.8rem; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .kpi .value { font-size: 1.4rem; font-weight: 700; color: #222; margin-top: 0.2rem; }
  .kpi .caption { font-size: 0.78rem; color: #666; margin-top: 0.3rem; }
  .kpi .metric {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-top: 0.3rem; font-size: 0.95rem;
  }
  .kpi .m-label { color: #666; font-size: 0.85rem; }
  .kpi .m-value { font-weight: 700; color: #222; }
  .oom-hint {
    padding: 0.7rem 0.9rem;
    background: #fff7ec; color: #8a3f00;
    border: 1px solid #f0c890; border-radius: 0.3rem;
    font-size: 0.9rem; line-height: 1.4;
  }
  @media (max-width: 700px) {
    .kpi-row.primary, .kpi-row.disagg { grid-template-columns: 1fr; }
  }
</style>

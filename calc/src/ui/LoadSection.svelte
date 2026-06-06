<script lang="ts">
  import { simInputDisagg, concurrencyOverride, nMaxDecode } from './stores'
  import { loadCurve } from '../engine/queueModel'
  import LoadCharts from './LoadCharts.svelte'

  // Sweep range: 1 to nMaxDecode, capped at 256 sample points so very large
  // nMax doesn't blow the chart with thousands of <path> nodes.
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

  // `simInputDisagg` produces a new reference on every slider tick — sub-ms
  // at nMax ≤ 256, acceptable until profiling says otherwise.
  $: points = ($simInputDisagg && ns.length > 0) ? loadCurve($simInputDisagg, ns) : []

  // Clamp to nMaxDecode for display; don't mutate the store.
  $: rawSelected = $concurrencyOverride ?? nMax
  $: selectedN = nMax > 0 ? Math.max(1, Math.min(nMax, rawSelected)) : 1
  $: clamped = ($concurrencyOverride !== null) && ($concurrencyOverride > nMax)

  // When nMax > 256, ns is strided — fall back to nearest sampled neighbor.
  $: selectedPoint = points.find(p => p.n === selectedN)
    ?? (points.length > 0 ? points.reduce((acc, p) => (Math.abs(p.n - selectedN) < Math.abs(acc.n - selectedN) ? p : acc)) : null)

  // Saturation N: smallest sampled N where throughputReqS first reaches 95% of
  // the curve's max. Below that, adding concurrency still buys real throughput;
  // beyond, the curve plateaus.
  $: saturationN = (() => {
    if (points.length === 0) return null
    const maxReq = Math.max(...points.map(p => p.throughputReqS))
    if (maxReq <= 0) return null
    const sat = points.find(p => p.throughputReqS >= 0.95 * maxReq)
    return sat?.n ?? null
  })()

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

    <div class="top-row">
      <div class="slider-col">
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
        {#if saturationN !== null}
          <div class="saturation-hint">Throughput saturates around N = {saturationN}</div>
        {/if}
      </div>
      <div class="chart-col">
        <LoadCharts {points} {selectedPoint} {nMax} />
      </div>
    </div>

    <div class="kpi-row latency">
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
        <div class="value">{fmt(selectedPoint.latencyS, 's')}</div>
        <div class="caption">deterministic v1 (uniform arrivals, identical workload)</div>
      </div>
    </div>

    <div class="kpi-row throughput">
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
      <div class="kpi">
        <div class="label">Aggregate throughput</div>
        <div class="tp-row"><span class="tp-label">Input</span><span class="tp-value">{fmt(selectedPoint.inputTokPerS, 'tok/s')}</span></div>
        <div class="tp-row"><span class="tp-label">Output</span><span class="tp-value">{fmt(selectedPoint.throughputTokS, 'tok/s')}</span></div>
        <div class="tp-row"><span class="tp-label">Req</span><span class="tp-value">{selectedPoint.throughputReqS.toPrecision(3)} req/s</span></div>
      </div>
    </div>

    <div class="pd-ratio-text">
      <strong>P:D instance ratio at N={selectedN}:</strong> {selectedPoint.pdRatio.toPrecision(3)} —
      {#if selectedPoint.pdRatio > 1}
        prefill-bound: need {selectedPoint.pdRatio.toPrecision(3)} prefill nodes per decode node
      {:else}
        decode-bound: {selectedPoint.pdRatio.toPrecision(3)} prefill nodes per decode node sustain the batch
      {/if}
    </div>
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

  .top-row {
    display: grid; grid-template-columns: 1fr 2fr; gap: 1rem;
    align-items: stretch;
    padding: 0.6rem 0.9rem;
    border: 1px solid #d4d4d4; border-radius: 0.4rem; background: #fff;
  }
  .slider-col {
    display: flex; flex-direction: column; justify-content: center;
  }
  .slider-col .slider-label { display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.85rem; color: #555; }
  .slider-col .slider-label input[type=range] { width: 100%; }
  .slider-col .readout { margin-top: 0.6rem; font-size: 0.95rem; color: #333; }
  .slider-col .readout strong { font-size: 1.4rem; }
  .slider-col .clamped { display: block; font-size: 0.75rem; color: #8a3f00; font-style: italic; margin-top: 0.3rem; }
  .slider-col .saturation-hint { margin-top: 0.5rem; font-size: 0.82rem; color: #555; font-style: italic; }
  .chart-col {
    display: flex; align-items: stretch;
  }

  .kpi-row.latency, .kpi-row.throughput {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;
    margin-top: 0.75rem;
  }

  .kpi {
    border: 1px solid #d4d4d4; border-radius: 0.4rem; padding: 0.8rem 1rem;
    background: #fff;
  }
  .kpi .label {
    font-size: 1.1rem; font-weight: 600; letter-spacing: 0.02em; color: #888;
  }
  .kpi .value {
    font-size: 1.5rem; font-weight: 700; line-height: 1.1; margin-top: 0.1rem;
  }
  .kpi .caption {
    font-size: 0.78rem; color: #666; margin-top: 0.3rem;
  }
  .tp-row {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-top: 0.25rem;
  }
  .tp-label { font-size: 0.85rem; color: #666; }
  .tp-value { font-size: 0.95rem; font-weight: 700; color: #222; }

  .pd-ratio-text {
    margin-top: 0.75rem; padding: 0.5rem 0.9rem;
    font-size: 0.9rem; color: #333;
    border-left: 3px solid #6b46c1;
    background: #f7f5fb;
  }

  .oom-hint {
    padding: 0.7rem 0.9rem;
    background: #fff7ec; color: #8a3f00;
    border: 1px solid #f0c890; border-radius: 0.3rem;
    font-size: 0.9rem; line-height: 1.4;
  }

  @media (max-width: 800px) {
    .top-row { grid-template-columns: 1fr; }
    .kpi-row.latency, .kpi-row.throughput { grid-template-columns: 1fr; }
  }
</style>

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

  $: points = ($simInputDisagg && ns.length > 0) ? loadCurve($simInputDisagg, ns) : []

  // Selected N: user's override (if set), else nMaxDecode (= run at the cap).
  // Clamp to nMaxDecode for display — the user's override might be larger
  // (legitimately, for the Calc-tab context); we don't mutate the store.
  $: rawSelected = $concurrencyOverride ?? nMax
  $: selectedN = nMax > 0 ? Math.max(1, Math.min(nMax, rawSelected)) : 1
  $: clamped = ($concurrencyOverride !== null) && ($concurrencyOverride > nMax)

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

    <div class="kpi-row">
      <div class="kpi">
        <div class="label">Aggregate throughput</div>
        <div class="value">{fmt(selectedPoint.throughputTokS, 'tok/s')}</div>
        <div class="caption">{selectedPoint.throughputReqS.toPrecision(3)} req/s</div>
      </div>
      <div class="kpi">
        <div class="label">Per-request total</div>
        <div class="value">{fmt(selectedPoint.totalS, 's')}</div>
        <div class="caption">TPOT {fmt(selectedPoint.tpotS, 's')}</div>
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
  .kpi-row { display: grid; grid-template-columns: 1fr 1fr 1.5fr; gap: 0.75rem; }
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
  .oom-hint {
    padding: 0.7rem 0.9rem;
    background: #fff7ec; color: #8a3f00;
    border: 1px solid #f0c890; border-radius: 0.3rem;
    font-size: 0.9rem; line-height: 1.4;
  }
  @media (max-width: 700px) {
    .kpi-row { grid-template-columns: 1fr; }
  }
</style>

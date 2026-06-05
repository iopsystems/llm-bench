# Disagg Load Section — Design

## Goal

Add an "Under load (disagg)" section to the Simulator tab that shows how a disagg deployment behaves as the decode cluster runs a continuous batch of N in-flight requests. Two side-by-side charts (throughput-vs-N, latency-vs-N) sweep N from 1 to N_max (the KV-cap), with an operating-point slider that highlights a chosen N and reports per-request KPIs plus the prefill-to-decode instance ratio needed to keep the decode batch fed.

Out of scope for v1: monolithic load curves (entangled with chunked-prefill / scheduling choices — needs more thought), variable workload distributions (identical requests only), discrete-event sim (deterministic model is enough for identical requests).

## Why N, not λ

Arrival rate is mathematically natural for queueing theory but cognitively expensive: "what's my system's λ?" is rarely a question an operator can answer. In-flight count N is what they actually pick — it's the knob that directly determines (a) KV cache HBM occupancy, (b) the continuous-batch size the decoder runs with, (c) per-token latency via memory-bandwidth pressure on the decode pass. The model is closed-loop by construction: prefill keeps the decode batch full, decode services N requests per step, throughput = N / (output_tokens × tpot(N)).

## Architecture

### Module layout

```
src/engine/queueModel.ts        # pure functions, no UI/store dependencies
src/ui/LoadCharts.svelte        # SVG, two side-by-side charts
src/ui/LoadSection.svelte       # InputPanel-style N slider + KPI block + LoadCharts
src/ui/Simulator.svelte         # mount LoadSection below the disagg block
src/ui/stores.ts                # concurrencyOverride + effectiveConcurrency
```

Charts use inline SVG (consistent with `simulatorGantt.ts` / `SimulatorGantt.svelte`). No new charting dep.

### Data flow

`LoadSection` reads `simInputDisagg` (the existing disagg-side CalcInput, already correctly resolves het=on / off). It computes:

1. `nMaxDecode = computeNMax(simInputDisagg)` — KV-cap ceiling for the decode cluster (heterogeneity-aware)
2. `points = loadCurve(simInputDisagg, range(1, nMaxDecode))` — one LoadPoint per N
3. The user picks an operating N via a slider; the matching `LoadPoint` drives the KPI block

**Slider semantics, and the relationship to Calc-tab concurrency.** The slider and the Calc-tab concurrency textbox both bind to the same `concurrencyOverride` store, so a value the user picks in one place persists when they switch tabs. They show different *defaults* though, because they're computed against different contexts:

- Calc tab default: `nMaxCalc = computeNMax(input)` — derived from the shared (monolithic-style) hw the Calc tab is sizing for.
- LoadSection slider default: `nMaxDecode = computeNMax(simInputDisagg)` — derived from the disagg decode cluster, which can differ under het=on.

When `concurrencyOverride === null`, each context displays its own default. When the override is set, both honor it. If the override exceeds `nMaxDecode` in the LoadSection context (decode cluster has less HBM than the Calc-tab hw), the LoadSection clamps its slider position visually to `nMaxDecode` and shows a "decode cluster limits N to {nMaxDecode}" badge without mutating the override.

The Sim tab's `simInputMonolithic` / `simInputDisagg` continue to clamp `workload.concurrency` to 1 (single-request gantt is unaffected). The LoadSection internally re-introduces N by passing it explicitly to `loadCurve(simInputDisagg, [N])`.

## Engine: `src/engine/queueModel.ts`

Pure module — no Svelte, no DOM, no stores. Importable from CLI.

### `computeNMax(input: CalcInput): { nMax: number, boundBy: 'kv' | 'weights' }`

KV-cap derivation. For the decode cluster (uses `decodeAccelerator` / `decodeMultiDevice` when present, else falls back to prefill — same as `computeMemory`):

```
hbmFreePerRank  = hbmCapacityPerRank - weightsPerRank
perReqKvPerRank = kvCachePerRequest / replicas  (replicas accounts for parallelism)
perReqActPerRank = decodeActivationsPerRequestPerRank   (computed at N=1, then scaled by N)
nMax = floor(hbmFreePerRank / (perReqKvPerRank + perReqActPerRank))
```

`boundBy = 'weights'` when `hbmFreePerRank <= 0` (model itself doesn't fit; nMax is 0). Otherwise `boundBy = 'kv'`.

### `loadCurve(input: CalcInput, ns: number[]): LoadPoint[]`

For each N, runs the existing engine primitives with concurrency=N on the decode cluster:

```ts
interface LoadPoint {
  n: number
  tpotS: number              // decode step time at batch size N
  prefillS: number           // prefill latency on prefill cluster (independent of N)
  kvTransferS: number        // shipping cost between clusters
  totalS: number             // prefill + kv + output_tokens × tpot(N) — per-request latency
  throughputTokS: number     // aggregate output tokens/s, bottleneck-bound (see below)
  throughputReqS: number     // aggregate req/s, bottleneck-bound
  pdRatio: number            // prefill instances per decode instance to keep batch fed
}
```

`prefillS` is computed once (prefill cluster runs serial, single-request prefill at the configured promptTokens). `tpotS(N)` reuses `computeDecode` with `workload.concurrency = N` — already exists, no new math needed. `loadCurve` internally constructs a per-iteration `CalcInput` with `workload.concurrency` overridden to `N` (the input passed in carries the simInputDisagg's clamp of 1; loadCurve unconditionally replaces it for each point on the sweep). `kvTransferS` reuses today's per-request fabric transfer cost.

**Why prefill runs serially.** Prefill arithmetic intensity is `2 × tokens / bytes_per_weight` FLOPs/byte (the weights load is amortized across all tokens in the batch — concretely, an H200 hits its bf16 roofline crossover at ~200 tokens, fp8 at ~100). Above the crossover a single request already saturates the tensor cores, so batching multiple prefills together would not reduce wall time. Below the crossover the cluster is HBM-bound on the weight load and batching short prefills would amortize that bandwidth — but typical chat / code-gen prompts sit well above the crossover, so the v1 model assumes prefill is compute-bound and a single serial slot is the right abstraction. If we ever want to study sub-200-token regimes (autocomplete, classifiers), prefill batching becomes a knob — track in v2.

Throughput is bottleneck-bounded across the two stations:
- decode-side request rate = `N / (output_tokens × tpotS(N))` (N requests in batch, each takes that long to drain)
- prefill-side request rate = `1 / prefillS` (single serial prefill slot)
- `throughputReqS = min(decode-rate, prefill-rate)`
- `throughputTokS = throughputReqS × output_tokens`

`pdRatio(input, n) = (n * prefillS) / (output_tokens * tpotS(n))`. Reading: ratio > 1 means prefill is the bottleneck at this N (need more prefill nodes); ratio < 1 means decode is the bottleneck (typical at large N).

`ns` is chosen by the caller. `LoadSection` passes `[1, 2, …, nMax]` (or a log-stride sample if nMax is large, capped at ~256 chart points).

### Edge cases

- `nMax === 0` (weights bigger than HBM): `loadCurve` returns []; UI shows "Decode side OOM at any N — pick a larger SKU or add parallelism."
- `prefillS > output_tokens × tpotS(n)` at the chosen N: prefill is the bottleneck, decode batch cannot stay full at this N. v1 still reports the KPIs as if it could; the `pdRatio` makes the constraint visible (ratio > 1 means more prefill instances needed). v2 might add a regime marker.

## Stores: `src/ui/stores.ts`

Add:

```ts
export const concurrencyOverride: Writable<number | null> = writable(null)

export const nMaxCalc: Readable<number> = derived(
  [input], ([$input]) => $input ? computeNMax($input).nMax : 0
)

export const nMaxDecode: Readable<number> = derived(
  [simInputDisagg], ([$d]) => $d ? computeNMax($d).nMax : 0
)

// Effective concurrency for Calc-tab consumers (memory panel, throughput).
export const effectiveConcurrency: Readable<number> = derived(
  [concurrencyOverride, nMaxCalc],
  ([$override, $nMax]) => $override ?? Math.max(1, $nMax)
)
```

Update `workload` consumers: today the `workload` store carries `concurrency: 1` literally. Change to read from `effectiveConcurrency` where the engine consumes it. Cleanest path: introduce a derived `effectiveWorkload: Workload = { ...workload, concurrency: effectiveConcurrency }`, and rewire `input` to use it instead of `workload`. The Sim tab's `simInputMonolithic` / `simInputDisagg` already clamp concurrency to 1, so they're unaffected.

Initial `concurrencyOverride` is null. The Calc-tab input shows `nMaxCalc` as the rendered value when override is null, the user's number when override is set. Clearing the input reverts to null → tracks `nMaxCalc`. The LoadSection slider works analogously against `nMaxDecode`, with the clamp-display behavior described in the architecture section.

## URL state: `src/ui/share.ts`

`c=` continues to encode the *override*, not the effective value:

- override === null → omit `c=` from URL
- override === N → emit `c=N`

A URL without `c=` decodes to `concurrencyOverride = null` (recipient sees their own nMax). A URL with `c=N` decodes to `concurrencyOverride = N` (sticky to the shared value).

Backward compat: old URLs with `c=1` continue to set override=1. No behavioral change for existing share links.

## UI: `src/ui/LoadSection.svelte`

```
┌─ Under load ─────────────────────────────────────────┐
│  N (in-flight decode batch):  [████████░░░░] 18 / 32 │  (slider)
│                                                       │
│  ┌──── Aggregate throughput ────┬── Per-request ────┐│
│  │   12.4 k tok/s               │  Total:   3.2 s   ││
│  │   0.30 req/s                 │  TTFT:    180 ms  ││
│  │                              │  TPOT:    16.5 ms ││
│  │  P:D ratio at N=18: 0.45     │                   ││
│  │  (need 0.45 prefill nodes    │                   ││
│  │   per decode node to feed)   │                   ││
│  └──────────────────────────────┴───────────────────┘│
│                                                       │
│  ┌── Throughput (tok/s) ──┐  ┌── Latency (s) ──────┐ │
│  │      ▁▂▃▄▅▆▇▇▇▇▇▇      │  │            ▁▂▄▇   │ │
│  │   ●                     │  │      ▁▂▃▄▅▇        │ │
│  │ 1 ── N ────── 32        │  │ 1 ── N ────── 32   │ │
│  └─────────────────────────┘  └────────────────────┘ │
│       N_max = 32 (KV-bound)                           │
└───────────────────────────────────────────────────────┘
```

Slider: 1 to nMax, integer steps. Default value = nMax. Updating the slider re-highlights the dot on both charts.

KPI block: 4 fields on the left (throughput tokens/s, throughput req/s, P:D ratio, P:D explanation tooltip text), 3 fields on the right (Total, TTFT, TPOT). Uses the same `kpi` / `op` CSS classes as `Simulator.svelte` for visual consistency.

Charts: each is ~280px wide × 140px tall. X-axis: N from 1 to nMax. Y-axis: linear, auto-scaled per chart. The user's operating N is a filled circle on both curves with a vertical guide line.

Gating: section is gated by `$disaggKvTransferFabricId !== ''` AND `simResultDisagg !== null` (no fabric → nothing to render; disagg error → the existing error block above handles messaging).

## Mount in Simulator.svelte

Below the existing disagg-block render path, inside the `{#if $disaggKvTransferFabricId}` branch, after the `{:else if rowsDisagg.length > 0}` block. Same gating means the section only appears when the disagg single-request block is also visible.

## Calc tab: Concurrency input

`InputPanel.svelte` already renders a concurrency input. Change:

- Bound value reads from `effectiveConcurrency` (derived), writes go to `concurrencyOverride`
- Placeholder text when override is null: shows the nMax value lightly
- When user types a number, `concurrencyOverride.set(n)`
- When user clears the field (or types 0), `concurrencyOverride.set(null)` → tracks nMax

This requires no other Calc-tab changes. The Memory panel and Throughput panels already render whatever concurrency is in `input` — they'll see the new effective value.

## Testing

### `test/engine/queueModel.test.ts`

1. `computeNMax(h100 + llama-3.3-70b @ bf16 + 2k/512 workload)` returns a positive integer; recomputed by hand matches.
2. `computeNMax(model > HBM)` returns `{ nMax: 0, boundBy: 'weights' }`.
3. `loadCurve(input, [1])` returns a single LoadPoint whose `tpotS` / `prefillS` / `totalS` match `calculate(input).perf[opPointId]` for the same op-point — invariant: N=1 closed-loop = single-request.
4. `loadCurve(input, [1, 2, 4, 8])` shows monotonically non-decreasing `tpotS` (more KV per step = more memory traffic).
5. `pdRatio` formula spot-check: contrived input where N × prefillS = output_tokens × tpotS → ratio = 1.

### `test/ui/sim-load-stores.test.ts`

1. Default `concurrencyOverride = null`; `effectiveConcurrency = nMaxCalc`.
2. Setting `concurrencyOverride` to N changes `effectiveConcurrency` to N.
3. Changing `acceleratorId` re-derives `nMaxCalc`; if override is null, `effectiveConcurrency` follows; if override is set, `effectiveConcurrency` stays at the override.
4. `effectiveConcurrency` floors at 1 (even when `nMaxCalc = 0`, so the engine isn't fed concurrency=0).
5. Under het=on with a smaller-HBM decode cluster, `nMaxDecode < nMaxCalc`. Override at `nMaxCalc` leaves the effective value at `nMaxCalc` for Calc but the LoadSection clamps its slider visually.

### `test/ui/share.test.ts` / `test/ui/share-route.test.ts`

1. Encoding state with `concurrencyOverride = null` omits `c=` from URL.
2. Encoding state with `concurrencyOverride = 5` emits `c=5`.
3. Decoding URL without `c=` leaves `concurrencyOverride` undefined (recipient defaults to null → tracks nMax).
4. Decoding URL with `c=5` sets `concurrencyOverride = 5`.
5. Backward compat: old `c=1` URLs still set override = 1.

### Manual verification

1. Open Simulator, no fabric → no load section visible.
2. Pick fabric → load section appears with slider at nMax, both charts populated.
3. Drag slider → KPI block + chart markers update live.
4. Change decode hw (in heterogeneous mode) → nMax recomputes, slider clamps, charts re-render.
5. Open Calc tab → concurrency field shows nMax by default; type a number → input is sticky; clear field → reverts to nMax.

## Open questions deferred to v2

- Monolithic load curves (entangled with scheduler choices)
- Variable / sampled workload distributions (would re-introduce the percentile sim)
- P:D ratio as a third small chart (vs. just the KPI line)
- Past-N_max behavior: today the slider caps at N_max; could add a "what if we had more HBM?" extrapolation
- Prefill batching for short-prompt regimes (< ~200 tokens, where prefill is HBM-bound on weight load) — batched prefill amortizes the weight read across requests; today's serial assumption is wrong here. Would need a batch-size knob and revisit `prefillS` to scale with the per-batch token total instead of one request's promptTokens.

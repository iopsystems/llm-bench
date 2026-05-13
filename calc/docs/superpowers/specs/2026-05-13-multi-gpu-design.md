# Multi-Accelerator Support — Design

**Status:** Draft, awaiting user review
**Date:** 2026-05-13
**Scope:** Wire the existing `SYSTEMS` / `INTERCONNECTS` / `PARALLELISM_MODES` data registries into the calculation engine and UI. UI presents multi-accelerator systems as first-class entries in the accelerator dropdown ("HGX H100", "GB200 NVL72", ...). Engine decomposes the selected system into the underlying chip × count × interconnect × parallelism mode and adds a per-parallelism-mode communications cost to the roofline as a third ceiling.

This is the project's first non-single-accelerator feature and the largest UI / engine change in the calc to date.

## Motivation

The single-accelerator calc answers "what fits, how fast, on one chip." Production inference deployments overwhelmingly run multi-accelerator: TP=8 across an HGX baseboard, TP × EP across GB200 NVL72, TP × DP across multiple H100 nodes. The calc currently can't represent any of this.

The data registries that describe the universe were landed earlier this session:
- `INTERCONNECTS` (PR #102) — 13 fabric types: NVLink generations, NVSwitch, InfiniBand, Infinity Fabric, ICI, NeuronLink, etc.
- `PARALLELISM_MODES` (PR #102) — TP, PP, EP, DP, CP, SP with collective and volume formula metadata
- `SYSTEMS` (PR #103) — 10 named products: HGX H100/H200/B200, GB200 NVL72, MI300X / MI325X 8-OAM nodes, Gaudi3 HLS, TPU v5p/Trillium 8-chip, AWS Trn2 48xl

The data is there. This feature wires it through the engine and surfaces it in the UI.

## UI: how it appears to the user

### Accelerator dropdown unification

Today the dropdown lists single accelerators (`H100 SXM-80`, `B200`, `MI300X`, ...) grouped by vendor. After this feature, the dropdown's union includes named multi-accelerator systems, intermixed and grouped by vendor:

```
NVIDIA
  H100 SXM-80                       ← single
  H100 SXM-94
  HGX H100 (8×, NVLink-4)           ← multi-system
  HGX H200 (8×, NVLink-4)           ← multi-system
  HGX B200 (8×, NVLink-5)           ← multi-system
  GB200 NVL72 (72×, NVLink-5)       ← multi-system
  B200 SXM-180                      ← single
  ...
AMD
  MI300X                            ← single
  MI300X 8× Node (8×, Infinity Fabric) ← multi-system
  MI325X 8× Node                    ← multi-system
  ...
```

Single accelerators and multi-systems share a vendor section but are visually distinguished — multi-system entries show the count and fabric inline (`HGX H100 (8×, NVLink-4)`), single accelerators don't.

### What appears when a multi-system is selected

When `multiDevice` becomes populated in `CalcInput`, the input panel reveals three things:

1. **Parallelism mode picker** — dropdown showing the auto-default for `(system, model)` with a chevron to override. Shows the active mode tuple (`TP=8`, `TP=8 × EP=9`, etc.) inline. The dropdown lists all combinations valid for the system (count + which modes can compose).
2. **Concurrency tooltip change** — "Concurrency" label gets a tooltip clarifying it's *cluster-total request concurrency* (the input to the system funnel), and the engine internally derives per-replica batch size from `concurrency / DP` (DP=1 unless data-parallel is active).
3. **Memory bar mode** — switches from "single-GPU memory" to "per-GPU memory" semantics. Same bar shape (weights / kv / activations), values divided per-rank by the active parallelism mode's sharding. Cluster aggregate shown as a small label below the bar (e.g., "8 × 92 GB = 736 GB cluster").

When a plain single accelerator is selected, none of these appear (today's UI).

### Roofline plot — hierarchical extension

Today's plot has two ceilings (compute, HBM-bandwidth) and a workload marker. After this feature, when `multiDevice` is active, a **third ceiling** appears: the interconnect-bandwidth slope, shallower than HBM (interconnect BW is typically an order of magnitude below HBM).

The workload marker's regime — today `'compute' | 'memory'` — gains a third value `'comms'`. The decode panel's regime pill displays whichever ceiling is binding. At long context with TP, expect `'comms'` to dominate decode — that's the diagnostic this feature surfaces.

## Engine: schema additions

### `CalcInput` gains an optional `multiDevice` axis

```ts
interface CalcInput {
  // ...existing
  multiDevice?: {
    system: MultiAcceleratorSystem            // resolved from the dropdown pick
    parallelism: ParallelismMode['id'][]       // e.g. ['tp'] or ['tp', 'ep'] or ['tp', 'pp']
    parallelismDegrees: Record<ParallelismMode['id'], number>  // e.g. { tp: 8, ep: 9 }
  }
}
```

When `multiDevice` is undefined: existing single-accelerator behavior (no change for any current code path).

When populated: the engine reads:
- `system.accelerator` to resolve the underlying chip + variant
- `system.interconnectId` to look up `INTERCONNECTS`
- `parallelism` + `parallelismDegrees` to compute comms term + per-rank memory split

The UI translates the dropdown selection into this shape: picking "HGX H100" sets `system` to that entry and `parallelism` / `parallelismDegrees` to the heuristic default (below).

### Default parallelism heuristic

When the user picks a multi-system, the UI auto-fills parallelism with the following rule (user can always override via the dropdown):

```
N = system.accelerator.count

if model is dense:
  TP = min(N, 8)
  if N > 8: also PP = N / 8
else (MoE):
  TP = min(N, 8)
  EP = N                        // experts distributed across all accelerators
  if N > 8: also PP = ⌈N / 8⌉
```

Examples:
- HGX H100 (N=8) + Llama 70B (dense) → `TP=8`
- HGX H100 (N=8) + Mixtral 8x7B (MoE) → `TP=8, EP=8`
- GB200 NVL72 (N=72) + DeepSeek V3 (MoE) → `TP=8, EP=72, PP=9`
- MI300X-8 (N=8) + V3 → `TP=8, EP=8`

These defaults are *heuristic guidance*. Production deployments tune differently per workload; the override exposes the choice.

## Engine: math additions

### Per-rank memory split

Per-parallelism-mode rules:

| Mode | Weights | KV cache | Activations |
|---|---|---|---|
| **TP** | `/N` (sharded by head/hidden dim) | `/min(N, numKvHeads)` (sharded if heads ≥ N) | `/N` |
| **PP** | `/N` (each stage holds `L/N` layers) | `× (L/N) / L` (each stage holds its own layers' KV) | full (per-stage forward) |
| **EP** | non-expert weights replicated, expert weights `/N` | replicated (each rank computes its tokens' attention) | replicated |
| **DP** | replicated | per-replica concurrency `× kv_per_request` | per-replica |
| **CP** | replicated | sequence-split `/N` | sequence-split `/N` |
| **SP** | replicated | (composed with TP; same as TP for memory) | sequence-split `/N` |

For composed modes (e.g., TP × DP): apply sequentially. `weights_per_rank = total / (TP × DP)` for the TP+DP case.

`MemoryResult` gains `perRank: { weights, kvCachePerRequest, activationsPeak, total, fits }` alongside the existing aggregate fields. The aggregate fields keep their cluster-total semantics.

### Comms term per parallelism mode

Per-step communications volume (bytes per forward pass) summed across modes:

- **TP**: `2 × layers × 2 × (N-1)/N × B × hidden × bytes(activations)` (two all-reduces per layer, ring algorithm)
- **PP**: `(N-1) × B × hidden × bytes(activations)` per microbatch (point-to-point at each stage boundary)
- **EP**: `2 × layers_with_moe × (1 - 1/N) × B × hidden × bytes(activations)` (all-to-all per MoE layer, forward gather + scatter)
- **CP**: `2 × layers × B × hidden × bytes(activations) / N` (per-layer activation exchange)
- **DP**: 0 (inference; gradients don't apply)
- **SP**: composes with TP at zero additional volume (decomposed but same total)

`B` = effective batch in the layer. For TP/PP: B = per-replica concurrency (= total concurrency / DP). For decode: B = concurrency (each request contributes one token per step). For prefill: B = prompt × concurrency.

`commsBytesPerStep` = sum over active modes.

### Hierarchical roofline

The existing `roofline()` function takes the max of `flops/tflops` and `bytes/bw`. Extend it:

```ts
function roofline({
  flops, bytes, tflops, bwGBs,
  commsBytes, interconnectBwGBs  // NEW — optional; if either absent, comms contributes 0
}: RooflineInputs): { timeS, regime }

time_per_step = max(
  flopsPerStep / tflops,                  // compute bound (existing)
  hbmBytesPerStep / hbmBW,                // HBM bound (existing)
  commsBytesPerStep / interconnectBW      // NEW: interconnect bound
)
regime ∈ 'compute' | 'memory' | 'comms'   // determined by which max wins
```

`commsBytes` / `interconnectBwGBs` are undefined for single-accelerator calls; the new ceiling contributes 0 to the max in that case. So existing tests stay byte-for-byte identical.

### `interconnectBW` per system

`system.interconnectId` → `INTERCONNECTS[id]`. The relevant bandwidth for ring all-reduce is `perDirectionGBs` (= `perGpuBandwidthGBs / 2` if `perDirectionGBs` is unset). For non-ring topologies the math may differ; first cut uses the conservative ring formula.

### Concurrency semantics

`workload.concurrency` is **cluster-total** — requests fed into the system at the top of the funnel. Engine computes per-replica batch size as `concurrency / DP`. For non-DP modes (TP, PP, EP, ...), one replica handles the whole funnel, so per-replica = total.

This matches how operators reason about throughput ("we get 1000 qps; can the system handle it?") rather than how internal engines reason about microbatches.

## Data

No new data — uses existing `SYSTEMS`, `INTERCONNECTS`, `PARALLELISM_MODES` registries that PR #102/#103 landed.

Initial UI dropdown population: all 10 systems in `SYSTEMS`:
- HGX H100 (8×) / HGX H200 (8×) / HGX B200 (8×) / GB200 NVL72 (72×)
- MI300X 8× / MI325X 8×
- Gaudi3 HLS (8×)
- TPU v5p-8 / TPU Trillium-8
- AWS Trn2 48xl (16×)

A `DGX H100 SuperPOD (256×)` and `B200 NVL576` could be added as data-only follow-ups once the engine math is validated on the smaller scale-up systems.

## Testing

### Math layer

- **Per-rank memory split**: synthetic dense model with TP=8 → weights / 8, kv / 8, activations / 8 (all sharded).
- **DP split**: synthetic model with DP=2, concurrency=8 → per-replica concurrency = 4; per-replica weights replicated.
- **EP split for MoE**: expert weights / N, non-expert weights replicated.
- **PP split**: weights × (1/N) (per-stage), KV cache × (1/N) (per-stage).
- **Composed**: TP=8 × DP=2 = 16 ranks total; per-rank weights = total / 16.
- **Layer-count invariant**: parallelism `pp > model.layers` throws (PP requires at least one layer per stage).

### Comms layer

- **TP all-reduce volume**: per-step `2 × layers × 2 × (N-1)/N × B × hidden × bytes` for synthetic model.
- **PP boundary volume**: `(N-1) × B × hidden × bytes` per microbatch.
- **EP all-to-all volume**: `2 × moe_layers × (1 - 1/N) × B × hidden × bytes`.
- **Composed**: TP+EP volumes sum correctly.
- **`commsBytesPerStep` zero when `multiDevice` undefined**.

### Roofline layer

- **Third regime**: synthetic call with `commsBytes / interconnectBW > flops/tflops` and `> hbmBytes/hbmBW` → `regime === 'comms'`.
- **Backward compat**: omitting comms-related args yields existing `'compute' | 'memory'` regimes; existing tests byte-identical.

### Integration

- **HGX H100 + Llama 70B + TP=8**: weights/8 ≈ 17.6 GB per rank → `perRank.fits === true` (vs aggregate-fits=false on single H100).
- **HGX H100 + Llama 70B + TP=8 at prompt=2048**: decode regime likely `'memory'` (per-rank weights still dominate).
- **GB200 NVL72 + DeepSeek V3 + TP=8 × EP=72 at long context**: comms term materializes; some regime flips to `'comms'`. Diagnostic value verified.
- **All existing tests** pass byte-for-byte (multiDevice optional + roofline extension backward-compat).

Net additions: ~15 unit tests + ~3 integration tests on top of current 110.

## UI: implementation surface

- **`InputPanel.svelte`**: the accelerator picker merges `ACCELERATORS` and `SYSTEMS` into one combined list grouped by vendor. Selecting a system populates `multiDevice` in the store; selecting a single accelerator leaves `multiDevice` undefined.
- **New component `ParallelismPicker.svelte`**: shown only when `multiDevice` is active. Displays current mode tuple; clicking opens a dropdown of valid alternatives.
- **`MemoryPanel.svelte`**: when `multiDevice` is active, bar shows `perRank` values; label below shows cluster aggregate.
- **`RooflinePanel.svelte`**: adds the third (interconnect) ceiling slope when `multiDevice` is active. Marker color picked by `regime`.
- **`stores.ts`**: derived store `defaultParallelism(system, model)` implementing the heuristic.

## Out of scope (deferred follow-ups)

- **Heterogeneous multi-accelerator systems** (mixed GPU types in one cluster) — punt; no current system has this.
- **Pipeline bubble overhead** for PP — current PP math ignores fill/drain time; OK at large microbatch counts but underestimates small-batch PP latency. Footnote.
- **Comms-compute overlap** — modern frameworks overlap collectives with compute; our `time = max(compute, hbm, comms)` upper-bounds this. Refinement could subtract an overlap factor per mode.
- **Non-ring topology adjustments** — TPU torus, NVSwitch direct, etc. Current model uses ring formulas as the conservative bound.
- **Achievable-tier comms BW** — `INTERCONNECTS` schema has slots for `contention` and `tiers` (analytical fabric model + empirical measurements) — these are unpopulated. First cut uses spec-sheet peak. Sustained tiers are a future PR.
- **Auto-parallelism optimizer** — picking the BEST parallelism for a (system, model, workload) tuple is a real optimization problem. We default heuristically and let the user override; we don't search.

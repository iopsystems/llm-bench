# Heterogeneous PD-disagg — Design

**Status:** Approved (brainstorm 2026-06-02)
**Builds on:** [2026-05-31-single-request-disagg-design.md](2026-05-31-single-request-disagg-design.md) (v1 symmetric P=D).
**Scope:** Let the user pick **different hardware and parallelism for the prefill cluster vs the decode cluster** in the simulator's disagg block. v1 symmetric remains the default. Quant, model, workload, and fabric stay shared across both sides. Asymmetric memory modeling is part of this PR (no longer parked).

## Goal

A user can model `H100 prefill + H200 decode` (KV-heavy workloads), or `H200 prefill + H100 decode` (compute-cheap decode), or any cross-vendor combo, and see honest TTFT/TPOT/Total numbers reflecting each cluster's actual capability. Parallelism degrees can differ per side (real production: prefill on TP=4 for compute throughput, decode on TP=2 + replicas for memory bandwidth).

## Behavior contract

For the disagg block, when **heterogeneous P/D is enabled**:

| Indicator | Source |
|---|---|
| **Prefill perf** | computed on prefill hw + prefill parallelism + prefill operating point |
| **Decode perf** | computed on decode hw + decode parallelism + decode operating point |
| **KV transfer time** | unchanged from v1: `kvCachePerRequest / fabricBw`. KV size depends on `quant.kv` + model (both shared); fabric is shared. |
| **TTFT (firstTokenOnPrefill=true)** | `prefill.timeS + (1 decode step on the PREFILL cluster)` — the prefill cluster generates token #1 locally, so the "first decode step" uses prefill hw's TFLOPS/HBM. |
| **TTFT (firstTokenOnPrefill=false)** | `prefill.timeS + kvTransferS` — unchanged. |
| **Total** | `TTFT + tpotS × (N − 1) + stutterS` — `tpotS` from decode cluster, `stutterS` from the existing case-B-slow formula. |
| **OOM** | two-sided: `prefillSide.total ≤ prefillHBM` AND `decodeSide.total ≤ decodeHBM`. |

When heterogeneous is OFF: identical to v1 symmetric — decode-side fields are absent from `CalcInput`, and the engine uses the prefill side for both phases.

## Non-goals (v2)

- **Per-side quant** (different KV dtype per cluster). Real but uncommon.
- **Per-side model** (cross-model PD-disagg is exotic).
- **Per-side fabric** (the fabric IS the link, only one).
- **Per-side workload semantics** (e.g. different prompts to prefill cluster).
- **Multi-stream / batched disagg.** Single-request scope per the parent simulator.

## Engine refactor

### CalcInput

```ts
interface CalcInput {
  // existing — used for prefill (and for both phases when decode fields absent)
  accelerator: AcceleratorSpec
  acceleratorVariantId: string
  multiDevice?: MultiDeviceConfig

  // NEW — decode cluster, used in heterogeneous disagg.
  // Absent ⇒ engine reuses prefill side (= v1 symmetric).
  decodeAccelerator?: AcceleratorSpec
  decodeAcceleratorVariantId?: string
  decodeMultiDevice?: MultiDeviceConfig

  // unchanged, shared across both sides
  model: ModelArch
  quant: Quantization
  workload: Workload
  disaggKvTransferFabricId?: string
  disaggFirstTokenOnPrefill?: boolean
}
```

### `calculate()`

1. Resolve `prefillVariant` from `accelerator + acceleratorVariantId`.
2. Resolve `decodeVariant` from `decodeAccelerator + decodeAcceleratorVariantId`, falling back to prefill side if either is missing.
3. Compute `memory = computeMemory(input)` — returns the two-phase result described below.
4. Compute `kvTransferS` from `memory.kvCachePerRequest / fabricBw` (unchanged).
5. **Op-point pairing.** Build a list of paired op-points via `pairOpPoints(prefillVariant, decodeVariant)` (new pure helper). For each pair:
   - `prefill = computePrefill(input, prefillOp, memory, side='prefill')`
   - `decode = computeDecode(input, decodeOp, memory, side='decode')`
   - For `firstTokenOnPrefill=true`: also `prefillFirstStep = computeDecode(input, prefillOp, memory, side='prefill')` — one decode step on the prefill cluster.
   - TTFT composition uses `prefillFirstStep.timePerTokenS` instead of `decode.timePerTokenS` in case-B; case-C and case-A unchanged.
   - StutterS and totalS formulas unchanged from v1.
6. `perf[pair.id] = { prefill, decode, ttftS, kvTransferS, ... }`. Pair id = `prefillOp.id` when names match across sides, else `"prefill/decode"` composite for the cross-name case.

`computePrefill` and `computeDecode` gain an optional `side` argument so they know which variant's TFLOPS/HBM bandwidth to use. Default (current callers) keeps using the prefill side.

### Op-point pairing helper

New pure module `calc/src/engine/opPoints.ts`:

```ts
export function pairOpPoints(
  prefill: AcceleratorVariant,
  decode: AcceleratorVariant,
): Array<{ prefillOp: OperatingPoint; decodeOp: OperatingPoint; id: string }>
```

Algorithm:
- For each `prefillOp` in `prefill.operatingPoints`:
  - Find `decodeOp` in `decode.operatingPoints` with the same `id`.
  - If no match, fall back to `decode.operatingPoints[0]`.
  - Pair id: `prefillOp.id` if matched, else `"${prefillOp.id}/${decodeOp.id}"`.
- Return the resulting list, in `prefill.operatingPoints` order.

Symmetric P=D collapses to today's behavior: same variant on both sides → same op-points → matched ids → identical list to today's `variant.operatingPoints`.

### Two-phase memory model

`computeMemory()` is refactored to expose per-side memory profiles:

```ts
interface MemoryResult {
  weights: number
  kvCachePerRequest: number
  kvCacheTotal: number

  // NEW: split activations
  prefillActivationsPeak: number   // = today's activationsPeak (scales with prompt × hidden)
  decodeActivationsPeak: number    // = 1 × (hidden + intermediate) × bytes(act_dtype) × 2 — single-token forward pass, single-layer working set; orders of magnitude smaller than prefillActivationsPeak

  // NEW: per-side totals
  prefillSide: MemorySide
  decodeSide: MemorySide

  // Backward-compat — = max(prefillSide.total, decodeSide.total). Roughly today's number
  // (prefill side dominates because prefill activations >> decode activations).
  total: number
  hbmCapacityGB: number     // prefill HBM by default; per-side HBM check is on the side slices
  headroom: number
  fits: boolean

  perRank?: { /* unchanged structure, reflects prefill side's parallelism for backward compat */ }
}

interface MemorySide {
  weights: number
  activations: number       // prefillActivationsPeak or decodeActivationsPeak
  kvCache: number           // = kvCacheTotal on both sides (decode holds it; prefill builds it
                            // and briefly holds the full KV at end-of-prefill before shipping)
  total: number             // sum of the above
  hbmCapacityGB: number     // capacity of this side's accelerator
  headroom: number
  fits: boolean
  perRank?: { weights, kvCachePerRequest, activations, total, headroom, fits }
}
```

**Why `kvCache` shows up on BOTH sides:** the prefill cluster reaches its peak memory at end-of-prefill, when KV is fully built but not yet shipped — `weights + prefill_activations + full_KV`. The decode cluster holds KV throughout decode — `weights + decode_activations + full_KV`. Both sides are bound by KV at peak.

What the per-side split changes vs today's combined `total`:
- **Decode side gets the benefit:** decode peak = `weights + KV + tiny_decode_activations`, no longer pessimistically includes prefill activations. Decode cluster can be smaller-HBM than today's check allows.
- **Prefill side ≈ same as today:** prefill peak = `weights + KV + prefill_activations` (= today's `total`). No accuracy change.

### Backward-compat for callers that read `memory.fits` / `memory.total`

- The existing `memory.total` field stays (= `max(prefillSide.total, decodeSide.total)`, ≈ prefill side).
- `memory.fits` stays (= `prefillSide.fits`). The monolithic Simulator block still reads `memory.fits` and matches today's behavior.
- The disagg block (Simulator.svelte) reads `prefillSide.fits` AND `decodeSide.fits` separately. The existing v1 disagg block (when heterogeneous toggle off) now uses the two-sided check — strictly more permissive than today's combined check for the rare cases where decode side fits but combined doesn't.

## Stores

New writables in `stores.ts`:

```ts
export const decodeAcceleratorId = writable<string>('')
export const decodeVariantId     = writable<string>('')
export const decodeSystemId      = writable<string>('')
export const decodeParallelismOverride = writable<ParallelismConfig | null>(null)
export const heterogeneous       = writable<boolean>(false)
```

New derived:
- `decodeMultiDevice: Readable<MultiDeviceConfig | undefined>` — mirrors `multiDevice`, but built from the decode-side stores.

Modified derived `simInputDisagg`:
- When `$heterogeneous === false`: same as v1 (decode-side fields absent on the `CalcInput`).
- When `$heterogeneous === true`: spread `decodeAccelerator`, `decodeAcceleratorVariantId`, `decodeMultiDevice` into the CalcInput. Per-field fallback to the prefill side when a decode-side store is empty (e.g., `decodeAcceleratorId === ''` ⇒ use `acceleratorId`; `decodeParallelismOverride === null` ⇒ use the prefill `parallelismOverride`). This lets the user toggle heterogeneous on without immediately committing to a choice on every dimension — the engine just sees "prefill side" until the user changes something.

`simInputMonolithic` is unchanged — monolithic is one cluster, heterogeneous doesn't apply.

## URL state

New encode/decode keys in `share.ts`, all conditional on `heterogeneous === true`:

- `het=1` (omit when false)
- `a2=<decodeAcceleratorId>` (when single-chip decode)
- `v2=<decodeVariantId>`
- `s2=<decodeSystemId>` (when multi-device decode; takes precedence over a2/v2)
- `p2=<encodeParallelism(decodeParallelismOverride)>` (when set)

Encoding rule mirrors the existing prefill-side `s` vs `a+v` precedence. When `het` is absent or `0`, none of the decode-side keys decode (returns to symmetric mode). A sim URL with `het=1&a=h100&v=sxm-80&a2=h200&v2=sxm-141` round-trips back to heterogeneous mode on load.

## UI

### `DisaggInputPanel.svelte` — grows the heterogeneous controls

Layout:

```
Disagg block
├─ KV transfer fabric: [IB-NDR — 100 GB/s/GPU ▼]
├─ ☑ 1st token on prefill (hide transfer in TTFT)
├─ ☐ Use different hardware for decode cluster   ← new toggle (binds $heterogeneous)
│
└─ ── Decode cluster ──                           (renders only when $heterogeneous)
    ├─ Accelerator: [<combo dropdown, same as prefill side> ▼]
    ├─ Variant: [SXM-141 ▼]   (when single-chip selected)
    └─ Parallelism: [TP=8 ▼]  (when decode system selected, via existing ParallelismPicker)
```

When the toggle is first flipped on, decode-side stores are pre-populated with the current prefill-side values, so the user transitions from symmetric to asymmetric by changing one knob.

The decode-side combo dropdown reuses the existing `orderSkus(ACCELERATORS, SYSTEMS)` grouping — same UI affordance as the prefill side. The decode-side parallelism picker is the existing `ParallelismPicker` component, reading/writing the decode-side stores instead.

### `Simulator.svelte` — OOM message identifies the failing side(s)

The existing OOM gate (`memory.perRank?.fits ?? memory.fits`) is replaced with a two-sided gate reading from `$simResultDisagg.memory.prefillSide` and `$simResultDisagg.memory.decodeSide`. The amber notice identifies which side(s) fail:

- "Out of memory on prefill cluster" — prefill side's weights + prefill activations + KV exceed prefill HBM
- "Out of memory on decode cluster" — decode side's weights + KV + decode activations exceed decode HBM
- "Out of memory on both clusters" — both fail

Remediation hint differs slightly per side ("trim promptTokens" for prefill activations, "add parallelism" for decode KV pressure).

### Op-point pair labels in multi-op-point cards

In multi-op-point rendering (per the v1 KPI-card stacking):
- When the pair's prefill and decode ids match: render the op-name as just the id (e.g. "peak"). Same as today.
- When they differ (cross-name fallback): render as `"prefillId / decodeId"` (e.g. "peak / achievable"). Disambiguates the cross-fallback case visually.

## Files

| File | Action | Purpose |
|---|---|---|
| `calc/src/engine/types.ts` | modify | Add `decodeAccelerator?`, `decodeAcceleratorVariantId?`, `decodeMultiDevice?` to `CalcInput`. Add `prefillActivationsPeak`, `decodeActivationsPeak`, `prefillSide`, `decodeSide`, and `MemorySide` interface to `MemoryResult` / new export. |
| `calc/src/engine/memory.ts` | modify | Split `activationsPeak` into prefill + decode profiles. Compute per-side totals + perRank slices. Keep backward-compat `total` and `fits`. |
| `calc/src/engine/calc.ts` | modify | Resolve `prefillVariant` and `decodeVariant` (fall back to prefill when decode absent). Build paired op-points via `pairOpPoints()`. For each pair: compute prefill/decode on respective sides; for case-B, also compute `prefillFirstStep` for TTFT formula. |
| `calc/src/engine/prefill.ts`, `decode.ts` | modify | Add optional `side: 'prefill' \| 'decode'` argument so the function picks the right variant's TFLOPS/HBM. Default 'prefill' for backward compat. |
| `calc/src/engine/opPoints.ts` | create | Pure helper `pairOpPoints(prefillVariant, decodeVariant)`. Unit-tested. |
| `calc/src/ui/stores.ts` | modify | Add the 5 new writables (`decode*Id`, `decodeParallelismOverride`, `heterogeneous`). Add `decodeMultiDevice` derived. Update `simInputDisagg` to spread decode-side fields when `$heterogeneous === true`. |
| `calc/src/ui/share.ts` | modify | Encode/decode `het`, `a2`, `v2`, `s2`, `p2`. All conditional on heterogeneous. |
| `calc/src/ui/DisaggInputPanel.svelte` | modify | Add heterogeneous toggle. When on, render the "Decode cluster" section: combo dropdown (accelerator + variant + system), parallelism picker. Pre-populate from prefill side on first toggle. |
| `calc/src/ui/Simulator.svelte` | modify | Two-sided OOM gate. Updated amber notice. Op-pair label rendering in the resultBlock snippet (slash-separator when ids differ). |

## Testing

- **`opPoints.ts`:** matched-name pairing, mismatched-name fallback (peak/peak, peak/achievable), single-op-point on either side.
- **`memory.ts`:** `prefillSide.total > decodeSide.total` for typical Llama-70B + 8k prompt (sanity check). `prefillActivationsPeak >> decodeActivationsPeak` for the same. Per-side `perRank` slices correct under parallelism.
- **`calc.ts`:** asymmetric input (H100 prefill + H200 decode) returns prefill perf from H100 and decode perf from H200 — assert ttftS uses prefill TFLOPS and tpotS uses decode HBM bandwidth. firstTokenOnPrefill=true: assert TTFT includes prefill-cluster decode-step time (not decode-cluster's).
- **`stores.ts`:** when `$heterogeneous === false`, `simInputDisagg` does NOT include decode fields. When `$heterogeneous === true`, it does. No write-back to prefill-side stores.
- **`share.ts`:** round-trip URL `het=1&a=h100&v=sxm-80&a2=h200&v2=sxm-141&...` correctly. URL without `het` decodes to symmetric (no decode-side state).
- **In-browser smoke:** toggle the het checkbox on → decode-side selectors appear pre-populated. Change decode accelerator to H200 → KPI cards update with asymmetric numbers. Two-sided OOM check displays correctly. URL reflects the new decode-side keys.

## Rationale (key calls)

- **Hardware + parallelism per side, not full asymmetry:** matches real production (DeepSeek PD, NVIDIA Dynamo) where compute and memory profiles diverge between prefill and decode roles. Per-side quant is rare and would balloon UI surface; skip for v2.
- **Explicit `heterogeneous` toggle vs auto-detect:** keeps "symmetric" the unambiguous default and avoids "happens-to-be-the-same-SKU is ambiguous with explicit symmetric" confusion. Toggle persists in URL so shared links stay deterministic.
- **Match-by-name op-point pairing:** users naturally compare "peak vs peak, achievable vs achievable." Cartesian product would yield 4 rows for the common case (both sides have peak + achievable), most of which (peak prefill + achievable decode etc.) aren't realistic deployment choices. Fallback to first op of the other side handles the unmatched case without forcing the user to make a choice.
- **Two-phase memory model in this PR rather than parked:** the asymmetric use case ("small-HBM decode + large-HBM prefill") doesn't work under today's conservative combined-total check. Conservative-only would defeat the headline value of heterogeneous. Splitting the activations into prefill/decode profiles is a contained engine change with a clear backward-compat path (keep `total` and `fits` for monolithic callers).
- **Inherit decode side from prefill when toggle is on but decode fields empty:** lets the user transition gradually (toggle on → starts symmetric → user changes one knob → asymmetric). Avoids a "you must pick decode hw immediately" friction.

## Open follow-ups (parked, not blockers)

- Per-side quant (especially KV dtype, since decode holds the KV).
- Asymmetric workload semantics (smaller prompts for prefill cluster, larger for decode).
- Per-side fabric tier (different network class between clusters and within each cluster).
- Multi-stream / batched disagg (queueing model — separate from this single-request simulator).

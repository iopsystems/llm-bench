# Single-request simulator — PD-disagg configuration

**Status:** Approved (brainstorm 2026-05-31)
**Builds on:** [2026-05-26-single-request-simulator-design.md](2026-05-26-single-request-simulator-design.md) (monolithic).
**Scope:** Add **prefill/decode disaggregation** as the simulator's second configuration block. Stacked under the existing "Single request, monolithic" block. Symmetric prefill = decode in v1; asymmetric P/D documented as a TODO.

## Goal

When the user picks a KV-transfer fabric, the simulator displays a parallel **"Single request, PD-disagg"** block below the monolithic one, so the user can compare end-to-end latency of one request under monolithic vs. PD-disagg deployments on the same hardware. The disagg-specific knobs (fabric, first-token-on-prefill) live in a dedicated input panel inside that block.

## Behavior contract

For the disagg block:

| Indicator | Source |
|---|---|
| **TTFT** (incl. KV transfer overhead) | `perf[op].ttftS` — engine's existing disagg-aware formula. |
| **TPOT** | `perf[op].decode.timePerTokenS` — unchanged. |
| **Total latency** | `TTFT + TPOT × (outputTokens − 1)` — same formula as monolithic. Caveat from the monolithic spec carries forward unchanged. |
| **Sustained output** | `1 / TPOT` tokens/sec. |
| **Sustained input** | `inputTokenRate` from engine (`prompt / prefill.timeS`). |
| **Prefill regime** | `perf[op].prefill.regime`. |
| **Decode regime** | `perf[op].decode.regime`. |
| **Gantt geometry** | Existing `computeGanttGeometry`. Cases A/B/C as designed; this PR exercises B and C for the first time in production. |

Hardware, model, quant, and parallelism are **inherited from the shared inputs above**. The disagg block does not duplicate those controls in v1.

## Non-goals (v1)

- **Asymmetric P/D** (different hw or parallelism for prefill vs decode). Real production setups do this (e.g. H100 prefill + H200 decode), but it requires non-trivial engine extensions. Tracked in the roadmap as a follow-up.
- **Per-side memory check.** Symmetric P=D → both sides have identical memory budget → one check suffices.
- **Multi-stream / batched disagg.** Single-request scope per the parent simulator design.
- **Modeling fidelity beyond peak BW**: no latency floor, no congestion / contention, no achievable-vs-theoretical factor. Tracked as the "modeling fidelity" angle in the roadmap.
- **Side-by-side comparison rows / delta summaries.** The two stacked blocks let the user compare visually; we don't compute an explicit "delta TTFT" widget in v1.
- **Calc tab UI rework** beyond what's necessary to keep its disagg picker functioning after the engine refactor.

## Engine refactor

`disaggKvTransferFabricId` and `disaggFirstTokenOnPrefill` move from `MultiDeviceConfig` to `CalcInput` top-level.

Rationale: PD-disagg describes the **topology** (the wire between two clusters), not a property of one cluster's parallelism. Two single H100 nodes connected by RoCE-400 is a valid PD-disagg setup that today's UI can't express because disagg is gated on multi-device-system selection.

```ts
// Before
interface MultiDeviceConfig {
  system: MultiAcceleratorSystem
  parallelism: ParallelismId[]
  parallelismDegrees: Partial<Record<ParallelismId, number>>
  disaggKvTransferFabricId?: string
  disaggFirstTokenOnPrefill?: boolean
}

// After
interface MultiDeviceConfig {
  system: MultiAcceleratorSystem
  parallelism: ParallelismId[]
  parallelismDegrees: Partial<Record<ParallelismId, number>>
}

interface CalcInput {
  accelerator: AcceleratorSpec
  acceleratorVariantId: string
  model: ModelArch
  quant: Quantization
  workload: Workload
  multiDevice?: MultiDeviceConfig
  disaggKvTransferFabricId?: string
  disaggFirstTokenOnPrefill?: boolean
}
```

Engine math:
- `calc.ts`: read `input.disaggKvTransferFabricId` / `input.disaggFirstTokenOnPrefill` instead of `input.multiDevice?.disagg…`. Same fabric lookup, same `kvTransferS = kvCachePerRequest / (bw * 1e9)` formula, same TTFT composition (3 cases).
- No change to memory math or per-op-point computation.

Stores:
- `disaggKvTransferFabricId` and `disaggFirstTokenOnPrefill` stay as standalone writables. Their values flow into `input` (the derived `CalcInput` store) at the top level, not nested under `multiDevice`.

Share/URL:
- URL keys `dk` / `df` remain (preserves shareable URLs).
- Decoding: keys apply regardless of system selection now.
- Encoding: emit `dk` whenever the value is non-empty, emit `df=0` when the user opted into sequential handoff (existing convention).
- The current encoder already emits these only when `state.systemId && state.disaggKvTransferFabricId` (share.ts:73-78). Drop the `state.systemId` gate so `?a=h100&v=sxm-80&dk=roce-400` works.

## Catalog additions

Five new entries in `calc/src/data/interconnects.ts`, all `scale: 'scale-out'`:

| id | name | perGpuBandwidthGBs | notes |
|---|---|---|---|
| `roce-200` | RoCEv2 200 GbE | 50 | ConnectX-6 generation. Same physical layer as IB-HDR; different transport. |
| `roce-400` | RoCEv2 400 GbE | 100 | ConnectX-7 generation. The modal Ethernet AI fabric today. |
| `roce-800` | RoCEv2 800 GbE | 200 | ConnectX-8 generation. Matches IB-XDR per-port BW. |
| `spectrum-x-400` | NVIDIA Spectrum-X 400G | 100 | RoCEv2 + adaptive routing + Spectrum-X congestion control; lossless Ethernet for AI. |
| `spectrum-x-800` | NVIDIA Spectrum-X 800G | 200 | SN5600 + ConnectX-8. Blackwell-era pairing. |

`perDirectionGBs` = half of `perGpuBandwidthGBs` (same convention as the IB rows).

Note: the engine doesn't currently distinguish lossy RoCE from lossless Spectrum-X. They produce identical `kvTransferS` for the same BW. If/when we add achievable-vs-theoretical modeling, Spectrum-X would diverge favorably from vanilla RoCE under contention.

Deliberate omissions: Ultra Ethernet (UEC — spec finalizing, deployments thin), AWS EFAv4 (not yet a publicly named generation as of 2026-05). Both are roadmap follow-ups.

## InterconnectSpec schema extension

```ts
interface InterconnectSpec {
  // … existing fields …
  compatibleAcceleratorIds?: string[]  // scale-up fabrics: which accelerator families can host them
}
```

Populated only for the 4 scale-up fabrics that are eligible as PD-disagg "wires":

| Fabric | `compatibleAcceleratorIds` |
|---|---|
| `nvlink-4-nvl-256` (DGX H100 SuperPOD) | `['h100', 'h200']` |
| `nvlink-5-nvl72` (NVL72) | `['gb200', 'b100', 'b200']` |
| `tpu-ici-v5p` | `['tpu-v5p']` |
| `tpu-ici-trillium` | `['tpu-trillium']` |

Scale-out fabrics leave the field undefined → compatible with any accelerator. Intra-node and die-to-die fabrics (NVLink baseboard, xGMI mesh, Gaudi RoCE, PCIe, UltraFusion) also leave it undefined; they're filtered out separately by the `scale` check in the picker.

## UI

### Layout

```
Sim tab
├─ Shared InputPanel (hideConcurrency=true, hideDisagg=true)
│   HW · Model · Quant · Parallelism · Prompt/Output tokens
│
├─ Block: "Single request, monolithic"
│   KPI cards · Gantt
│
└─ Block: "Single request, PD-disagg"        (when fabric set)
    DisaggInputPanel
      Fabric picker (filtered + grouped) · First-token-on-prefill toggle
    KPI cards · Gantt
```

The shared input panel renders with `hideDisagg={true}` (new prop) to remove the disagg picker from the shared inputs — the disagg block owns those inputs now.

### `DisaggInputPanel.svelte` (new)

A small sibling of `InputPanel.svelte`. v1 contents:

- **Fabric picker** — `<select>` bound to `$disaggKvTransferFabricId`, structured with two `<optgroup>`s ("Intra-domain", "Cross-rack") and a leading `<option value="">— off (monolithic only) —</option>`. Options labeled `{name} — {perGpuBandwidthGBs} GB/s/GPU`.
  - Scale-up entries filtered by `compatibleAcceleratorIds.includes(selectedAcceleratorId)`.
  - Scale-out entries shown unconditionally.
  - Sort within each group by `perGpuBandwidthGBs` descending (fastest first).
- **First-token-on-prefill** — checkbox bound to `$disaggFirstTokenOnPrefill`, label "1st token on prefill (hide transfer in TTFT)". Same control as the calc tab's existing one. Defaults true.

When fabric is `""` (off): the entire **"Single request, PD-disagg"** block does not render. The user sees only the monolithic block.

When fabric is set but the user's accelerator doesn't appear in the scale-up compatibility list, that scale-up entry simply isn't in the picker — no warnings needed.

v2 will grow `DisaggInputPanel`: separate prefill/decode hardware selectors, possibly per-side parallelism. Today's symmetric inputs sit on top of the shared inputs from the parent panel.

### Block visibility

`{#if disaggKvTransferFabricId}` — block renders. No system selection required (the engine refactor decouples disagg from `MultiDeviceConfig`).

### OOM gating

Symmetric P=D → same hw both sides → same memory footprint → existing single OOM check applies to both blocks. If the configuration OOMs, both blocks hide and the existing amber notice replaces them (already shipped).

### Calc tab

After the engine refactor, the calc tab's existing disagg picker (in `InputPanel.svelte`, currently gated by `$systemId`) is **no longer gated** — it's visible whenever the user picks any hw. The `hideDisagg` prop defaults `false` so calc tab keeps the picker. The picker's contents follow the same filter rules as the sim tab's `DisaggInputPanel`.

## Files

| File | Action | Purpose |
|---|---|---|
| `calc/src/engine/types.ts` | modify | Move `disaggKvTransferFabricId` + `disaggFirstTokenOnPrefill` from `MultiDeviceConfig` to `CalcInput`. Add `compatibleAcceleratorIds?: string[]` to `InterconnectSpec`. |
| `calc/src/engine/calc.ts` | modify | Read disagg fields from `input.…` directly, not `input.multiDevice?.…`. |
| `calc/src/data/interconnects.ts` | modify | Add 5 RoCE/Spectrum-X entries. Populate `compatibleAcceleratorIds` for the 4 scale-up eligible fabrics. |
| `calc/src/ui/stores.ts` | modify | `disaggKvTransferFabricId` and `disaggFirstTokenOnPrefill` flow into top-level `CalcInput` (in the derived `input` store), no longer nested under `multiDevice`. **Add `simInputMonolithic` and `simInputDisagg` derived stores** (both clamp concurrency to 1; the monolithic variant additionally clears the disagg fields; the disagg variant preserves them). Add corresponding `simResultMonolithic` and `simResultDisagg` derived stores that call `calculate()` on each. Existing `simInput`/`simResult`/`simError` can stay as aliases or be replaced by the two new pairs — decision deferred to plan-writing. |
| `calc/src/ui/share.ts` | modify | Encode `dk`/`df` unconditionally (drop the `state.systemId` gate). Decode unchanged. |
| `calc/src/ui/InputPanel.svelte` | modify | Add `hideDisagg` prop (default false). Drop the `{#if $systemId}` gate on the disagg picker. Use the eligible-fabric filter (scale-up by compatibility; scale-out unconditional). Show "name — BW GB/s/GPU" labels with optgroups. |
| `calc/src/ui/DisaggInputPanel.svelte` | create | Sim-tab disagg input panel: fabric picker + first-token toggle. Same filter/labeling logic as the InputPanel picker (factor out into a small helper in this PR — see catalog helpers below). |
| `calc/src/ui/disaggFabrics.ts` | create | Pure helper exporting `eligibleDisaggFabrics(acceleratorId): InterconnectSpec[]` and `formatFabricLabel(f: InterconnectSpec): string`. Reused by InputPanel and DisaggInputPanel. Unit-testable in node. |
| `calc/src/ui/Simulator.svelte` | modify | Pass `hideDisagg={true}` to the shared InputPanel. Factor the existing "build rows per op-point" code into a small block-rendering subcomponent (inline `{#snippet}` or a local helper); invoke it twice — once for monolithic (reads `$simResultMonolithic`), once for disagg (reads `$simResultDisagg`, gated by `{#if $disaggKvTransferFabricId}`). The disagg block hosts a `<DisaggInputPanel />` between its header and KPI cards. |

## Testing

- **Engine refactor:** existing `calc.test.ts` tests should still pass after migration (the disagg test there constructs `multiDevice.disaggKvTransferFabricId = 'ib-ndr'`; update it to set the field at the top level). Add one new test: disagg works without a `multiDevice` config (single-chip + scale-out fabric).
- **`disaggFabrics.ts` (pure):** unit tests for `eligibleDisaggFabrics`: scale-up filter by compatibility; scale-out always present; intra-node never present; sorted by BW desc within groups.
- **Catalog:** existing skill-sync tests cover model arch; no test for interconnect schema directly, but the new entries should typecheck.
- **Stores / share.ts:** existing tests stay green. Add: round-trip URL `?a=h100&v=sxm-80&dk=roce-400` decodes/encodes correctly (no system selected, disagg fabric set).
- **InputPanel / DisaggInputPanel:** presentational; verified in-browser per existing convention.
- **In-browser smoke:** monolithic-only when no fabric; fabric → disagg block appears below; both blocks render KPI cards + gantt; toggling first-token reshapes the disagg gantt; switching to a non-Blackwell accelerator hides NVL72 from the picker; URL round-trips.
- Full `npm test` + `npm run check` + `npm run build` must stay green.

## Implementation order (sketch — plan owns final ordering)

1. Engine refactor: `CalcInput` hoist + `calc.ts` read-site + update existing tests.
2. `interconnects.ts` schema extension (`compatibleAcceleratorIds`) + 5 new entries.
3. `disaggFabrics.ts` pure helper + tests.
4. `share.ts` unconditional `dk`/`df` encode + tests.
5. `stores.ts` migration (disagg fields no longer nested).
6. `InputPanel.svelte` updates (`hideDisagg` prop, dropped system gate, new filter/labels).
7. `DisaggInputPanel.svelte` (new component).
8. `Simulator.svelte` — pass `hideDisagg`, add disagg block.
9. In-browser smoke + final review.

## Rationale (key calls)

- **Engine hoist over keeping disagg under `MultiDeviceConfig`:** PD-disagg is a deployment topology, not a property of one cluster's parallelism. Users want to model two single GPUs over RoCE; the current type model forbids that. Hoisting also simplifies the v2 asymmetric extension (no need to invent a second `MultiDeviceConfig` for the other side).
- **Symmetric P=D in v1:** the engine's *math* already supports this case (the calc tab has been computing disagg TTFT for symmetric configs since the disagg feature shipped). The PR's engine work is purely a schema migration — moving two fields out of `MultiDeviceConfig` so single-chip + scale-out disagg becomes expressible. Asymmetric requires real math extensions (per-side memory, per-side parallelism, per-side TFLOPS) that warrant their own design pass.
- **Stacked blocks over toggle:** the "Single request, monolithic" header from the monolithic PR already anticipates this. Stacking lets users compare without clicking; scales naturally to future configurations (speculative decoding, draft model, etc).
- **Disagg inputs in a dedicated `DisaggInputPanel`** rather than inline: future-proofs for asymmetric. v1's inputs are sparse (fabric + toggle), but v2 will add separate hw + parallelism per side — the structural home is already correct.
- **Inherit hardware from shared inputs in v1:** zero duplication, no extra widgets, clear v2 evolution path (the panel grows to override-only the inputs that need to differ between sides).
- **No comparison delta widget:** two stacked blocks rendering the same metrics is enough for visual comparison; an explicit "Δ TTFT" widget would compete with the gantt for attention and presumes which metric the user cares about.
- **Compatibility filter as a data field, not a side table:** lives next to the fabric definition, where reviewers can keep it correct as new accelerators land. Optional field keeps schema migration trivial.

## Open follow-ups (parked, not blockers)

- Achievable-vs-theoretical fabric model (latency floor, contention, SRD vs lossless).
- Asymmetric P/D (separate prefill cluster + decode cluster SKUs, plus per-side parallelism).
- Ultra Ethernet / EFAv4 catalog entries when they're real production options.
- Tiered KV cache (already on the roadmap backlog; interacts with disagg via decode-side memory).

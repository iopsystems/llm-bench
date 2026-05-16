# Info Page (SKU / Model Spec Sheets) — Design

**Status:** Approved (brainstorm 2026-05-15)
**Scope:** Roadmap item #1 of the calc roadmap. Self-contained; presentation-only, no engine math changes.

## Goal

A browsable Info tab that summarizes the characteristics of every model and every SKU (accelerator / multi-accelerator system) in the catalog, as static reference. Establishes the tab/route infrastructure that roadmap items #5 (multi-candidate comparison) and #6 (cloud-SKU tab) will reuse.

## Non-goals

- No workload/quant coupling — the Info page does not react to calculator state.
- No "Use in Calculator" hand-off action (deferred until #6 needs the pattern).
- No search box (catalog is ≈38 models / 27 accelerators / 12 systems grouped by publisher — scannable).
- No engine/math changes; reuse existing engine helpers.

## 1. Routing model

- New `src/ui/route.ts`: a writable `route` store with values `'calc' | 'info'`, plus an optional detail target (`{ kind: 'model' | 'sku', id: string }`). Initialized from `location.hash`, kept in sync via a `hashchange` listener, and updates `location.hash` on navigation.
- Hash grammar:
  - `#calc` (default / empty hash) → calculator view.
  - `#info` → Info catalog list.
  - `#info/model/<id>` → model spec sheet.
  - `#info/sku/<id>` → SKU spec sheet (id resolves against accelerators first, then systems).
- Relationship to `share.ts`: `share.ts` owns shareable **calculator input state** in the hash payload; `route.ts` owns the **view path**. To avoid two owners of `location.hash`:
  - `route` owns the leading path segment (`calc` / `info/...`).
  - Calculator share-state restore fires **only** on the `calc` route.
  - Info routes are payload-free (Info is stateless reference; nothing to share).
- `App.svelte` renders `<TabBar/>` then switches on `$route` between the existing calculator view (InputPanel / MemoryPanel / PerfPanel / RooflinePanel / DerivationDrawer) and `<InfoPanel/>`.

## 2. Component boundaries

- `TabBar.svelte` — renders the tab strip (Calculator | Info; Compare/Cloud omitted until #5/#6 exist), sets `route`. Active tab reflects `$route`.
- `InfoPanel.svelte` — owns the Info view: a Models / SKUs sub-toggle, the grouped catalog list, the current-selection pin, and routing into a detail sheet.
- `ModelSpecSheet.svelte` — prop: a `ModelArch`. Pure presentational.
- `SkuSpecSheet.svelte` — prop: an `AcceleratorSpec | MultiAcceleratorSystem`. Pure presentational.
- `catalogMetrics.ts` — pure functions `modelMetrics(model)` and `skuMetrics(spec)` returning derived numbers, **reusing existing engine helpers** (KV-bytes-per-token-per-layer, roofline ridge) rather than reimplementing — prevents drift, consistent with `docs/data-philosophy.md`.

`ModelSpecSheet` / `SkuSpecSheet` / `catalogMetrics` are the units #5 reuses N-up for comparison.

## 3. Content

### Model spec sheet

**Raw (from `ModelArch`):** name, publisher, family, releaseDate, paramCount (+ active param count for MoE), layers, hiddenDim, intermediateDim, numHeads, numKvHeads, headDim, vocabSize, maxContext, attention variant + its sub-params, architecture (dense / MoE: experts total · active · shared), numNextnLayers (MTP depth).

**Derived static** (fixed **fp16 reference**, labeled as such; reuses engine fns):
- KV bytes per token per layer for the attention variant.
- Total KV bytes per token across the model (engine's attended-layer logic).
- GQA ratio (numHeads / numKvHeads).
- One-line attention-variant explanation (full / GQA / MLA / sliding / hybrid / …).
- MoE active/total parameter ratio.

### SKU spec sheet (accelerator or system)

**Raw:** vendor, family, releaseDate; per variant: HBM capacity, and per operating point the TFLOPS-by-dtype table, HBM BW, provenance (`asOf` / sources / notes). For systems: composition (accelerator × count, variant id), interconnect, formFactor, aggregate (total HBM, fabric bidirectional), cloud availability.

**Derived static:**
- Arithmetic-intensity ridge (peak FLOPS ÷ HBM BW, FLOP/byte) per operating point / dtype.
- Peak → achievable efficiency delta where both operating points exist.
- For systems: aggregate compute / memory rollups.

### Future enhancements (not v1)

- **Topology diagram** for multi-accelerator systems: visualize the interconnect (8-GPU NVSwitch mesh, GB200 NVL72 rack, TPU 3D-torus) derived from the system's `topology` / `scale` / `maxScaleUpGpus` + count. v1 shows topology as raw text/fields; the diagram is a later visual layer over the same data.

## 4. Catalog list affordance

Grouped by publisher (models) / vendor (SKUs) using the existing `catalogOrder` ordering — identical order to the pickers for consistency. Info tab has two sub-sections: **Models** and **SKUs**. The model and SKU currently selected in the calculator are pinned + highlighted at the top of their respective lists. No search box.

## 5. Testing

- `catalogMetrics.ts` — vitest suite, exact expected values, TDD, mirroring existing engine test style.
- `route.ts` — vitest suite for hash parse / serialize round-trips (incl. default/empty hash, model vs sku detail, unknown id fallback).
- Spec-sheet + TabBar + InfoPanel components — no unit tests (presentational); verified in-browser, consistent with how existing panels are treated.

## Architecture rationale

Approach A (chosen over inline-render and a router library): a small `route` store + dedicated Info components + a pure tested metrics module. Marginal cost over inline rendering is one store and splitting the spec sheet into its own component — and that component plus the metrics module are exactly what #5 reuses. A router library is YAGNI for 2–4 tabs; the existing hash mechanism proves it's unnecessary.

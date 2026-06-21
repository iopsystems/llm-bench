# Accelerator-Tier UI Hookup — Design

## Goal

Surface the `tier: 'datacenter' | 'consumer'` field on `AcceleratorSpec` (landed in PR #153) as a filtering affordance across the four accelerator-picker UIs in the app — the Calc-tab InputPanel, the two cluster pickers in DisaggInputPanel, and the Info-tab catalog. Consumer SKUs are hidden by default to keep the serving-focused views uncluttered; a "Show consumer GPUs" checkbox next to each picker reveals them.

Out of scope: persisted preference (localStorage / URL state), per-tab independent toggles, multi-tier presets.

## Why

PR #153 added `tier` to every accelerator entry (17 datacenter, 21 consumer) but didn't surface it. The dropdowns currently list all 38 entries indiscriminately, mixing consumer cards (RTX 5090, Apple M-series, Radeon RX 7900 etc.) into the same publisher groups as datacenter parts (H100, MI300X). The calc app's serving-focus intent makes consumer SKUs more like a "browse mode" than the default.

## Architecture

### Single shared toggle store

```ts
// src/ui/stores.ts
//
// In-memory only — resets on reload. Default hides consumer SKUs to keep
// dropdowns and catalog focused on serving-scale hardware. URL state is not
// needed because the auto-show-when-current-selection-is-consumer rule
// below handles shared links that select a consumer SKU.
export const showConsumerSkus = writable<boolean>(false)
```

### Pure filter helper

```ts
// src/ui/catalogOrder.ts (alongside the existing orderSkus / orderModels)
//
// Filter accelerators by tier, always preserving explicitly-selected ids so
// a shared `?a=rtx-5090` URL still renders that option even when the toggle
// is off.
export function filterByTier(
  accelerators: AcceleratorSpec[],
  showConsumer: boolean,
  alwaysShowIds: string[] = []
): AcceleratorSpec[] {
  return accelerators.filter(a =>
    showConsumer ||
    a.tier === 'datacenter' ||
    alwaysShowIds.includes(a.id)
  )
}
```

### Toggle UI — 4 checkbox sites, all bound to the same store

Inline next to each accelerator-label, small grey font so it doesn't compete with the primary picker:

```svelte
<label class="show-consumer">
  <input type="checkbox" bind:checked={$showConsumerSkus} />
  <span>Show consumer GPUs</span>
</label>
```

Sites:
1. **`InputPanel.svelte`** — next to the existing "Accelerator" label in the Hardware fieldset
2. **`DisaggInputPanel.svelte`** — next to the "Accelerator" label in each cluster picker (×2: prefill cluster, decode cluster)
3. **`InfoPanel.svelte`** — at the top of the SKU catalog section

CSS:
```css
.show-consumer {
  display: inline-flex; align-items: center; gap: 0.3rem;
  margin-left: 0.6rem;
  font-size: 0.78rem; font-weight: 400; color: #666;
}
.show-consumer input[type=checkbox] { width: auto; margin: 0; }
```

### Per-site filter wiring

Each site replaces its current `orderSkus(ACCELERATORS, SYSTEMS)` call with a filtered version. The `alwaysShowIds` argument differs per site:

- **InputPanel**: `[$acceleratorId]` — the Calc-tab single accelerator
- **DisaggInputPanel** prefill cluster: `[$prefillAcceleratorId || $acceleratorId]`
- **DisaggInputPanel** decode cluster: `[$decodeAcceleratorId || $prefillAcceleratorId || $acceleratorId]`
- **InfoPanel** catalog: `[]` — no "current selection" in the catalog

Example wiring (InputPanel):
```svelte
$: skuGroups = orderSkus(
  filterByTier(ACCELERATORS, $showConsumerSkus, [$acceleratorId]),
  SYSTEMS
)
```

Each call is reactive: re-runs when `$showConsumerSkus` flips or the currently-selected id changes.

## Data flow

1. User opens any tab → `$showConsumerSkus = false` (initial). Each picker filters to datacenter-only via `filterByTier`. Their `alwaysShowIds` arg pulls in the current selection (datacenter by default, so a no-op).
2. User flips the checkbox in any site → `$showConsumerSkus = true` → all 4 sites' `filterByTier` calls recompute → all dropdowns + catalog now include consumer SKUs.
3. User flips back to false → consumer SKUs disappear from dropdowns except wherever they're the current selection.
4. User lands on a URL with `?a=rtx-5090` → URL parser writes `acceleratorId.set('rtx-5090')` → `filterByTier`'s `alwaysShowIds=['rtx-5090']` keeps RTX 5090 in the dropdown even though `$showConsumerSkus` is false.

## Testing

### `test/ui/filterByTier.test.ts` (new file)

Pure function tests with inline accelerator fixtures (don't rely on the real registry):

```ts
const fixtures: AcceleratorSpec[] = [
  { id: 'dc1', name: 'DC1', tier: 'datacenter', /* … */ },
  { id: 'dc2', name: 'DC2', tier: 'datacenter', /* … */ },
  { id: 'con1', name: 'Con1', tier: 'consumer', /* … */ },
  { id: 'con2', name: 'Con2', tier: 'consumer', /* … */ },
]
```

Cases:
1. `(fixtures, false, [])` → only `dc1`, `dc2`
2. `(fixtures, true, [])` → all four
3. `(fixtures, false, ['con1'])` → `dc1`, `dc2`, `con1`
4. `(fixtures, false, ['dc1'])` → `dc1`, `dc2` (no duplicate — already included via tier)
5. `(fixtures, true, ['con1'])` → all four (toggle wins)
6. `(fixtures, false, ['nonexistent'])` → `dc1`, `dc2` (unknown id is a no-op)
7. `([], false, [])` → `[]`

### No new store tests

`showConsumerSkus` is a plain `writable<boolean>(false)` — no derivation logic to test.

### No component-level tests

The project doesn't have Svelte component DOM tests. Manual smoke covers the checkbox UX.

### Manual smoke (during implementation)

1. Calc tab loads → accelerator dropdown shows only the 17 datacenter SKUs in publisher groups. "Show consumer GPUs" checkbox visible next to the label, unchecked.
2. Toggle on → consumer SKUs appear in their publisher groups (NVIDIA: RTX 5090/4090 added; AMD: Radeon RX series added; Apple: M-series visible).
3. Toggle off → consumer SKUs hidden again.
4. URL `#calc?a=rtx-5090&v=sku` → RTX 5090 visible in dropdown even with toggle off.
5. Sim tab → both cluster pickers show the same filter behavior; toggling one updates both.
6. Info tab → catalog filtered to datacenter cards by default; toggle reveals consumer cards.

## Open questions deferred to v2

- **Persisted preference**: localStorage / URL state for the toggle. Skipped because the auto-show-current-selection rule handles the share-link case, and "remember my preference across reloads" is a low-value ask for v1.
- **System filtering**: the catalog's multi-accelerator systems (HGX H100, GB200 NVL72, etc.) aren't tier-filtered in v1 — they're all datacenter-style by definition. If a consumer baseboard system ever ships, revisit.
- **Per-tab independent toggles**: scope-creep; users wanting consumer everywhere or nowhere is the common case.

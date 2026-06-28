# Accelerator-Tier UI Hookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the `tier` field on `AcceleratorSpec` (landed in PR #153) as a UI filter — consumer SKUs are hidden by default in all 4 accelerator pickers (InputPanel + DisaggInputPanel × 2 + InfoPanel catalog), with a small "Show consumer GPUs" checkbox to reveal them.

**Architecture:** Pure data filtering via a new `filterByTier(accelerators, showConsumer, alwaysShowIds)` helper colocated with `orderSkus` in `src/ui/catalogOrder.ts`. A shared Svelte writable `showConsumerSkus` (in `src/ui/stores.ts`) drives all 4 sites. Each site adds a `<label class="show-consumer">` checkbox bound to the store and passes its current selection id(s) into `alwaysShowIds` so shared `?a=rtx-5090` URLs still render the selected option.

**Tech Stack:** TypeScript, Svelte 5, Vitest. Pure additive change — no engine math, no URL schema change.

## Global Constraints

- **Default**: `showConsumerSkus` starts `false` — consumer SKUs hidden everywhere on load.
- **Auto-show rule**: any accelerator id passed to `filterByTier` via `alwaysShowIds` is preserved regardless of tier. Each site passes its own current-selection id(s).
- **In-memory only**: no localStorage, no URL state.
- **CSS**: the checkbox markup is identical across all 4 sites, using the class name `show-consumer`. The class is scoped per-component (Svelte scopes by default) — duplicate the rule in each file rather than introducing a global stylesheet.

---

## File Structure

**Modified:**
- `src/ui/catalogOrder.ts` — add `filterByTier` export.
- `src/ui/stores.ts` — add `showConsumerSkus` writable.
- `src/ui/InputPanel.svelte` — wire filter through `skuGroups`, add checkbox.
- `src/ui/DisaggInputPanel.svelte` — wire filter through both cluster pickers' `skuGroups`, add 2 checkboxes.
- `src/ui/InfoPanel.svelte` — make `skuGroups` reactive, add checkbox at top of SKU section.

**New tests:**
- `test/ui/filterByTier.test.ts` — pure-function tests for the new helper.

---

## Task 1: `filterByTier` helper + `showConsumerSkus` store

**Files:**
- Modify: `src/ui/catalogOrder.ts` (add `filterByTier` export)
- Modify: `src/ui/stores.ts` (add `showConsumerSkus` writable)
- Create: `test/ui/filterByTier.test.ts`

**Interfaces:**
- Produces: `filterByTier(accelerators: AcceleratorSpec[], showConsumer: boolean, alwaysShowIds?: string[]): AcceleratorSpec[]` from `src/ui/catalogOrder.ts`. Returns a filtered array preserving the input order.
- Produces: `showConsumerSkus: Writable<boolean>` (default `false`) from `src/ui/stores.ts`.

- [ ] **Step 1: Write the failing tests**

Create `test/ui/filterByTier.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { filterByTier } from '../../src/ui/catalogOrder'
import type { AcceleratorSpec } from '../../src/engine/types'

const make = (id: string, tier: 'datacenter' | 'consumer'): AcceleratorSpec => ({
  id,
  name: id.toUpperCase(),
  vendor: 'Test',
  releaseDate: '2024-01',
  tier,
  variants: [],
})

const fixtures: AcceleratorSpec[] = [
  make('dc1', 'datacenter'),
  make('dc2', 'datacenter'),
  make('con1', 'consumer'),
  make('con2', 'consumer'),
]

describe('filterByTier', () => {
  it('shows only datacenter when showConsumer=false and no alwaysShowIds', () => {
    expect(filterByTier(fixtures, false).map(a => a.id)).toEqual(['dc1', 'dc2'])
  })

  it('shows all when showConsumer=true', () => {
    expect(filterByTier(fixtures, true).map(a => a.id)).toEqual(['dc1', 'dc2', 'con1', 'con2'])
  })

  it('preserves consumer entries listed in alwaysShowIds when showConsumer=false', () => {
    expect(filterByTier(fixtures, false, ['con1']).map(a => a.id)).toEqual(['dc1', 'dc2', 'con1'])
  })

  it('does not duplicate datacenter entries that are also in alwaysShowIds', () => {
    expect(filterByTier(fixtures, false, ['dc1']).map(a => a.id)).toEqual(['dc1', 'dc2'])
  })

  it('returns all when showConsumer=true ignoring alwaysShowIds', () => {
    expect(filterByTier(fixtures, true, ['con1']).map(a => a.id)).toEqual(['dc1', 'dc2', 'con1', 'con2'])
  })

  it('ignores unknown ids in alwaysShowIds', () => {
    expect(filterByTier(fixtures, false, ['nonexistent']).map(a => a.id)).toEqual(['dc1', 'dc2'])
  })

  it('returns empty for empty input', () => {
    expect(filterByTier([], false)).toEqual([])
  })

  it('preserves input order (no sorting)', () => {
    const shuffled = [fixtures[2], fixtures[0], fixtures[3], fixtures[1]]
    expect(filterByTier(shuffled, true).map(a => a.id)).toEqual(['con1', 'dc1', 'con2', 'dc2'])
  })
})
```

- [ ] **Step 2: Run the tests to confirm failure**

Run: `npx vitest run test/ui/filterByTier.test.ts`
Expected: FAIL — `filterByTier` not exported from `src/ui/catalogOrder.ts`.

- [ ] **Step 3: Add `filterByTier` to `src/ui/catalogOrder.ts`**

Open `src/ui/catalogOrder.ts`. After the existing `orderSkus` export (or wherever the file's exports end), append:

```ts
// Filter accelerators by tier, always preserving explicitly-selected ids so
// a shared `?a=rtx-5090` URL still renders that option even when the
// consumer toggle is off. Pure / order-preserving — no sort, no dedup.
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

If `AcceleratorSpec` is not already imported in `catalogOrder.ts`, add `import type { AcceleratorSpec } from '../engine/types'` near the existing imports.

- [ ] **Step 4: Add `showConsumerSkus` writable to `src/ui/stores.ts`**

Open `src/ui/stores.ts`. Find a logical insertion point near other UI-state writables (e.g., near `showMath` or `concurrencyOverride`). Add:

```ts
// In-memory toggle: when true, consumer-tier accelerators (RTX 4090/5090,
// Apple M-series, Radeon RX, etc.) appear in the accelerator pickers and
// the Info-tab catalog. Default false — the calc app is serving-focused so
// datacenter SKUs are the primary surface. URL state isn't needed; the
// auto-show-current-selection rule in `filterByTier` handles shared links
// that point to a consumer SKU.
export const showConsumerSkus = writable<boolean>(false)
```

If `writable` is not already imported in that section, it should be from the existing `import { writable, derived, type Readable } from 'svelte/store'` line at the top of the file.

- [ ] **Step 5: Run the tests to confirm pass + full check**

Run: `npx vitest run test/ui/filterByTier.test.ts`
Expected: PASS — all 8 tests green.

Run: `npm run check 2>&1 | tail -3 && npm test 2>&1 | grep "Tests " | tail -1`
Expected: 0 type errors, full suite passes (delta = +8 tests from this task).

- [ ] **Step 6: Commit**

```bash
git add src/ui/catalogOrder.ts src/ui/stores.ts test/ui/filterByTier.test.ts
git commit -m "feat(calc): filterByTier helper + showConsumerSkus store"
```

---

## Task 2: Wire the filter through all 4 picker sites + checkbox UI

**Files:**
- Modify: `src/ui/InputPanel.svelte`
- Modify: `src/ui/DisaggInputPanel.svelte`
- Modify: `src/ui/InfoPanel.svelte`

**Interfaces:**
- Consumes: `filterByTier(accelerators, showConsumer, alwaysShowIds)` from `src/ui/catalogOrder.ts` (Task 1).
- Consumes: `showConsumerSkus: Writable<boolean>` from `src/ui/stores.ts` (Task 1).

- [ ] **Step 1: InputPanel — import the helper + store, wire `skuGroups`, add checkbox**

Open `src/ui/InputPanel.svelte`.

In the `<script>` block:
- Add `showConsumerSkus` to the existing `stores` import.
- Add `filterByTier` to the existing `catalogOrder` import.
- Find the line `const skuGroups = orderSkus(ACCELERATORS, SYSTEMS)` (around line 31) and replace with the reactive form:

```ts
$: skuGroups = orderSkus(
  filterByTier(ACCELERATORS, $showConsumerSkus, [$acceleratorId]),
  SYSTEMS
)
```

In the markup, find the existing `<label>Accelerator <select>...</select></label>` block (around line 142). Immediately after the closing `</label>`, add a sibling label for the checkbox:

```svelte
<label class="show-consumer">
  <input type="checkbox" bind:checked={$showConsumerSkus} />
  Show consumer GPUs
</label>
```

In the `<style>` block at the bottom of the file, add the matching CSS rule:

```css
.show-consumer {
  display: inline-flex; align-items: center; gap: 0.3rem;
  margin-left: 0.6rem;
  font-size: 0.78rem; font-weight: 400; color: #666;
}
.show-consumer input[type=checkbox] { width: auto; margin: 0; }
```

- [ ] **Step 2: DisaggInputPanel — wire both cluster pickers and add 2 checkboxes**

Open `src/ui/DisaggInputPanel.svelte`.

In the `<script>` block:
- Add `showConsumerSkus` to the existing `stores` import.
- Add `filterByTier` to the existing `catalogOrder` import.
- Find the `$: skuGroups = orderSkus(ACCELERATORS, SYSTEMS)` line (around line 24) and replace with TWO reactive declarations — one per cluster — so each picker keeps its own currently-selected id in `alwaysShowIds`:

```ts
$: prefillSkuGroups = orderSkus(
  filterByTier(ACCELERATORS, $showConsumerSkus,
    [$prefillAcceleratorId || $acceleratorId]),
  SYSTEMS
)
$: decodeSkuGroups = orderSkus(
  filterByTier(ACCELERATORS, $showConsumerSkus,
    [$decodeAcceleratorId || $prefillAcceleratorId || $acceleratorId]),
  SYSTEMS
)
```

In the markup, the file currently uses `{#each skuGroups as g}` in both cluster picker blocks (lines ~132 and ~165). Update each to use the cluster-specific name:
- The prefill cluster's accelerator dropdown iterates `prefillSkuGroups`.
- The decode cluster's accelerator dropdown iterates `decodeSkuGroups`.

After each cluster's `</label>` that wraps the accelerator dropdown (one for prefill, one for decode), add the show-consumer checkbox:

```svelte
<label class="show-consumer">
  <input type="checkbox" bind:checked={$showConsumerSkus} />
  Show consumer GPUs
</label>
```

In the `<style>` block, append the same CSS:

```css
.show-consumer {
  display: inline-flex; align-items: center; gap: 0.3rem;
  margin-left: 0.6rem;
  font-size: 0.78rem; font-weight: 400; color: #666;
}
.show-consumer input[type=checkbox] { width: auto; margin: 0; }
```

- [ ] **Step 3: InfoPanel — make `skuGroups` reactive, add checkbox above the catalog**

Open `src/ui/InfoPanel.svelte`.

In the `<script>` block:
- Add `showConsumerSkus` to the existing `stores` import.
- Add `filterByTier` to the existing `catalogOrder` import.
- Find the line `const skuGroups = orderSkus(ACCELERATORS, SYSTEMS)` (around line 18) and change it from a non-reactive `const` to a reactive `$:` so the filter responds to the store:

```ts
$: skuGroups = orderSkus(
  filterByTier(ACCELERATORS, $showConsumerSkus, []),
  SYSTEMS
)
```

The downstream `$: skuByPublisher = ...` and `$: skuColumns = ...` already use `$:` so they'll re-run on each filter change — no further script changes needed.

In the markup, find the `{#if effSection === 'skus'}` branch (use grep / scan — the file is medium-sized). At the top of the SKU catalog section content, add the checkbox:

```svelte
<label class="show-consumer catalog-toggle">
  <input type="checkbox" bind:checked={$showConsumerSkus} />
  Show consumer GPUs
</label>
```

In the `<style>` block, append:

```css
.show-consumer {
  display: inline-flex; align-items: center; gap: 0.3rem;
  font-size: 0.78rem; font-weight: 400; color: #666;
}
.show-consumer input[type=checkbox] { width: auto; margin: 0; }
.catalog-toggle { margin-bottom: 0.6rem; }
```

The extra `catalog-toggle` modifier gives this site its own spacing (top of catalog rather than inline-margin-left).

- [ ] **Step 4: Type check + tests + build**

Run: `npm run check 2>&1 | tail -3`
Expected: 0 errors, 0 warnings.

Run: `npm test 2>&1 | grep "Tests " | tail -1`
Expected: full suite passes (same count as Task 1 — no new tests added in this task).

Run: `npm run build 2>&1 | tail -3`
Expected: clean build.

- [ ] **Step 5: Manual smoke**

Run `npm run dev` and open the Calc tab in a browser.

1. **InputPanel checkbox**: visible next to the "Accelerator" dropdown label, unchecked. Dropdown shows only datacenter SKUs (17 entries grouped by publisher; no RTX 4090/5090, no Apple M-series, no Radeon RX).
2. **Toggle on**: all 38 SKUs appear in their publisher groups.
3. **Toggle off**: consumer SKUs disappear from the dropdown.
4. **Auto-show via URL**: visit `http://localhost:5173/#calc?a=rtx-5090&v=sku&m=llama-3.3-70b&w=bf16&kv=fp16&ac=bf16&pt=2048&ot=512` with toggle off → RTX 5090 visible in dropdown.
5. **Sim tab**: open. DisaggInputPanel's prefill and decode cluster pickers each show the same filter behavior. Toggling the checkbox on one updates both (shared store).
6. **Info tab → SKUs**: catalog filtered to datacenter only by default. Checkbox at top reveals consumer cards. The 3-column publisher layout still renders (NVIDIA / Google-AWS-Cerebras / Intel-rest).

Stop the dev server when done.

- [ ] **Step 6: Commit**

```bash
git add src/ui/InputPanel.svelte src/ui/DisaggInputPanel.svelte src/ui/InfoPanel.svelte
git commit -m "feat(calc): wire showConsumerSkus + filterByTier through accelerator pickers and Info catalog"
```

---

## Self-review notes

- Spec coverage: filterByTier helper (Task 1) + store (Task 1) + 4 picker sites (Task 2: InputPanel + 2 cluster sites in DisaggInputPanel + InfoPanel catalog) — all covered.
- The auto-show-current-selection rule is implemented via `alwaysShowIds` and tested in Task 1.
- No new URL keys, no schema changes, no engine math touched.
- Total: ~50 LOC across 5 files + 1 new test file (~50 LOC), 2 commits.

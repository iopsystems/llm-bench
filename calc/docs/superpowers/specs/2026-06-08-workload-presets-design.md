# Workload Presets — Design

## Goal

Add a benchmark-preset dropdown to the Workload fieldset in `InputPanel.svelte` so a user can pick a publicly-known workload (HumanEval, MBPP, LiveCodeBench, SWE-Bench Verified, etc.) instead of hand-entering token counts. Selecting a preset fills the existing prompt/output token inputs with sourced median values; the user can still edit either field afterward and the dropdown silently flips to "Custom" when values diverge from any preset.

Out of scope for v1: multi-turn presets (the Sim-tab multi-turn item in the calc roadmap covers that), distributional workloads (variable prompt/output sizes), suggested concurrency per preset, automatic tokenizer-aware re-computation.

## Why a preset menu

Two real costs the existing UI imposes:
- Token counts for benchmarks aren't memorized — users either guess or skip running "what would HumanEval look like on this hardware" because they don't want to look up the numbers.
- Hand-entered numbers aren't shareable as benchmark intent — a URL with `pt=150&ot=200` doesn't communicate "this is HumanEval" to anyone receiving it.

The preset menu solves both: a single click loads sourced numbers, and the UI surfaces the source as a citation alongside.

## Architecture

### Module layout

```
src/data/workload-presets.ts       # WORKLOAD_PRESETS registry
src/ui/WorkloadPresetPicker.svelte # the <select>, source caption, divergence detection
src/ui/InputPanel.svelte           # mounts the picker inside the Workload fieldset
test/data/workload-presets.test.ts # registry sanity
test/ui/workload-preset-matching.test.ts # matchPreset() pure function
```

### Data flow

The picker reads the existing `workload` store and writes to it. No new stores, no new URL keys.

- **On select**: `WorkloadPresetPicker` writes the chosen preset's `promptTokens` + `outputTokens` into the `workload` store.
- **On render**: the picker's currently-selected value is computed reactively from the current workload via `matchPreset({ promptTokens, outputTokens }, WORKLOAD_PRESETS)`. If a preset's values match exactly, the picker shows that preset; otherwise it shows "Custom".
- **URL state**: `pt=` and `ot=` continue to be the source of truth. A shared link whose `(pt, ot)` happens to match a preset auto-selects that preset in the recipient's UI via the same `matchPreset` derivation — no `wp=` URL key needed.

Trade-off accepted: if a preset's sourced values change in a future release, old shared URLs stop auto-selecting that preset. Acceptable — sourced workload sizes are rarely revised.

## Preset schema

```ts
export interface WorkloadPreset {
  id: string                    // slug, e.g. 'humaneval'
  name: string                  // display name shown in the dropdown, e.g. 'HumanEval'
  group: 'code-gen' | 'other'   // for <optgroup> rendering
  promptTokens: number          // sourced median, tokenized with Llama-3 reference tokenizer
  outputTokens: number          // sourced median
  sourceUrl: string             // citation URL (HF dataset card or canonical paper)
  sourceAccessedAt: string      // YYYY-MM-DD
  description: string           // one-line context shown as a tooltip / aria-label
}
```

## Initial preset list (v1 candidates)

**Code-gen:**
- HumanEval — function-completion, small problems
- HumanEval+ (EvalPlus) — same shape, enhanced test coverage
- MBPP — Python programming problems
- MBPP+ (EvalPlus) — enhanced
- LiveCodeBench — competitive-programming-style problems
- SWE-Bench Verified — real GitHub issues with repo context

**Other (contrasting anchors):**
- Chat (ShareGPT-shape median)
- MMLU — short multiple-choice, tiny
- LongBench (summarization subset) — long prompts, short outputs
- AlpacaEval — instruction-following

11 candidates. Actual token counts and `sourceUrl` values get sourced at implementation time from HuggingFace dataset cards or the canonical paper, tokenized against Llama-3 as the reference. The `verifying-achievable-perf-numbers` skill ethos applies: no fabricated citations, no inferred-from-adjacent-knowledge numbers. If a preset can't be cleanly sourced at implementation time, drop it from v1 rather than ship a fuzzy number.

## UI

### Layout

Inside the existing `<fieldset class="island"><legend>Workload</legend>` in `InputPanel.svelte`, add a new row above the current prompt/output/concurrency row:

```
┌─ Workload ──────────────────────────────────────────────────────────┐
│  Benchmark preset                                                   │
│  [ HumanEval                                              ▾ ]       │
│  Source: huggingface.co/datasets/openai_humaneval (as of 2026-06-07)│ ← caption, only when preset active
│                                                                      │
│  Prompt tokens   Output tokens   Concurrency                         │
│  [ 150       ]   [ 200       ]   [ auto (12)  ]                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Picker markup

```svelte
<label class="preset-row">
  Benchmark preset
  <select value={selectedPresetId} on:change={onPresetChange}>
    <option value="custom">Custom</option>
    <optgroup label="Code-gen">
      {#each codeGenPresets as p}
        <option value={p.id} title={p.description}>{p.name}</option>
      {/each}
    </optgroup>
    <optgroup label="Other">
      {#each otherPresets as p}
        <option value={p.id} title={p.description}>{p.name}</option>
      {/each}
    </optgroup>
  </select>
</label>
{#if activePreset}
  <div class="preset-source">
    Source: <a href={activePreset.sourceUrl} target="_blank" rel="noopener">{prettyHost(activePreset.sourceUrl)}</a>
    (as of {activePreset.sourceAccessedAt}; tokenized with Llama-3 reference tokenizer, ±10–20% on other tokenizers)
  </div>
{/if}
```

### Style

- Picker uses the same `<select>` styling as the existing accelerator/model/variant pickers in the file (matches `font-size: 1rem; padding: 0.25rem; min-width: 180px` baseline). Width: full width of its row.
- Source caption: `font-size: 0.78rem; color: #666` (matches the other caption styling already in use across the calc UI).
- Link uses default browser link styling — no special treatment needed.
- Mobile: the `<select>` uses native UA picker, optgroups supported on all major mobile browsers.

### Divergence detection

```ts
// Pure function — exported for testing.
export function matchPreset(
  workload: Pick<Workload, 'promptTokens' | 'outputTokens'>,
  presets: WorkloadPreset[]
): WorkloadPreset['id'] | 'custom' {
  const m = presets.find(
    p => p.promptTokens === workload.promptTokens
      && p.outputTokens === workload.outputTokens
  )
  return m?.id ?? 'custom'
}
```

The picker computes its `selectedPresetId` reactively from `matchPreset($workload, WORKLOAD_PRESETS)`. When the user types in the prompt or output inputs, the workload store updates, the reactive re-runs, the picker silently flips to "Custom". No flicker, no extra state, no "reset" affordance needed — the dropdown itself is the affordance back to a preset.

### On preset change

```ts
function onPresetChange(e: Event) {
  const id = (e.target as HTMLSelectElement).value
  if (id === 'custom') return  // selecting "Custom" is a no-op; the inputs stay as-is
  const preset = WORKLOAD_PRESETS.find(p => p.id === id)
  if (!preset) return  // safety net for an unknown id (e.g. a removed preset)
  workload.update(w => ({
    ...w,
    promptTokens: preset.promptTokens,
    outputTokens: preset.outputTokens,
  }))
}
```

Concurrency is not touched — it's its own input with its own auto/override logic.

## URL state

No schema change. `pt=` and `ot=` remain the source of truth for workload state. The picker's selection is purely derived from the current values.

Backward compat: existing shared URLs without any presets continue to work identically. New URLs naturally auto-select a preset when the values match.

## Testing

### `test/data/workload-presets.test.ts`

For each preset in `WORKLOAD_PRESETS`:
- `id` is a non-empty string, no whitespace.
- `name` is non-empty.
- `group` is `'code-gen'` or `'other'`.
- `promptTokens > 0` and `outputTokens > 0`, both integers.
- `sourceUrl` parses via `new URL(...)` without throwing.
- `sourceAccessedAt` matches `/^\d{4}-\d{2}-\d{2}$/`.
- `description` is non-empty.

Top-level:
- All `id` values are unique.
- All `(promptTokens, outputTokens)` pairs are unique (otherwise `matchPreset` ambiguity).

### `test/ui/workload-preset-matching.test.ts`

`matchPreset` pure function:
- Exact match on `(promptTokens, outputTokens)` returns that preset's id.
- One-token deviation in `promptTokens` returns `'custom'`.
- One-token deviation in `outputTokens` returns `'custom'`.
- Empty preset list returns `'custom'`.
- Workload with `promptTokens === preset.promptTokens` but `outputTokens === some_other_preset.outputTokens` returns `'custom'` (no partial match).

### `test/ui/share.test.ts`

No new tests required — URL state unchanged. Optionally add one assertion that an existing round-trip test with `pt=150&ot=200` continues to land in a state where the picker would derive HumanEval (if HumanEval ends up with those values), but this is mostly a no-op since the share layer doesn't know about presets.

### Manual smoke

1. Open Calc tab → picker shows "Custom" (default 2048/512 doesn't match any preset).
2. Select HumanEval → prompt + output fields update; picker stays on HumanEval; source caption appears with link.
3. Tweak prompt by 1 → picker flips to "Custom"; caption disappears.
4. Re-select HumanEval → preset values restored.
5. Share the URL → open in a fresh tab → picker auto-selects HumanEval, caption shown.
6. Switch to Sim tab → picker also visible there (InputPanel is shared); selecting a preset on Sim updates Calc too.

## Open questions deferred to v2

- **Tokenizer-aware values.** v1 fixes the tokenizer to Llama-3. A future version could let the user pick a tokenizer and rescale preset values accordingly.
- **Suggested concurrency per preset.** Some benchmarks have canonical batch sizes (e.g. SWE-Bench with N=8). Could add later if it proves useful; v1 leaves concurrency on its own auto/override logic.
- **Multi-turn presets.** MT-Bench, agentic trajectories. Covered by the broader "Multi-turn workload support" item in the calc roadmap — not in scope here.
- **Distribution-shaped presets.** A preset could carry a prompt/output *distribution* (lognormal mean + sigma) instead of a single point. Re-introduces percentile sim work (deferred from disagg load section). Not in v1.

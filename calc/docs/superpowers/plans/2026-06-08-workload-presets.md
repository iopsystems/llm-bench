# Workload Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a benchmark-preset dropdown to the Workload fieldset in `InputPanel.svelte` so a user can pick a publicly-known benchmark (HumanEval, MBPP, LiveCodeBench, SWE-Bench Verified, plus a few non-code-gen anchors) instead of hand-entering token counts. The picker fills `promptTokens` + `outputTokens` from sourced medians and silently flips to "Custom" when the user edits the inputs.

**Architecture:** A pure-data registry in `src/data/workload-presets.ts` (no engine math changes). A new `<WorkloadPresetPicker>` Svelte component mounted in `InputPanel.svelte`'s Workload fieldset, above the existing prompt/output/concurrency row. URL state is unchanged — `pt=` and `ot=` remain the source of truth; the picker derives its selection by matching current values against the registry.

**Tech Stack:** TypeScript, Svelte 5, Vite, Vitest. Follows the existing data-registry pattern from `src/data/models.ts`.

---

## File structure

**New files:**
- `src/data/workload-presets.ts` — exports `WorkloadPreset` interface, `WORKLOAD_PRESETS: WorkloadPreset[]`, and pure helper `matchPreset(workload, presets)`.
- `src/ui/WorkloadPresetPicker.svelte` — the `<select>` with optgroups and the source-caption strip.
- `test/data/workload-presets.test.ts` — schema validation across the registry.
- `test/ui/workload-preset-matching.test.ts` — `matchPreset` behavior tests (uses inline fixtures, not the real registry).

**Modified files:**
- `src/ui/InputPanel.svelte` — mount `<WorkloadPresetPicker />` inside the Workload fieldset, above the prompt/output/concurrency row.

---

## Task 1: Schema, `matchPreset` helper, empty registry

**Files:**
- Create: `src/data/workload-presets.ts`
- Create: `test/data/workload-presets.test.ts`
- Create: `test/ui/workload-preset-matching.test.ts`

- [ ] **Step 1: Write failing matchPreset tests**

Create `test/ui/workload-preset-matching.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { matchPreset, type WorkloadPreset } from '../../src/data/workload-presets'

const fixtures: WorkloadPreset[] = [
  { id: 'p1', name: 'P1', group: 'code-gen', promptTokens: 100, outputTokens: 200,
    sourceUrl: 'https://example.com/p1', sourceAccessedAt: '2026-06-08', description: 'fixture 1' },
  { id: 'p2', name: 'P2', group: 'other', promptTokens: 500, outputTokens: 500,
    sourceUrl: 'https://example.com/p2', sourceAccessedAt: '2026-06-08', description: 'fixture 2' },
]

describe('matchPreset', () => {
  it('returns the preset id when prompt+output match exactly', () => {
    expect(matchPreset({ promptTokens: 100, outputTokens: 200 }, fixtures)).toBe('p1')
    expect(matchPreset({ promptTokens: 500, outputTokens: 500 }, fixtures)).toBe('p2')
  })

  it("returns 'custom' when prompt is off by 1", () => {
    expect(matchPreset({ promptTokens: 101, outputTokens: 200 }, fixtures)).toBe('custom')
  })

  it("returns 'custom' when output is off by 1", () => {
    expect(matchPreset({ promptTokens: 100, outputTokens: 201 }, fixtures)).toBe('custom')
  })

  it("returns 'custom' on empty preset list", () => {
    expect(matchPreset({ promptTokens: 100, outputTokens: 200 }, [])).toBe('custom')
  })

  it("returns 'custom' when prompt matches one preset and output matches another (no partial match)", () => {
    // promptTokens=100 (from p1) and outputTokens=500 (from p2) — no preset has both.
    expect(matchPreset({ promptTokens: 100, outputTokens: 500 }, fixtures)).toBe('custom')
  })
})
```

- [ ] **Step 2: Run test, verify failure**

Run: `npx vitest run test/ui/workload-preset-matching.test.ts`
Expected: FAIL — module `src/data/workload-presets` not found.

- [ ] **Step 3: Implement schema + helper + empty registry**

Create `src/data/workload-presets.ts`:

```ts
// Public-benchmark workload presets surfaced in the Calc/Sim Workload picker.
// Each preset carries sourced median (promptTokens, outputTokens) values so the
// user can pick "HumanEval" instead of hand-entering numbers. Values are
// tokenized against the Llama-3 reference tokenizer; assume ±10–20% variance
// on other tokenizers.

export interface WorkloadPreset {
  id: string                    // slug; URL-safe; must be unique within the registry
  name: string                  // display name in the dropdown
  group: 'code-gen' | 'other'   // for <optgroup> rendering
  promptTokens: number          // sourced median, positive integer
  outputTokens: number          // sourced median, positive integer
  sourceUrl: string             // citation URL (HF dataset card or canonical paper)
  sourceAccessedAt: string      // YYYY-MM-DD when the source was fetched
  description: string           // short one-line context, used as <option title>
}

// Pure helper — exported for testing. Returns the id of the preset whose
// promptTokens AND outputTokens both exactly match the provided workload,
// else 'custom'. The picker's reactive selection uses this.
export function matchPreset(
  workload: { promptTokens: number; outputTokens: number },
  presets: WorkloadPreset[]
): string {
  const m = presets.find(
    p => p.promptTokens === workload.promptTokens
      && p.outputTokens === workload.outputTokens
  )
  return m?.id ?? 'custom'
}

export const WORKLOAD_PRESETS: WorkloadPreset[] = []
```

- [ ] **Step 4: Run matchPreset tests, verify pass**

Run: `npx vitest run test/ui/workload-preset-matching.test.ts`
Expected: PASS — all 5 tests green.

- [ ] **Step 5: Write registry sanity tests**

Create `test/data/workload-presets.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { WORKLOAD_PRESETS } from '../../src/data/workload-presets'

describe('WORKLOAD_PRESETS schema', () => {
  it('every preset has the required shape', () => {
    for (const p of WORKLOAD_PRESETS) {
      expect(p.id, `preset ${JSON.stringify(p)}: id`).toMatch(/^[a-z0-9-]+$/)
      expect(p.name.length, `preset ${p.id}: name non-empty`).toBeGreaterThan(0)
      expect(['code-gen', 'other'], `preset ${p.id}: group enum`).toContain(p.group)
      expect(p.promptTokens, `preset ${p.id}: promptTokens > 0`).toBeGreaterThan(0)
      expect(Number.isInteger(p.promptTokens), `preset ${p.id}: promptTokens int`).toBe(true)
      expect(p.outputTokens, `preset ${p.id}: outputTokens > 0`).toBeGreaterThan(0)
      expect(Number.isInteger(p.outputTokens), `preset ${p.id}: outputTokens int`).toBe(true)
      expect(() => new URL(p.sourceUrl), `preset ${p.id}: sourceUrl parses`).not.toThrow()
      expect(p.sourceAccessedAt, `preset ${p.id}: sourceAccessedAt YYYY-MM-DD`).toMatch(/^\d{4}-\d{2}-\d{2}$/)
      expect(p.description.length, `preset ${p.id}: description non-empty`).toBeGreaterThan(0)
    }
  })

  it('preset ids are unique', () => {
    const ids = WORKLOAD_PRESETS.map(p => p.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('(promptTokens, outputTokens) pairs are unique (matchPreset needs deterministic mapping)', () => {
    const pairs = WORKLOAD_PRESETS.map(p => `${p.promptTokens}/${p.outputTokens}`)
    expect(new Set(pairs).size).toBe(pairs.length)
  })
})
```

- [ ] **Step 6: Run registry tests, verify pass**

Run: `npx vitest run test/data/workload-presets.test.ts`
Expected: PASS — registry is empty, all tests trivially green (loops iterate zero times, all-unique on empty set is true).

- [ ] **Step 7: Commit**

```bash
git add src/data/workload-presets.ts test/data/workload-presets.test.ts test/ui/workload-preset-matching.test.ts
git commit -m "feat(calc): workload-presets — schema, matchPreset helper, empty registry"
```

---

## Task 2: Source the code-gen presets

**Files:**
- Modify: `src/data/workload-presets.ts`

This task does real research — for each candidate, fetch the HuggingFace dataset card or the canonical paper, identify the median `(promptTokens, outputTokens)` per task instance, and record the citation. **Don't fabricate numbers; if a source can't be located cleanly, drop the preset from v1.**

**Sourcing protocol per preset:**
1. Fetch the HF dataset card via WebFetch.
2. Look for dataset stats: "average prompt length", "median tokens", or per-task token counts.
3. Cross-reference against community-published numbers in well-cited serving papers (vLLM, SGLang, Sarathi, DistServe) which often report median benchmark sizes — but the canonical HF dataset card or paper wins if they disagree.
4. Tokenize against Llama-3 (`meta-llama/Meta-Llama-3-8B` tokenizer) as the reference. If the source uses a different tokenizer (cl100k_base, etc.), note the disparity in a comment but still record the value — the spec already disclaims ±10–20% variance.
5. Today's date (run `date +%Y-%m-%d`) goes in `sourceAccessedAt`.

**Code-gen candidates (target 4–6):**

| id | name | Notes / expected range | Where to look |
|---|---|---|---|
| `humaneval` | HumanEval | ~100/100 (function sig + docstring → function body) | `openai/openai_humaneval` on HF + Codex paper |
| `humaneval-plus` | HumanEval+ (EvalPlus) | Similar to HumanEval, slightly more test stub | `evalplus/humanevalplus` on HF + EvalPlus paper |
| `mbpp` | MBPP | ~150/100 (problem + example I/O → solution) | `google-research-datasets/mbpp` on HF + paper |
| `mbpp-plus` | MBPP+ (EvalPlus) | Similar to MBPP, more tests | `evalplus/mbppplus` on HF |
| `livecodebench` | LiveCodeBench | ~1.5k/800 (contest problem statement → solution) | `livecodebench/code_generation_lite` on HF + paper |
| `swe-bench-verified` | SWE-Bench Verified | ~12k/2k (issue text + retrieved files → patch). With Oracle retrieval. | `princeton-nlp/SWE-bench_Verified` on HF + paper |

If HumanEval+ and HumanEval (or MBPP+ and MBPP) end up with literally identical sourced values, only keep one (the `+` variant) — the registry forbids duplicate `(prompt, output)` pairs. If they're close-but-different, keep both.

- [ ] **Step 1: Fetch + add HumanEval**

Use WebFetch on `https://huggingface.co/datasets/openai/openai_humaneval`. Look at the dataset card description and any reported stats. Cross-reference against the Codex paper (arXiv 2107.03374, Appendix). If unsure, sanity-check against vLLM or SGLang serving-paper benchmark tables (which typically report HumanEval median in/out).

Add to `WORKLOAD_PRESETS` in `src/data/workload-presets.ts`:

```ts
{
  id: 'humaneval',
  name: 'HumanEval',
  group: 'code-gen',
  promptTokens: <SOURCED>,
  outputTokens: <SOURCED>,
  sourceUrl: 'https://huggingface.co/datasets/openai/openai_humaneval',
  sourceAccessedAt: '<TODAY>',
  description: '164 Python function-completion problems from OpenAI Codex paper',
},
```

Replace `<SOURCED>` with the actual numbers and `<TODAY>` with today's date in YYYY-MM-DD. If you cannot find a defensible value, do NOT add the entry — note the gap in your status report.

- [ ] **Step 2: Run registry tests, verify pass**

Run: `npx vitest run test/data/workload-presets.test.ts`
Expected: PASS.

- [ ] **Step 3: Fetch + add MBPP, LiveCodeBench, SWE-Bench Verified, plus the `+` variants if sourceable**

Repeat the sourcing protocol for each candidate. Add entries to the registry in the order listed above. After each addition, run `npx vitest run test/data/workload-presets.test.ts` to catch schema regressions.

For SWE-Bench specifically: the prompt size depends heavily on the retrieval setup. Use "Oracle retrieval" (the canonical evaluation) median values from the SWE-Bench paper or the leaderboard's documentation, NOT the raw issue-text size.

For LiveCodeBench: pick the `code_generation_lite` subset (most commonly cited in serving papers). Document the subset choice in the `description` field.

- [ ] **Step 4: Run all tests after each addition**

Run: `npm test 2>&1 | grep "Tests "`
Expected: count increases by 1 per registry add (no — actually the count stays the same; the schema test iterates the registry but is one test). Total should still be all green; the schema test now exercises each preset entry.

- [ ] **Step 5: Commit**

```bash
git add src/data/workload-presets.ts
git commit -m "feat(calc): workload-presets — code-gen entries (HumanEval, MBPP, LiveCodeBench, SWE-Bench Verified, +variants)"
```

Status report: enumerate which presets were added with their sourced values + citations. Flag any that couldn't be sourced and were dropped.

---

## Task 3: Source the non-code-gen anchor presets

**Files:**
- Modify: `src/data/workload-presets.ts`

Same protocol as Task 2 but for the four anchor presets that contrast with code-gen.

**Anchor candidates (target 3–4):**

| id | name | Notes / expected range | Where to look |
|---|---|---|---|
| `chat-typical` | Chat (typical) | ~500/500 (single-turn conversational) | ShareGPT dataset stats + serving papers (vLLM, SGLang report this median) |
| `mmlu` | MMLU | ~50/5 (multiple-choice, tiny output) | `cais/mmlu` on HF + Hendrycks paper |
| `longbench-summary` | LongBench (summarization) | ~10k/300 (long-context summarization) | `THUDM/LongBench` on HF — pick the `gov_report` subtask |
| `alpaca-eval` | AlpacaEval | ~50/200 (instruction-following) | `tatsu-lab/alpaca_eval` on HF + Alpaca paper |

For Chat-typical: the canonical source is ShareGPT-shape stats reported in serving papers; there's no single "official" Chat benchmark. Use the most-cited median from the vLLM paper (Kwon et al. 2023) or SGLang paper as the sourceUrl. If the paper doesn't have a stable URL, use the GitHub README link.

For LongBench: pick ONE subtask (recommend `gov_report` for summarization; or `narrativeqa` for QA — but stick with one). Document the choice in the description field.

- [ ] **Step 1: Fetch + add Chat-typical**

Add entry. The promptTokens/outputTokens should reflect a SINGLE-TURN chat — not a full conversation.

```ts
{
  id: 'chat-typical',
  name: 'Chat (typical)',
  group: 'other',
  promptTokens: <SOURCED>,
  outputTokens: <SOURCED>,
  sourceUrl: <serving paper URL or GH README>,
  sourceAccessedAt: '<TODAY>',
  description: 'Single-turn conversational query, ShareGPT-shape median',
},
```

Run `npx vitest run test/data/workload-presets.test.ts`.

- [ ] **Step 2: Fetch + add MMLU, LongBench, AlpacaEval**

Repeat protocol. After each addition, run the registry test.

- [ ] **Step 3: Run full test suite**

Run: `npm test 2>&1 | grep "Tests "`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add src/data/workload-presets.ts
git commit -m "feat(calc): workload-presets — anchor entries (Chat, MMLU, LongBench, AlpacaEval)"
```

Status report: enumerate added entries + citations.

---

## Task 4: `WorkloadPresetPicker.svelte` component

**Files:**
- Create: `src/ui/WorkloadPresetPicker.svelte`

The component reads the `workload` store, derives the currently-selected preset via `matchPreset`, renders a `<select>` with optgroups, and writes back on change. Source caption shown when a preset is active.

- [ ] **Step 1: Create the component**

Create `src/ui/WorkloadPresetPicker.svelte`:

```svelte
<script lang="ts">
  import { workload } from './stores'
  import { WORKLOAD_PRESETS, matchPreset } from '../data/workload-presets'

  $: selectedPresetId = matchPreset(
    { promptTokens: $workload.promptTokens, outputTokens: $workload.outputTokens },
    WORKLOAD_PRESETS
  )
  $: activePreset = WORKLOAD_PRESETS.find(p => p.id === selectedPresetId)

  $: codeGenPresets = WORKLOAD_PRESETS.filter(p => p.group === 'code-gen')
  $: otherPresets   = WORKLOAD_PRESETS.filter(p => p.group === 'other')

  function onPresetChange(e: Event) {
    const id = (e.target as HTMLSelectElement).value
    if (id === 'custom') return
    const preset = WORKLOAD_PRESETS.find(p => p.id === id)
    if (!preset) return
    workload.update(w => ({
      ...w,
      promptTokens: preset.promptTokens,
      outputTokens: preset.outputTokens,
    }))
  }

  function prettyHost(url: string): string {
    try {
      const u = new URL(url)
      return u.host + u.pathname.replace(/\/+$/, '')
    } catch {
      return url
    }
  }
</script>

<label class="preset-row">
  Benchmark preset
  <select value={selectedPresetId} on:change={onPresetChange}>
    <option value="custom">Custom</option>
    {#if codeGenPresets.length > 0}
      <optgroup label="Code-gen">
        {#each codeGenPresets as p}
          <option value={p.id} title={p.description}>{p.name}</option>
        {/each}
      </optgroup>
    {/if}
    {#if otherPresets.length > 0}
      <optgroup label="Other">
        {#each otherPresets as p}
          <option value={p.id} title={p.description}>{p.name}</option>
        {/each}
      </optgroup>
    {/if}
  </select>
</label>
{#if activePreset}
  <div class="preset-source">
    Source: <a href={activePreset.sourceUrl} target="_blank" rel="noopener">{prettyHost(activePreset.sourceUrl)}</a>
    &middot; as of {activePreset.sourceAccessedAt}
    &middot; tokenized with Llama-3 reference tokenizer (±10–20% on other tokenizers)
  </div>
{/if}

<style>
  .preset-row {
    display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.9rem;
  }
  .preset-row select {
    font-size: 1rem; padding: 0.25rem; min-width: 220px;
  }
  .preset-source {
    margin-top: 0.3rem;
    font-size: 0.78rem; color: #666;
    line-height: 1.4;
  }
  .preset-source a { color: #2b6cb0; }
</style>
```

- [ ] **Step 2: Verify type check**

Run: `npm run check 2>&1 | tail -3`
Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```bash
git add src/ui/WorkloadPresetPicker.svelte
git commit -m "feat(calc): WorkloadPresetPicker — preset dropdown + source caption"
```

---

## Task 5: Mount the picker in `InputPanel.svelte`

**Files:**
- Modify: `src/ui/InputPanel.svelte`

Mount `<WorkloadPresetPicker />` inside the `<fieldset class="island"><legend>Workload</legend>` block, above the existing `<div class="row">` that holds the prompt/output/concurrency inputs.

- [ ] **Step 1: Add the import**

Edit `src/ui/InputPanel.svelte`. In the `<script>` block, add:

```ts
import WorkloadPresetPicker from './WorkloadPresetPicker.svelte'
```

Place this near the other component imports (e.g. next to `import ParallelismPicker from './ParallelismPicker.svelte'`).

- [ ] **Step 2: Mount the picker**

Find the existing block (approximately around line 216-217):

```svelte
<fieldset class="island">
  <legend>Workload</legend>
  <div class="row">
    <label>
      Prompt tokens
```

Insert `<WorkloadPresetPicker />` between the `<legend>` and the existing `<div class="row">`:

```svelte
<fieldset class="island">
  <legend>Workload</legend>
  <WorkloadPresetPicker />
  <div class="row">
    <label>
      Prompt tokens
```

- [ ] **Step 3: Verify check + tests + build**

Run: `npm run check 2>&1 | tail -3 && npm test 2>&1 | grep "Tests " | tail -1 && npm run build 2>&1 | tail -3`
Expected: 0 type errors, all tests pass, build succeeds.

- [ ] **Step 4: Commit**

```bash
git add src/ui/InputPanel.svelte
git commit -m "feat(calc): mount WorkloadPresetPicker in InputPanel's Workload fieldset"
```

---

## Task 6: Whole-feature verification + manual smoke

**Files:** all of the above (verification pass — no new code unless smoke turns up a bug).

- [ ] **Step 1: Full automated verification**

Run: `npm run check 2>&1 | tail -3 && npm test 2>&1 | grep "Tests " | tail -1 && npm run build 2>&1 | tail -3`
Expected: 0 type errors, all tests pass (registry tests + matchPreset tests added by Tasks 1–3), build succeeds.

- [ ] **Step 2: Manual smoke**

Run `npm run dev` and open the Calc tab in a browser.

Verify:
1. Workload fieldset shows the new "Benchmark preset" picker above the prompt/output/concurrency inputs.
2. Picker defaults to "Custom" (since the default workload 2048/512 doesn't match any preset).
3. Select HumanEval (or whichever preset Task 2 actually added with sourceable values). Verify:
   - Prompt tokens and Output tokens inputs update to the preset's values.
   - Picker stays selected on HumanEval.
   - Source caption appears below the picker with a clickable link.
4. Increment the Prompt tokens by 1. Verify:
   - Picker silently flips to "Custom".
   - Source caption disappears.
5. Re-select HumanEval. Verify the inputs revert.
6. Switch to Sim tab. Verify:
   - Picker is also visible there (InputPanel is shared).
   - Picker still shows HumanEval (state is shared via the workload store).
7. Reload the URL after picking HumanEval. Verify the new tab shows HumanEval auto-selected (URL `pt=` and `ot=` match HumanEval's values).
8. Pick a preset in the "Other" optgroup (e.g. Chat). Verify the inputs and picker update, source caption updates.

Stop the dev server when done.

- [ ] **Step 3: Report status**

Summarize: which presets actually shipped, which were dropped (and why), test count delta, any UX rough edges noticed during smoke.

Then invoke the `superpowers:finishing-a-development-branch` skill to wrap the branch (per project convention: ask the user before pushing / creating the PR).

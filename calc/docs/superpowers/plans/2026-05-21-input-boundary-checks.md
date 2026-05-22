# Input boundary checks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all three workload inputs (prompt tokens, output tokens, concurrency) snap `0` to `1` silently and reject negatives with the existing invalid badge — so user input is never silently ignored.

**Architecture:** One contract: `parseTokenCount` returns the effective (clamped) positive integer or `null` (rejected). Parser change makes `0`/`0k` snap to `1`; negatives keep returning `null` via the existing regex non-match path. Concurrency input switches from `bind:value` on a bare `<input type="number">` to the same parsed-input pattern prompt/output already use, reusing `parseTokenCount`.

**Tech Stack:** TypeScript + Svelte 5; Vitest; npm from `calc/`; git from repo root `/Users/yao/workspace/llm-perf`. Branch `fix/input-boundary-checks` (spec already committed).

**Spec:** `calc/docs/superpowers/specs/2026-05-21-input-boundary-checks-design.md`

---

### Task 1: `parseTokenCount` snaps non-positive numeric input to 1

**Files:**
- Modify: `calc/src/ui/parseTokens.ts:13` (the final guard)
- Modify: `calc/test/ui/parseTokens.test.ts:39-42` (existing "rejects zero and negative results" case)

- [ ] **Step 1: Update the test (TDD — change the contract first)**

In `calc/test/ui/parseTokens.test.ts`, find the existing block:
```ts
  it('rejects zero and negative results (min 1 token)', () => {
    expect(parseTokenCount('0')).toBeNull()
    expect(parseTokenCount('0k')).toBeNull()
  })
```
Replace it with two distinct cases (the new contract: snap `0` to `1`; reject malformed/negative as before):
```ts
  it('snaps non-positive numeric input to 1 (input is never silently ignored)', () => {
    expect(parseTokenCount('0')).toBe(1)
    expect(parseTokenCount('0k')).toBe(1)
    expect(parseTokenCount('0.4')).toBe(1)     // rounds to 0, then snaps to 1
  })
  it('rejects negative or malformed input (returns null → caller shows invalid badge)', () => {
    expect(parseTokenCount('-5')).toBeNull()    // regex won't match the sign
    expect(parseTokenCount('-5k')).toBeNull()
    // 'abc' / '' / '40g' / '40kk' rejection already covered in the
    // "rejects invalid input" case above; this case is specifically about
    // sign-rejection vs zero-snap so the contract split is explicit.
  })
```

(Negative-input coverage in the prior "rejects invalid input by returning null" case at line 35 `expect(parseTokenCount('-5')).toBeNull()` stays — it's now duplicated in the new case but that's intentional: makes the sign-rejection contract explicit alongside the zero-snap.)

- [ ] **Step 2: Run tests to verify the new contract fails**

Run from `/Users/yao/workspace/llm-perf/calc`:
```bash
npm test -- parseTokens 2>&1 | tail -6
```
Expected: FAIL — `parseTokenCount('0')` returns `null`, not `1`.

- [ ] **Step 3: Update the parser**

In `calc/src/ui/parseTokens.ts`, find the function (lines 3–15):
```ts
export function parseTokenCount(s: string): number | null {
  const m = s.trim().match(/^(\d+(?:\.\d+)?)\s*([kKmM]?)$/)
  if (!m) return null
  const n = parseFloat(m[1])
  const unit = m[2].toLowerCase()
  let v: number
  if (unit === 'k') v = n * 1024
  else if (unit === 'm') v = n * 1024 * 1024
  else v = n
  v = Math.round(v)
  if (!Number.isFinite(v) || v < 1) return null
  return v
}
```
Replace with:
```ts
export function parseTokenCount(s: string): number | null {
  const m = s.trim().match(/^(\d+(?:\.\d+)?)\s*([kKmM]?)$/)
  if (!m) return null   // unparseable (incl. negative — regex has no sign)
  const n = parseFloat(m[1])
  const unit = m[2].toLowerCase()
  let v: number
  if (unit === 'k') v = n * 1024
  else if (unit === 'm') v = n * 1024 * 1024
  else v = n
  if (!Number.isFinite(v)) return null
  // Clamp non-positive numeric input to 1 — never silently ignore the
  // user's input; the existing input handlers reflect the snapped value
  // back into the displayed text so the user sees it happen.
  return Math.max(1, Math.round(v))
}
```
(Two changes: split the `Number.isFinite` guard out, then `return Math.max(1, Math.round(v))` instead of the `v < 1 → null` rejection.)

- [ ] **Step 4: Run tests to verify the new contract passes**

Run: `npm test -- parseTokens 2>&1 | tail -6`
Expected: PASS — all `parseTokens` cases including the new `0 → 1` and `-5 → null` ones.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `npm test 2>&1 | grep -E "Tests " | tail -1`
Expected: all green (the old "rejects zero" assertion is gone; everything else unchanged).

- [ ] **Step 6: Commit (from repo root)**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/parseTokens.ts calc/test/ui/parseTokens.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "fix(calc): parseTokenCount snaps non-positive to 1; rejects negatives"
```
No `--no-verify`, no Co-Authored-By footer (project convention).

---

### Task 2: Concurrency input uses the parsed-input pattern; all three reflect the snap

**Files:**
- Modify: `calc/src/ui/InputPanel.svelte` (script: add `concurrencyInput`/`concurrencyInvalid`/`onConcurrencyInput`; update `onPromptInput`/`onOutputInput` to reflect snapped value back into the text; markup: replace the bare concurrency `<input type="number">`).

No unit test (presentational; verified in-browser per existing convention). The parser-level snap is already covered by Task 1.

- [ ] **Step 1: Add concurrency state + handler; reflect snap in the prompt/output handlers**

Current script block (around lines 52–75 of `calc/src/ui/InputPanel.svelte`):
```ts
  let promptInput = formatTokenCount($workload.promptTokens)
  let outputInput = formatTokenCount($workload.outputTokens)
  let promptInvalid = false
  let outputInvalid = false

  function onPromptInput(e: Event) {
    const v = (e.target as HTMLInputElement).value
    promptInput = v
    const n = parseTokenCount(v)
    if (n === null) { promptInvalid = true; return }
    promptInvalid = false
    workload.update(w => ({ ...w, promptTokens: n }))
  }

  function onOutputInput(e: Event) {
    const v = (e.target as HTMLInputElement).value
    outputInput = v
    const n = parseTokenCount(v)
    if (n === null) { outputInvalid = true; return }
    outputInvalid = false
    workload.update(w => ({ ...w, outputTokens: n }))
  }
```
Replace the entire block with:
```ts
  let promptInput = formatTokenCount($workload.promptTokens)
  let outputInput = formatTokenCount($workload.outputTokens)
  let concurrencyInput = String($workload.concurrency)
  let promptInvalid = false
  let outputInvalid = false
  let concurrencyInvalid = false

  function onPromptInput(e: Event) {
    const v = (e.target as HTMLInputElement).value
    promptInput = v
    const n = parseTokenCount(v)
    if (n === null) { promptInvalid = true; return }
    promptInvalid = false
    // Reflect snap (e.g. user typed "0" → parser returned 1) back into the
    // text so the change isn't silent.
    if (String(n) !== v.trim()) promptInput = String(n)
    workload.update(w => ({ ...w, promptTokens: n }))
  }

  function onOutputInput(e: Event) {
    const v = (e.target as HTMLInputElement).value
    outputInput = v
    const n = parseTokenCount(v)
    if (n === null) { outputInvalid = true; return }
    outputInvalid = false
    if (String(n) !== v.trim()) outputInput = String(n)
    workload.update(w => ({ ...w, outputTokens: n }))
  }

  function onConcurrencyInput(e: Event) {
    const v = (e.target as HTMLInputElement).value
    concurrencyInput = v
    const n = parseTokenCount(v)
    if (n === null) { concurrencyInvalid = true; return }
    concurrencyInvalid = false
    if (String(n) !== v.trim()) concurrencyInput = String(n)
    workload.update(w => ({ ...w, concurrency: n }))
  }
```
(Three additions: `concurrencyInput`/`concurrencyInvalid` state, an `onConcurrencyInput` handler mirroring the others, and a snap-reflection two-liner in each handler.)

- [ ] **Step 2: Update the Concurrency markup**

Find (line ~199–201 of `calc/src/ui/InputPanel.svelte`):
```svelte
      <label>
        Concurrency
        <input type="number" min="1" bind:value={$workload.concurrency} />
      </label>
```
Replace with (mirrors the Output Tokens block exactly — text input + handler + `.warn` badge):
```svelte
      <label>
        Concurrency
        <input
          type="text"
          inputmode="numeric"
          value={concurrencyInput}
          on:input={onConcurrencyInput}
          class:invalid={concurrencyInvalid}
          title="Positive integer (1 or more); 0 snaps to 1"
        />
        {#if concurrencyInvalid}
          <span class="warn">⚠ invalid — use a positive integer</span>
        {/if}
      </label>
```
The `.invalid` and `.warn` styles already exist in this file (used by prompt/output) — nothing new to add.

- [ ] **Step 3: Verify**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm run check 2>&1 | tail -1        # 0 errors
npm test 2>&1 | grep -E "Tests " | tail -1   # all green (no test added)
npm run build 2>&1 | tail -1        # clean
```

- [ ] **Step 4: Best-effort browser smoke (controller's job is the click-through)**

```bash
(npm run dev > /tmp/d.log 2>&1 &) ; sleep 4
P=$(grep -oE "localhost:[0-9]+" /tmp/d.log | head -1)
curl -s "http://$P/" -o /dev/null -w "HTTP %{http_code}\n"
pkill -f "node.*vite" 2>/dev/null
```
Expected: `HTTP 200`. (Interactive verification: typing `0` into any of the three inputs flips to `1`; typing `-5` shows the invalid badge and the prior value stays; valid `40k` parses unchanged.)

- [ ] **Step 5: Commit (from repo root)**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/InputPanel.svelte
git -C /Users/yao/workspace/llm-perf commit -m "fix(calc): concurrency input uses parsed-input pattern; reflect snap in display"
```

---

## Self-Review

**1. Spec coverage:**
- Spec "snap 0 to 1" for prompt/output → Task 1 (parser change covers both since both handlers call `parseTokenCount`). ✓
- Spec "snap 0 to 1" for concurrency → Task 2 (concurrency now also calls `parseTokenCount`). ✓
- Spec "reject negatives with invalid badge" for all three → Task 1's parser preserves the regex-non-match `null` for `-5`; all three handlers wire `null → *Invalid=true`. ✓
- Spec "reflect snapped value in displayed text" → Task 2 Step 1 adds the `if (String(n) !== v.trim()) ...Input = String(n)` line in each of the three handlers. ✓
- Spec "no change to share-URL encoding" → Plan touches neither share.ts nor stores.ts. ✓
- Spec "TDD at the parser, components verified in-browser" → Task 1 follows TDD; Task 2 is presentational + browser smoke. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases". Every step has full code or exact transform. Pass.

**3. Type consistency:** `parseTokenCount(s: string): number | null` — same signature, called identically in all three handlers; `workload.update(w => ({ ...w, <field>: n }))` pattern is identical across `promptTokens`/`outputTokens`/`concurrency` (the `Workload` interface already has all three as `number`). Consistent.

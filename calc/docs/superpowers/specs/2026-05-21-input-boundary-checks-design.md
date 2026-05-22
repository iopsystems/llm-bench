# Input boundary checks — Design

**Status:** Approved (brainstorm 2026-05-21)
**Scope:** Small bug fix. UI input sanitization only — no engine changes.

## Bug

When the user types `0` into the prompt-tokens or output-tokens box (or sets concurrency to `0` via the number spinner), the store quietly retains its previous value while the rest of the UI shows that value. The user perceives this as "their input was silently ignored and the default is being used." Negatives have the same class of risk on the concurrency `<input type="number">` (`bind:value` mirrors invalid values directly into the store).

## Behavior contract (after fix)

For **all three** numeric workload inputs — prompt tokens, output tokens, concurrency — uniformly:

| Input | After parse |
|---|---|
| `1`, `40k`, `1M`, valid positives | use as parsed |
| `0`, `0k` | **snap to `1`** (silent; display updates to `1` too so the user sees it) |
| `-5`, `abc`, malformed | **rejected**: show the existing `.warn` "invalid" badge; store keeps prior value |

## Implementation

Three files, ~30 LoC:

1. **`calc/src/ui/parseTokens.ts`** — change the post-parse guard from `if (!Number.isFinite(v) || v < 1) return null` to `return Math.max(1, Math.round(v))` after the regex match. Effect: `0` and `0k` now return `1` (snap). Negatives still return `null` because the existing regex `^(\d+(?:\.\d+)?)\s*([kKmM]?)$` doesn't accept the `-` sign — they fall out at the regex-non-match path and stay `null` (rejected → invalid badge, unchanged behavior).

2. **`calc/test/ui/parseTokens.test.ts`** — update the existing "rejects zero and negative results" case: split into two cases — "snaps zero to 1" (asserts `parseTokenCount('0') === 1`, `'0k' === 1024`? No — `0k = 0*1024 = 0`, post-snap `1`) and "rejects negatives" (asserts `parseTokenCount('-5') === null`). TDD-style: update first, see failures, change the parser, see green.

3. **`calc/src/ui/InputPanel.svelte`** — convert the **concurrency** `<input type="number" min="1" bind:value={$workload.concurrency}>` to the same pattern prompt/output already use: a local string `concurrencyInput`, a `concurrencyInvalid` flag, an `onConcurrencyInput` handler that calls `parseTokenCount(v)`, sets `concurrencyInvalid = true` on null (badge appears), or updates the store + reflects the (possibly snapped) value back into the input on success. KV reuse of `parseTokenCount` is fine — the parser accepts integers; users typing `1k` for concurrency get 1024 which is a coherent number. Add the `.warn` badge markup (mirrors the existing prompt/output blocks).
   Additionally, the prompt and output handlers already use `parseTokenCount`; their snap behavior follows from the parser change in (1). One small UX add: when the parser returns a snapped value that differs from what the user typed (e.g. `0` → `1`), reflect the snapped value in the displayed text so the user sees the snap happen rather than seeing their `0` stay in the box while the rest of the UI shows `1`. Implementation: after `const n = parseTokenCount(v)`, `if (n !== null && String(n) !== v.trim()) promptInput = String(n)` (and analogous for output, concurrency).

## Non-goals

- No change to engine math, roofline, or any store semantics beyond clamping.
- No change to KV/quant dtype handling.
- No change to share-URL encoding (the encoded value is already a positive integer per the existing decode guard `Number.isFinite(n) && n > 0`).

## Testing

- **TDD** at the parser: update + add `parseTokens.test.ts` cases (snap `0` to `1`; `0k` to `1`; reject `-5`, `abc` as before).
- **Component (InputPanel) change** verified in-browser per existing convention; no new component test.
- Full `npm test` + `npm run check` + `npm run build` must stay green.

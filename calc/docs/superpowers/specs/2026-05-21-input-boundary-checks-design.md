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
| `0`, `0k`, `-5`, `abc`, malformed | **rejected**: show the existing `.warn` "invalid" badge clarifying *positive integer required*; store keeps prior value |

Rationale for rejecting `0` (revised 2026-05-24): an earlier draft snapped `0` to `1` silently. In practice, watching `0` flip to `1` mid-typing was jarring — the user reads it as the UI fighting their input. Treating `0` the same as a malformed value (invalid badge, store untouched) is simpler and matches user mental model: *positive integers only.*

## Implementation

Two files, ~10 LoC:

1. **`calc/src/ui/parseTokens.ts`** — leave the existing `if (!Number.isFinite(v) || v < 1) return null` post-parse guard in place; `0`, `0k`, and negatives all return `null`. (No change required here once the original guard is restored.)

2. **`calc/test/ui/parseTokens.test.ts`** — keep a single "rejects zero and negative results (positive integer required)" case asserting `'0'`, `'0k'`, `'0.4'`, `'-5'`, `'-5k'` all return `null`. The existing "rejects invalid input" case still covers `''`, `'abc'`, `'40kk'`, `'40g'`.

3. **`calc/src/ui/InputPanel.svelte`** — convert the **concurrency** `<input type="number" min="1" bind:value={$workload.concurrency}>` to the same pattern prompt/output already use: a local string `concurrencyInput`, a `concurrencyInvalid` flag, an `onConcurrencyInput` handler that calls `parseTokenCount(v)`, sets `concurrencyInvalid = true` on null (badge appears), or updates the store on success. KV reuse of `parseTokenCount` is fine — the parser accepts integers; users typing `1k` for concurrency get 1024 which is a coherent number. Add the `.warn` badge markup (mirrors the existing prompt/output blocks). Update the prompt/output tooltips and warn messages to say *positive integer required* so the rejection of `0` reads correctly.

## Non-goals

- No change to engine math, roofline, or any store semantics beyond clamping.
- No change to KV/quant dtype handling.
- No change to share-URL encoding (the encoded value is already a positive integer per the existing decode guard `Number.isFinite(n) && n > 0`).

## Testing

- **TDD** at the parser: keep the single `parseTokens.test.ts` case rejecting `0`, `0k`, `0.4`, `-5`, `-5k` as null (positive-integer contract).
- **Component (InputPanel) change** verified in-browser per existing convention; no new component test.
- Full `npm test` + `npm run check` + `npm run build` must stay green.

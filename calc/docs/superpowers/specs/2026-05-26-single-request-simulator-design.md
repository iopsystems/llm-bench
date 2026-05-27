# Single-request simulator — Design

**Status:** Approved (brainstorm 2026-05-26)
**Scope:** New top-level tab. Time-domain view of one inference request — prefill + autoregressive decode, no concurrency, no batching, no contention. First of an anticipated family of "simulator" tabs.

## Goal

Give users a clear, visual answer to *"what does a single inference request actually look like on this hardware?"* The existing calc tab answers throughput/sizing questions; the simulator tab answers latency-experience questions.

## Behavior contract

For a given (accelerator/system, model, quant, prompt+output tokens, optional parallelism, optional disagg fabric), the simulator displays:

| Indicator | Formula | Source |
|---|---|---|
| **TTFT** (time to first token) | `perf[opPoint].ttftS` | engine, unchanged |
| **TPOT** (constant) | `perf[opPoint].decode.timePerTokenS` | engine, unchanged |
| **Total latency** | `TTFT + TPOT × (outputTokens − 1)` ¹ | computed in the sim view |
| **Sustained throughput** | `1 / TPOT` tokens/sec | computed in the sim view |
| **Prefill regime** | compute / memory / comms | `perf[opPoint].prefill.regime` |
| **Decode regime** | compute / memory / comms | `perf[opPoint].decode.regime` |

`concurrency` is clamped to `1` in the `CalcInput` the simulator constructs, regardless of the shared store value. The simulator never writes to `workload.concurrency`.

¹ **Total-latency formula caveat.** The engine's `ttftS` follows the production-runtime convention that prefill's last-position logits yield the first user-visible token (no separate decode step), so `Total = TTFT + TPOT × (outputTokens − 1)` is exact for the non-disagg and `firstTokenOnPrefill=true` cases. For disagg with `firstTokenOnPrefill=false`, the engine reports `ttftS = prefill + kvTransferS` without including a decode step (because no token has been emitted yet at that point), so the simple formula undercounts Total by ≈ one TPOT in that case. The error is bounded by one TPOT (single-digit to low-tens of ms typically) and below the calc's overall accuracy band — acceptable for v1. If it ever matters, the fix is a per-case formula in the sim view.

## Non-goals (v1)

- **No per-step decode modeling.** TPOT is constant. The reasoning-output latency differential (where TPOT grows with KV) is deliberately pushed to the TODO stack — for typical chat workloads the difference is <1%, and the closed-form per-step model can be added later without changing the sim's UI shape.
- No speculative decoding, chunked prefill, draft model — those are future simulator tabs.
- No comparison mode (sim-vs-sim, sim-vs-trace).
- No deep-link to a specific operating point (op-point follows whatever the calc tab is showing, consistent with current behavior).
- No engine math changes. Sim is a pure consumer of `calculate()`.

## UI

**Tab:** new "Simulator" entry in `TabBar`, sibling of Calc and Info.

**Layout (single column, top-to-bottom):**

1. **Input panel** — reuse `InputPanel.svelte` with a new boolean prop `hideConcurrency`. When true, the Concurrency `<input>` is omitted. All other inputs (hardware/variant/system, model, quant, prompt tokens, output tokens, parallelism picker, disagg fabric + first-token-on-prefill toggle) remain visible and edit the same shared stores the calc tab uses.

2. **KPI strip** — three cards side-by-side: TTFT, TPOT, Total latency. Each card carries a regime badge. The TPOT card additionally shows `1/TPOT tokens/sec` as a small caption.

3. **Visual gantt** — new SVG component `SimulatorGantt.svelte`. Segments follow the engine's TTFT decomposition exactly (see [calc.ts:60–73](../../../src/engine/calc.ts#L60-L73)):
   - **Non-disagg:** two segments. `[0 → prefill.timeS]` prefill (first token sampled from prefill's last-position logits — no separate decode step), then `[prefill.timeS → Total]` decode for the remaining `outputTokens − 1` tokens. Marker at `x = ttftS = prefill.timeS`, drawn at the prefill/decode boundary.
   - **Disagg with `firstTokenOnPrefill=true` (overlapped layout):** prefill cluster runs prefill, then performs one decode step (emitting the first user-visible token); KV-transfer happens in parallel with that decode step on the prefill cluster. The decode cluster picks up the remaining `outputTokens − 1` tokens starting at `prefill.timeS + kvTransferS`. Marker at `x = ttftS = prefill.timeS + decode.timePerTokenS`. Three visual elements: prefill bar, KV-transfer bar overlaid below the post-prefill region, decode bar on the decode cluster.
   - **Disagg with `firstTokenOnPrefill=false` (sequential handoff):** three segments. Prefill → full KV-transfer → decode on the decode cluster. Marker at `x = ttftS = prefill.timeS + kvTransferS` (at the KV-xfer/decode boundary; the engine does *not* add a first decode step in this case — first decode is the start of the decode segment).
   - Time axis below with labeled ticks at `0`, `TTFT`, `Total`. Color per segment by regime (compute / memory / comms). Legend below the axis.

The gantt never computes TTFT itself — it consumes `ttftS` from the engine and places the marker at exactly that x-coordinate. If the engine's formula evolves, the gantt follows automatically.

**No derivation drawer or breakdown table on the sim tab for v1.** Math is identical to calc; users who want the breakdown flip tabs.

## State and URL

**State:** shared with calc via the existing stores. No new writables.

**Operating-point selection:** sim follows the same op-point the calc tab is showing (op-point is not URL state today, consistent).

**URL encoding:** new hash prefix `#sim?<encoded>`. The payload is identical to `#calc?<encoded>` — same shared state, same param roster (`a/v/s/m/w/kv/ac/pt/ot/c/p/dk/df`). The difference is which tab the recipient lands on.

**Hash helpers:**
- `calcPayloadFromHash(hash)` → `tabPayloadFromHash(hash, tab: 'calc' | 'sim')`. Returns the payload after the `?` for the requested tab's prefix; empty string otherwise.
- `startUrlSync` writes to `#calc?...` or `#sim?...` depending on the active tab. Continues to skip writes when on the info tab.
- `readUrlIntoStores` accepts either prefix on load.

`Route` gains `{ tab: 'sim' }`. `parseRoute` recognizes `#sim` and `#sim?...`. `App.svelte` adds a third branch for `tab === 'sim'`.

## Files

| File | Action | Purpose |
|---|---|---|
| `calc/src/ui/Simulator.svelte` | create | Owns the sim layout: input panel + KPI strip + gantt + op-point selector. Builds `CalcInput` with `concurrency=1`, calls `calculate()`, renders. |
| `calc/src/ui/SimulatorGantt.svelte` | create | Pure SVG component. Takes `{prefillS, kvTransferS?, decodeS, ttftS, prefillRegime, decodeRegime, firstTokenOnPrefill?}` and draws the gantt. No store deps — easy to unit-test. |
| `calc/src/ui/route.ts` | modify | Add `{ tab: 'sim' }` to `Route` union; teach `parseRoute` and `serializeRoute` the `#sim` prefix. |
| `calc/src/ui/TabBar.svelte` | modify | Add the Simulator tab entry. |
| `calc/src/ui/App.svelte` | modify | Third conditional branch for `$route.tab === 'sim'`. |
| `calc/src/ui/InputPanel.svelte` | modify | New `hideConcurrency` prop (default `false`). Omits the Concurrency label/input when true. |
| `calc/src/ui/share.ts` | modify | Generalize `calcPayloadFromHash` → `tabPayloadFromHash(hash, tab)`. `startUrlSync` writes to the prefix matching the active tab. `readUrlIntoStores` accepts either prefix. |

## Testing

- **Engine math:** no new tests — `calculate()` and its existing test suite are unchanged.
- **`SimulatorGantt` (new):** unit tests for SVG output shape. Three cases: non-disagg (2 segments), disagg with `firstTokenOnPrefill=false` (3 segments), disagg with `firstTokenOnPrefill=true` (overlapped). Assert segment x/width and marker x against hand-computed values.
- **`route.ts`:** `parseRoute('#sim')` and `parseRoute('#sim?a=h100')` return `{ tab: 'sim' }`. Round-trip via `serializeRoute`.
- **`share.ts`:** `tabPayloadFromHash` extracts payload from both prefixes; rejects the wrong one. `startUrlSync` writes the right prefix when the active tab flips (verified by stubbing `window.location.hash` and inspecting the write).
- **`InputPanel.svelte`:** existing tests untouched. The `hideConcurrency` prop is presentational; verified in-browser per existing convention.
- **In-browser smoke (controller's job):** dev-server HTTP 200; flip between tabs, edit model on sim → calc tab reflects the change; toggle disagg → gantt gains a third segment; copy a sim URL into a new tab → restores on the sim tab.
- Full `npm test` + `npm run check` + `npm run build` must stay green.

## Implementation order (sketch — the actual ordering is the plan's job)

1. Generalize the hash/route helpers (route.ts + share.ts) and add the `sim` tab variant. Tests first.
2. `InputPanel.svelte` `hideConcurrency` prop.
3. `SimulatorGantt.svelte` with tests.
4. `Simulator.svelte` composing the above. In-browser smoke.
5. `TabBar` + `App.svelte` wiring.

## Rationale

- **Constant TPOT, not per-step:** for typical chat workloads the first-vs-last decode-token difference is <1% (numbers checked during brainstorm: Llama-3.3-70B GQA, H100 bf16; even at 128k prompt + 4k output the spread is <1%). The effect is real only for long-output reasoning workloads (~10–40% spread for 32k+ outputs). The closed-form per-step model can be added later without changing the sim's UI shape — when reasoning workloads become a focus.
- **Shared stores, not sim-local:** users edit model/hw on one tab and expect the other to follow. Forcing them to re-pick across tabs is friction. Concurrency is the one exception (meaningless for sim); hiding the input on the sim tab and clamping to 1 in the math is cleaner than maintaining a sim-specific store.
- **Own URL prefix:** keeps sim sharing self-describing (recipient lands on sim tab). Same encoder so the payload is interchangeable — useful if we later want a "view this config on the calc tab" link from sim and vice versa.
- **Visual gantt, not just numbers:** the user explicitly picked the gantt option as the framing for "first simulator." The gantt is what makes this a *simulator* rather than a re-styled calc.

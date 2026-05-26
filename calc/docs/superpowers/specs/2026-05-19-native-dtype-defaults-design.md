# Native-dtype model-aware quant defaults — Design

**Status:** Approved (brainstorm 2026-05-19)
**Scope:** Roadmap backlog item "native dtypes on model spec." Schema + data + UI/state. No engine math changes.

## Goal

Replace the hardcoded global `fp16/fp16/fp16` quant default with model-aware defaults: weights+activations default to the precision the model actually ships in (`bf16` for most, `fp8` for the DeepSeek family). A user-visible "Lock dtype" toggle pins the precision across model switches.

## Non-goals

- No engine/roofline math changes.
- KV-cache dtype is **not** driven by `nativeDtype` — it stays an independent serving choice (current `fp16` default unchanged).
- No new attention/architecture modeling. No model additions.

## 1. Schema + data

- Add required field `nativeDtype: Dtype` to `ModelArch` in `calc/src/engine/types.ts` (the precision the released weights ship in).
- Backfill **all** existing model entries in `calc/src/data/models.ts`, sourced:
  - `fp8` for DeepSeek native-fp8 releases: `deepseek-v3`, `deepseek-r1`, `deepseek-v3.2`, `deepseek-v4-flash`, `deepseek-v4-pro`.
  - `bf16` for every other model (the safe reference — never silently misrepresents quality; `bf16` is the modern datacenter reference, not `fp16`).
- `nativeDtype` is a **required** field, so every inline `ModelArch` constructed in tests must be updated or `svelte-check` fails. Known sites (from the prior `publisher`/`releaseDate` retrofit): `calc/test/fixtures.ts` (`testModel`), `calc/test/engine/sliding.test.ts` (multiple `base`/`id: 't'` literals), `calc/test/engine/parallelism.test.ts` (`dense`). Add `nativeDtype: 'bf16'` to each. The implementation plan must include a fixture-sweep task.
- `nativeDtype` is catalog metadata, **not** a HuggingFace `config.json` field. Therefore:
  - Add `'nativeDtype'` to `FIELDS_NOT_IN_TABLE` in `.claude/hooks/check-skill-sync.mjs` **and** the mirrored set in `calc/test/check-skill-sync.test.ts`.
  - Document it in `.claude/skills/adding-a-model/SKILL.md` alongside the `publisher`/`releaseDate` metadata note (where to source it: the model card / release notes; default `bf16` when unstated).
  - The pre-commit + Claude skill-sync hooks must stay green.

## 2. Behavior (re-seed + explicit lock)

- New store in `calc/src/ui/stores.ts`: `export const lockDtype = writable(false)`.
- Reactive rule (in `stores.ts`): when `modelId` changes and `lockDtype` is `false`, set `quant.weights` and `quant.activations` to the selected model's `nativeDtype`. `quant.kv` is left untouched.
- When `lockDtype` is `true`, model changes do not modify `quant`.
- Manually changing the weights/activations dropdowns does **not** auto-lock. Locking is explicit via the checkbox. (Accepted trade-off: an unlocked manual tweak is overwritten on the next model switch; the visible toggle makes this predictable — chosen over a hidden dirty-flag.)
- Implementation note: the re-seed must observe `modelId` changes without an infinite loop. Use a guarded subscription/`derived` that compares the previous model id, not a blanket `quant`↔`model` reactive cycle.

## 3. URL / share interaction

`calc/src/ui/share.ts` (`ShareableState`, `encodeState`, `decodeState`, `applyToStores`):

- Add `lockDtype` to `ShareableState`; encode as hash key `ld` (`ld=1` when true; omitted when false).
- On load (`readUrlIntoStores`, calc route only):
  - If the URL carries explicit quant (`w`/`kv`/`ac`): apply it **and force `lockDtype = true`** so a shared precision is never clobbered by the initial model's re-seed. (Legacy links with quant but no `ld` → treated as locked: preserves the sharer's intent.)
  - If `ld` is present, it sets `lockDtype` (and an explicit `ld=0` with quant is honored as the sharer's choice).
  - If no URL quant: `lockDtype = false` and `quant.weights/activations = selected model's nativeDtype` (model resolved from URL `m` or the default).
- Re-seed-on-load ordering: apply URL state first, then if unlocked and no URL quant, seed from the resolved model's `nativeDtype`.

## 4. UI

- `calc/src/ui/InputPanel.svelte`: a **"Lock dtype"** checkbox in the quant controls area, bound to `lockDtype`. Tooltip: "Keep this precision when switching models; otherwise weights/activations follow each model's native dtype."
- In the **weights** and **activations** `<select>`s: the `<option>` whose value equals the selected model's `nativeDtype` gets `class="native"` (CSS `font-weight: bold`, best-effort — Firefox honors it) **and** a ` — native` text suffix on the option label (the reliable cross-browser cue, since Chrome/Safari ignore per-option font styling).
- The **weights** `<select>` additionally **disables** options that aren't the native and whose bit-width is `>=` the native's (no upcast, no same-width sideways — post-release quant is downward-only). Activations and KV dropdowns are NOT constrained; activations may differ from weights (weight-only quant is a common deployment pattern), and KV is an independent serving axis.
- `calc/src/ui/ModelSpecSheet.svelte`: add a "Native precision" row to the **Design** section, rendering `model.nativeDtype`.

## 5. Testing (TDD)

- `calc/test/data/models-native-dtype.test.ts`: every `MODELS` entry has `nativeDtype` ∈ the `Dtype` union; the five DeepSeek entries are `fp8`; spot-check ≥3 representative models are `bf16`.
- `calc/test/ui/native-dtype-defaults.test.ts` (store behavior, no DOM): selecting a model while unlocked sets `quant.weights/activations` to its `nativeDtype` and leaves `quant.kv`; locked → `quant` unchanged on model switch; the unlock→switch→reseed path.
- `calc/test/ui/share-route.test.ts` / share tests: `encodeState`/`decodeState` round-trip `lockDtype`; URL with quant but no `ld` decodes to `lockDtype = true`.
- skill-sync: extend the existing `check-skill-sync` test to assert `nativeDtype` is excluded; hook run stays green.
- Components (InputPanel checkbox, option styling, spec-sheet row) verified in-browser per existing convention (no unit test).

## Architecture rationale

Single `nativeDtype: Dtype` (not separate weights/activations, not a set) — models ship at one precision; defaults need one value; YAGNI. Explicit "Lock dtype" toggle over a hidden dirty-flag — the behavior is visible and shareable. Re-seed lives in `stores.ts` as a guarded `modelId` subscription so the model→quant coupling is in one place and testable without the DOM. KV deliberately excluded from the coupling: weight-only / KV-cache quant is an independent axis (per the weights↔activations decoupling discussion).

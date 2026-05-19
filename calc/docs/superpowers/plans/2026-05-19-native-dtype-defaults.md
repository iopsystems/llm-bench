# Native-dtype Model-Aware Quant Defaults — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Default weights+activations quant to each model's native shipped precision (`bf16` most, `fp8` DeepSeek), with an explicit "Lock dtype" toggle to pin precision across model switches.

**Architecture:** New required `ModelArch.nativeDtype: Dtype` (catalog metadata, skill-sync-excluded). A `lockDtype` store + a guarded `modelId` subscription re-seeds `quant.weights/activations` (never KV) on model change unless locked. `lockDtype` joins shareable URL state; a shared explicit quant forces lock so links don't get clobbered.

**Tech Stack:** Svelte 5, TypeScript, Vitest. npm from `calc/`; git from repo root `/Users/yao/workspace/llm-perf`. Branch `feat/native-dtype-defaults` (spec already committed there — do NOT create/switch branches).

**Spec:** `calc/docs/superpowers/specs/2026-05-19-native-dtype-defaults-design.md`

---

### Task 1: Schema field + data backfill + fixtures + skill-sync

`nativeDtype` is **required**, so the field, the full data backfill, every inline test fixture, and the skill-sync exclusion must land together or `svelte-check` fails. Atomic task.

**Files:**
- Modify: `calc/src/engine/types.ts` (`ModelArch`)
- Modify: `calc/src/data/models.ts` (all entries)
- Modify: `calc/test/fixtures.ts`, `calc/test/engine/sliding.test.ts`, `calc/test/engine/parallelism.test.ts`
- Modify: `.claude/hooks/check-skill-sync.mjs`, `calc/test/check-skill-sync.test.ts`, `.claude/skills/adding-a-model/SKILL.md`
- Create: `calc/test/data/models-native-dtype.test.ts`

- [ ] **Step 1: Write the failing data test**

Create `calc/test/data/models-native-dtype.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { MODELS } from '../../src/data'

const DTYPES = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4']
const DEEPSEEK_FP8 = ['deepseek-v3', 'deepseek-r1', 'deepseek-v3.2', 'deepseek-v4-flash', 'deepseek-v4-pro']

describe('nativeDtype', () => {
  it('every model has a valid nativeDtype', () => {
    for (const m of MODELS) {
      expect(DTYPES, `${m.id}`).toContain(m.nativeDtype)
    }
  })
  it('DeepSeek native-fp8 releases are fp8', () => {
    for (const id of DEEPSEEK_FP8) {
      expect(MODELS.find(m => m.id === id)!.nativeDtype, id).toBe('fp8')
    }
  })
  it('representative models are bf16', () => {
    for (const id of ['llama-3.3-70b', 'qwen3-8b', 'phi-4', 'mistral-small-3.2-24b']) {
      expect(MODELS.find(m => m.id === id)!.nativeDtype, id).toBe('bf16')
    }
  })
})
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `cd /Users/yao/workspace/llm-perf/calc && npm test -- models-native-dtype`
Expected: FAIL (`nativeDtype` undefined / not a valid dtype).

- [ ] **Step 3: Add the schema field**

In `calc/src/engine/types.ts`, in `interface ModelArch`, immediately after the `releaseDate` field (the line `  releaseDate: string`) add:

```ts
  // Precision the released weights ship in (bf16 for most; fp8 for the
  // DeepSeek native-fp8 family). Drives the model-aware weights/activations
  // quant default. Catalog metadata, NOT from HuggingFace config.json.
  nativeDtype: Dtype
```

- [ ] **Step 4: Backfill all model entries**

Create `/tmp/backfill-native-dtype.mjs`:

```js
import { readFileSync, writeFileSync } from 'node:fs'
const p = '/Users/yao/workspace/llm-perf/calc/src/data/models.ts'
let s = readFileSync(p, 'utf8')
// Insert `    nativeDtype: 'bf16',` on the line after every `releaseDate:` line.
s = s.replace(/^(\s*)(publisher: '[^']*', releaseDate: '[^']*',)$/gm,
  `$1$2\n$1nativeDtype: 'bf16',`)
// Promote the DeepSeek native-fp8 releases.
for (const id of ['deepseek-v3','deepseek-r1','deepseek-v3.2','deepseek-v4-flash','deepseek-v4-pro']) {
  const re = new RegExp(`(id: '${id.replace(/[.]/g,'\\.')}',[\\s\\S]*?nativeDtype: ')bf16(',)`)
  s = s.replace(re, `$1fp8$2`)
}
writeFileSync(p, s)
console.log('done')
```

Run: `node /tmp/backfill-native-dtype.mjs`
Then verify: `grep -c "nativeDtype:" calc/src/data/models.ts` should equal the model count, and `grep -n "nativeDtype: 'fp8'" calc/src/data/models.ts` shows exactly the 5 DeepSeek entries. If the `publisher/releaseDate` line format differs for any entry (a `MISS`), fix that entry by hand — every entry must have `nativeDtype`.

- [ ] **Step 5: Fix inline ModelArch test fixtures**

`calc/test/fixtures.ts` — in `testModel`, after the `releaseDate: '2025-01',` line add `  nativeDtype: 'bf16',`.

`calc/test/engine/sliding.test.ts` and `calc/test/engine/parallelism.test.ts` — every inline model literal needs the field. Run:

```bash
cd /Users/yao/workspace/llm-perf/calc
perl -0pi -e "s/(releaseDate: '2025-01',)/\$1 nativeDtype: 'bf16',/g" test/engine/sliding.test.ts test/engine/parallelism.test.ts
grep -rn "publisher: '" test/engine/sliding.test.ts test/engine/parallelism.test.ts | grep -v nativeDtype || echo "all inline literals patched"
```

If any inline literal uses a different `releaseDate` value, add `nativeDtype: 'bf16',` to it by hand. The `grep ... || echo` line must print the "all patched" message.

- [ ] **Step 6: Exclude nativeDtype from skill-sync**

`.claude/hooks/check-skill-sync.mjs` — find the `FIELDS_NOT_IN_TABLE` set (currently contains `'publisher', 'releaseDate'`) and add `'nativeDtype'` to it.

`calc/test/check-skill-sync.test.ts` — the mirrored `FIELDS_NOT_IN_TABLE` set: add `'nativeDtype'` identically.

`.claude/skills/adding-a-model/SKILL.md` — find the metadata note that documents `publisher` and `releaseDate` (the bullet list under "`publisher` and `releaseDate` are catalog metadata"). Add a third bullet:

```markdown
- `nativeDtype` — the dtype the released weights ship in (the model-aware quant default). `bf16` for most; `fp8` for natively-fp8 releases (DeepSeek V3/R1/V3.2/V4). Source from the model card / release notes; default `bf16` when unstated. Not a `config.json` field.
```

- [ ] **Step 7: Run data test + full suite + check + skill-sync**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test -- models-native-dtype 2>&1 | tail -3   # PASS
npm test 2>&1 | grep -E "Tests " | tail -1        # all green
npm run check 2>&1 | tail -1                       # 0 errors
node /Users/yao/workspace/llm-perf/.claude/hooks/check-skill-sync.mjs   # ✓ in sync
```

All four must pass. If `npm run check` reports a missing `nativeDtype` in some inline literal, fix that literal and re-run.

- [ ] **Step 8: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/engine/types.ts calc/src/data/models.ts calc/test/fixtures.ts calc/test/engine/sliding.test.ts calc/test/engine/parallelism.test.ts .claude/hooks/check-skill-sync.mjs calc/test/check-skill-sync.test.ts .claude/skills/adding-a-model/SKILL.md calc/test/data/models-native-dtype.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): ModelArch.nativeDtype + backfill (bf16; fp8 for DeepSeek)"
```
No `--no-verify`, no Co-Authored-By footer (project convention).

---

### Task 2: lockDtype store + model-change re-seed

**Files:**
- Modify: `calc/src/ui/stores.ts`
- Modify: `calc/src/main.ts`
- Test: `calc/test/ui/native-dtype-defaults.test.ts`

- [ ] **Step 1: Write the failing test**

Create `calc/test/ui/native-dtype-defaults.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { modelId, quant, lockDtype, initNativeDtypeSync } from '../../src/ui/stores'
import { MODELS } from '../../src/data'

const fp8Model = MODELS.find(m => m.nativeDtype === 'fp8')!.id   // a DeepSeek
const bf16Model = MODELS.find(m => m.nativeDtype === 'bf16')!.id

describe('native-dtype re-seed', () => {
  let stop: () => void
  beforeEach(() => {
    lockDtype.set(false)
    quant.set({ weights: 'fp16', kv: 'fp16', activations: 'fp16' })
    modelId.set(bf16Model)
    stop?.()
    stop = initNativeDtypeSync()  // skips the current value; reseeds on change
  })

  it('unlocked: switching model reseeds weights+activations, not kv', () => {
    modelId.set(fp8Model)
    expect(get(quant)).toEqual({ weights: 'fp8', kv: 'fp16', activations: 'fp8' })
  })

  it('locked: switching model leaves quant untouched', () => {
    lockDtype.set(true)
    quant.set({ weights: 'fp4', kv: 'int8', activations: 'fp4' })
    modelId.set(fp8Model)
    expect(get(quant)).toEqual({ weights: 'fp4', kv: 'int8', activations: 'fp4' })
  })

  it('unlock then switch reseeds again', () => {
    lockDtype.set(true)
    modelId.set(fp8Model)            // locked: no change
    lockDtype.set(false)
    modelId.set(bf16Model)           // unlocked: reseed
    expect(get(quant).weights).toBe('bf16')
    expect(get(quant).activations).toBe('bf16')
  })
})
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `npm test -- native-dtype-defaults`
Expected: FAIL (`lockDtype` / `initNativeDtypeSync` not exported).

- [ ] **Step 3: Add store + sync**

In `calc/src/ui/stores.ts`, immediately after the `quant` store declaration (the block ending `})` on the line after `weights: 'fp16', kv: 'fp16', activations: 'fp16'`) add:

```ts

// When false (default), switching models reseeds weights+activations to the
// new model's nativeDtype. When true, the user has pinned the precision and
// model switches leave quant alone. Part of shareable URL state (see share.ts).
export const lockDtype = writable(false)

// Wire the model→quant coupling. Call once at startup AFTER readUrlIntoStores()
// so a shared URL's quant/model is in place first. Skips the current modelId
// value (no reseed on load); reseeds on subsequent changes only when unlocked.
// KV is never touched — it's an independent serving axis.
export function initNativeDtypeSync(): () => void {
  let first = true
  return modelId.subscribe($modelId => {
    if (first) { first = false; return }
    if (get(lockDtype)) return
    const m = MODELS.find(x => x.id === $modelId)
    if (!m) return
    quant.update(q => ({ ...q, weights: m.nativeDtype, activations: m.nativeDtype }))
  })
}
```

`get` is imported in `share.ts` but NOT in `stores.ts` — add `get` to the existing svelte/store import. Change line 1 of `stores.ts` from:

```ts
import { writable, derived, type Readable } from 'svelte/store'
```
to:
```ts
import { writable, derived, get, type Readable } from 'svelte/store'
```

- [ ] **Step 4: Call it from main.ts**

`calc/src/main.ts` currently is:

```ts
import { mount } from 'svelte'
import App from './ui/App.svelte'
import { readUrlIntoStores, startUrlSync } from './ui/share'
import { initRouteSync } from './ui/route'

readUrlIntoStores()
const app = mount(App, { target: document.getElementById('app')! })
startUrlSync()
initRouteSync()
export default app
```

Change to (add the import and the call after `readUrlIntoStores()`, before `mount`):

```ts
import { mount } from 'svelte'
import App from './ui/App.svelte'
import { readUrlIntoStores, startUrlSync } from './ui/share'
import { initRouteSync } from './ui/route'
import { initNativeDtypeSync } from './ui/stores'

readUrlIntoStores()
initNativeDtypeSync()
const app = mount(App, { target: document.getElementById('app')! })
startUrlSync()
initRouteSync()
export default app
```

- [ ] **Step 5: Run test + full suite + check**

```bash
npm test -- native-dtype-defaults 2>&1 | tail -3   # PASS
npm test 2>&1 | grep -E "Tests " | tail -1          # all green
npm run check 2>&1 | tail -1                         # 0 errors
```

- [ ] **Step 6: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/stores.ts calc/src/main.ts calc/test/ui/native-dtype-defaults.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): lockDtype store + model-change quant re-seed"
```

---

### Task 3: lockDtype in shareable URL state

**Files:**
- Modify: `calc/src/ui/share.ts`
- Test: `calc/test/ui/share-route.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `calc/test/ui/share-route.test.ts` (new `describe` at end of file):

```ts
import { encodeState, decodeState } from '../../src/ui/share'

describe('lockDtype share state', () => {
  const base = {
    acceleratorId: 'h100', variantId: 'sxm-80', systemId: '', modelId: 'llama-3.3-70b',
    quant: { weights: 'bf16', kv: 'fp16', activations: 'bf16' } as const,
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 },
    parallelismOverride: null, disaggKvTransferFabricId: '', disaggFirstTokenOnPrefill: true,
  }
  it('round-trips lockDtype=true via ld=1', () => {
    const enc = encodeState({ ...base, lockDtype: true })
    expect(enc).toContain('ld=1')
    expect(decodeState(enc).lockDtype).toBe(true)
  })
  it('omits ld when false', () => {
    expect(encodeState({ ...base, lockDtype: false })).not.toContain('ld=')
  })
  it('quant present but no ld → lockDtype true (preserve sharer intent)', () => {
    expect(decodeState('m=llama-3.3-70b&w=fp8&kv=fp16&ac=fp8').lockDtype).toBe(true)
  })
  it('explicit ld=0 with quant is honored', () => {
    expect(decodeState('m=llama-3.3-70b&w=fp8&kv=fp16&ac=fp8&ld=0').lockDtype).toBe(false)
  })
  it('no quant, no ld → lockDtype undefined (caller defaults false)', () => {
    expect(decodeState('m=llama-3.3-70b').lockDtype).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `npm test -- share-route`
Expected: FAIL (`lockDtype` not in `ShareableState`; not encoded/decoded).

- [ ] **Step 3: Extend ShareableState + encode**

In `calc/src/ui/share.ts`:

(a) Add `get` is already imported. Import `lockDtype`: the existing store import block (lines ~14-18) imports `quant, workload`. Add `lockDtype`:

Change:
```ts
import {
  acceleratorId, variantId, systemId, modelId,
  parallelismOverride, disaggKvTransferFabricId, disaggFirstTokenOnPrefill,
  quant, workload
} from './stores'
```
to:
```ts
import {
  acceleratorId, variantId, systemId, modelId,
  parallelismOverride, disaggKvTransferFabricId, disaggFirstTokenOnPrefill,
  quant, workload, lockDtype
} from './stores'
```

(b) In `interface ShareableState`, after `disaggFirstTokenOnPrefill: boolean` add:
```ts
  lockDtype: boolean
```

(c) In `encodeState`, immediately before `return p.toString()` add:
```ts
  if (state.lockDtype) p.set('ld', '1')
```

- [ ] **Step 4: Decode lockDtype (with the quant-implies-lock rule)**

In `decodeState`, the quant block currently reads:

```ts
  const w = params.get('w')
  const kv = params.get('kv')
  const ac = params.get('ac')
  if (w && kv && ac && isDtype(w) && isDtype(kv) && isDtype(ac)) {
    out.quant = { weights: w, kv, activations: ac }
  }
```

Immediately AFTER that block, add:

```ts
  // lockDtype: explicit `ld` wins; otherwise an explicit quant in the URL
  // implies the sharer pinned a precision, so lock to avoid reseeding it.
  const ld = params.get('ld')
  if (ld !== null) out.lockDtype = ld !== '0'
  else if (out.quant !== undefined) out.lockDtype = true
```

- [ ] **Step 5: readStoreState + applyToStores**

In `readStoreState` (the `return { ... }` of store getters), after `disaggFirstTokenOnPrefill: get(disaggFirstTokenOnPrefill),` add:
```ts
    lockDtype: get(lockDtype),
```

In `applyToStores`, after `if (partial.disaggFirstTokenOnPrefill !== undefined) disaggFirstTokenOnPrefill.set(partial.disaggFirstTokenOnPrefill)` add:
```ts
  if (partial.lockDtype !== undefined) lockDtype.set(partial.lockDtype)
```

- [ ] **Step 6: Mirror lockDtype to URL on change**

In `startUrlSync`, the `unsubs` array of `.subscribe(write)` calls — add a line alongside the others (e.g. after `workload.subscribe(write),`):
```ts
    lockDtype.subscribe(write),
```
(Find the array containing `quant.subscribe(write),` and `workload.subscribe(write),` and add the `lockDtype` line inside it.)

- [ ] **Step 7: Run test + full suite + check**

```bash
npm test -- share-route 2>&1 | tail -3   # PASS
npm test 2>&1 | grep -E "Tests " | tail -1
npm run check 2>&1 | tail -1
```

- [ ] **Step 8: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/share.ts calc/test/ui/share-route.test.ts
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): lockDtype in shareable URL; quant-in-URL implies lock"
```

---

### Task 4: InputPanel — Lock checkbox + native option cue

**Files:**
- Modify: `calc/src/ui/InputPanel.svelte`

(Presentational — verified in-browser per existing convention, no unit test.)

- [ ] **Step 1: Import lockDtype + resolve selected model's nativeDtype**

In the `<script>` of `calc/src/ui/InputPanel.svelte`, the stores import line is:
```ts
  import { acceleratorId, variantId, systemId, modelId, quant, workload, disaggKvTransferFabricId, disaggFirstTokenOnPrefill } from './stores'
```
Add `lockDtype`:
```ts
  import { acceleratorId, variantId, systemId, modelId, quant, workload, disaggKvTransferFabricId, disaggFirstTokenOnPrefill, lockDtype } from './stores'
```

There is already `$: selectedModel = MODELS.find(m => m.id === $modelId)` in this file (used for the context-window warning). Reuse it; if not present in scope of the quant block it still is module-level reactive, so `selectedModel?.nativeDtype` is available.

- [ ] **Step 2: Native cue on the Weights + Activations options**

Current Weights block:
```svelte
      <label>
        Weights
        <select bind:value={$quant.weights}>
          {#each DTYPES as d}<option value={d}>{d}</option>{/each}
        </select>
      </label>
```
Replace with:
```svelte
      <label>
        Weights
        <select bind:value={$quant.weights}>
          {#each DTYPES as d}<option value={d} class:native={d === selectedModel?.nativeDtype}>{d}{d === selectedModel?.nativeDtype ? ' — native' : ''}</option>{/each}
        </select>
      </label>
```
Current Activations block:
```svelte
      <label>
        Activations
        <select bind:value={$quant.activations}>
          {#each DTYPES as d}<option value={d}>{d}</option>{/each}
        </select>
      </label>
```
Replace with:
```svelte
      <label>
        Activations
        <select bind:value={$quant.activations}>
          {#each DTYPES as d}<option value={d} class:native={d === selectedModel?.nativeDtype}>{d}{d === selectedModel?.nativeDtype ? ' — native' : ''}</option>{/each}
        </select>
      </label>
```
(KV select is unchanged — native cue is weights/activations only.)

- [ ] **Step 3: Lock-dtype checkbox**

Immediately after the Activations `</label>` (and before the `</div>` that closes the quant `.row`), add:
```svelte
      <label class="lockdtype" title="Keep this precision when switching models; otherwise weights/activations follow each model's native dtype">
        <input type="checkbox" bind:checked={$lockDtype} />
        Lock dtype
      </label>
```

- [ ] **Step 4: Styles**

In the `<style>` block of `InputPanel.svelte` add:
```css
  option.native { font-weight: 700; }
  .lockdtype { font-size: 0.85rem; display: flex; align-items: center; gap: 0.35rem; }
```

- [ ] **Step 5: check + build + browser smoke**

```bash
npm run check 2>&1 | tail -1     # 0 errors
npm run build 2>&1 | tail -1     # clean
```
Then `npm run dev`, open the served URL: pick DeepSeek-R1 → Weights/Activations show `fp8 — native` (bold in Firefox) and the selects default to fp8; tick "Lock dtype", switch to Llama → quant stays fp8; untick, switch model → follows native; reload a copied URL → restores. Stop the dev server.

- [ ] **Step 6: Commit**

```bash
git -C /Users/yao/workspace/llm-perf add calc/src/ui/InputPanel.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): Lock-dtype checkbox + native-precision option cue"
```

---

### Task 5: ModelSpecSheet — Native precision row

**Files:**
- Modify: `calc/src/ui/ModelSpecSheet.svelte`

- [ ] **Step 1: Add the row**

The Design section currently is:
```svelte
  <h3>Design</h3>
  <dl>
    <dt>Architecture</dt>
    <dd>{arch.type === 'moe' ? 'Mixture of experts' : 'Dense'}</dd>
    <dt>Attention</dt><dd>{m.attentionLabel}</dd>
  </dl>
```
Replace with:
```svelte
  <h3>Design</h3>
  <dl>
    <dt>Architecture</dt>
    <dd>{arch.type === 'moe' ? 'Mixture of experts' : 'Dense'}</dd>
    <dt>Attention</dt><dd>{m.attentionLabel}</dd>
    <dt>Native precision</dt><dd>{model.nativeDtype}</dd>
  </dl>
```

- [ ] **Step 2: check + commit**

```bash
cd /Users/yao/workspace/llm-perf/calc && npm run check 2>&1 | tail -1   # 0 errors
git -C /Users/yao/workspace/llm-perf add calc/src/ui/ModelSpecSheet.svelte
git -C /Users/yao/workspace/llm-perf commit -m "feat(calc): show native precision in model spec sheet"
```

---

### Task 6: Final integration verification

**Files:** none (verification only)

- [ ] **Step 1: Full gate**

```bash
cd /Users/yao/workspace/llm-perf/calc
npm test 2>&1 | grep -E "Tests |Test Files" | tail -2     # all green
npm run check 2>&1 | tail -1                                # 0 errors
npm run build 2>&1 | tail -1                                # clean
node /Users/yao/workspace/llm-perf/.claude/hooks/check-skill-sync.mjs   # ✓ in sync
```

- [ ] **Step 2: Behavioral spot-check via dev server**

`npm run dev`; verify the spec's success criteria end-to-end: default model shows its native dtype selected; DeepSeek → fp8 default; lock pins across switches; unlock resumes following; a copied `#calc?...` URL with `ld=1` restores locked; a legacy `#calc?...w=fp8&...` (no `ld`) restores locked. Stop the dev server. No commit (verification only).

---

## Self-Review

**1. Spec coverage:**
- Spec §1 schema+data+skill-sync → Task 1 (field, backfill incl. 5 DeepSeek fp8, fixtures sweep, hook+test+SKILL.md, data test). ✓
- Spec §2 behavior (lockDtype store, guarded modelId reseed, KV untouched, no auto-lock) → Task 2. ✓
- Spec §3 URL/share (ld key, quant-implies-lock, legacy links, readStoreState/applyToStores/startUrlSync) → Task 3. ✓
- Spec §4 UI (Lock checkbox, native option bold + `— native` suffix, KV excluded, spec-sheet Native precision row) → Tasks 4 & 5. ✓
- Spec §5 testing (data test, store-behavior test, share round-trip incl. quant-implies-lock, skill-sync) → Tasks 1/2/3 tests; components in-browser (Tasks 4/5). ✓
- Spec inline-fixture gap → Task 1 Step 5 explicit. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step shows full code or exact transform. Pass.

**3. Type consistency:** `nativeDtype: Dtype` on `ModelArch` (Task 1) consumed in Task 2 (`m.nativeDtype`), Task 4 (`selectedModel?.nativeDtype`), Task 5 (`model.nativeDtype`). `lockDtype` writable + `initNativeDtypeSync(): () => void` defined Task 2, consumed in main.ts (Task 2) and share.ts (Task 3) and InputPanel (`$lockDtype`, Task 4). `ShareableState.lockDtype: boolean` (Task 3) used consistently in encode/decode/readStoreState/applyToStores. `get` import added to stores.ts (Task 2 Step 3) before use. Consistent.

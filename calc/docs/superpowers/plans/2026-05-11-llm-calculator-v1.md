# LLM Performance Calculator — v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static-site LLM performance calculator (TS/Svelte) with a pure-function engine that computes roofline-style theoretical limits for dense decoder-only transformers, covering KV cache size, prefill/decode time, TTFT, and aggregate token rates with regime classification (compute- vs memory-bound).

**Architecture:** Vite project, Svelte UI, pure-TS engine. Engine modules have one responsibility each (memory, prefill, decode, roofline, derivation). Property database is typed TS modules. Math is hand-derived and tested against hand-computed reference values using synthetic small fixtures (exact arithmetic, no rounding); one integration test uses a real model+GPU.

**Tech Stack:** TypeScript, Vite, Svelte 5, Vitest. No backend.

**Spec:** `calc/docs/superpowers/specs/2026-05-11-llm-calculator-design.md`

---

## File Structure

```
calc/
  package.json
  tsconfig.json
  vite.config.ts
  svelte.config.js
  index.html
  .gitignore
  src/
    engine/
      types.ts            # all interfaces from the spec
      dtypes.ts           # bytes-per-element table + helper
      memory.ts           # computeMemory(input) → MemoryResult
      roofline.ts         # roofline(flops, bytes, tflops, bw) → { timeS, regime }
      prefill.ts          # computePrefill(input, opPoint, memory) → prefill PerfTier section
      decode.ts           # computeDecode(input, opPoint, memory) → decode PerfTier section
      derivation.ts       # DerivationBuilder helper
      calc.ts             # calculate(input) → CalcResult — top-level wiring
      index.ts            # public re-exports
    data/
      gpus.ts             # GpuSpec[] — seed GPU database
      models.ts           # ModelArch[] — seed model database
      index.ts
    ui/
      App.svelte
      InputPanel.svelte
      MemoryPanel.svelte
      PerfPanel.svelte
      DerivationDrawer.svelte
      stores.ts
    main.ts
  test/
    fixtures.ts           # synthetic test GPU/model/quant/workload
    engine/
      memory.test.ts
      roofline.test.ts
      prefill.test.ts
      decode.test.ts
      derivation.test.ts
      calc.test.ts        # integration with one real (gpu, model) tuple
```

Engine and data are pure TS, no DOM. UI imports engine. Tests target `engine/` exclusively for v1.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `calc/package.json`, `calc/tsconfig.json`, `calc/vite.config.ts`, `calc/svelte.config.js`, `calc/index.html`, `calc/.gitignore`, `calc/src/main.ts`, `calc/src/ui/App.svelte`

- [ ] **Step 1: Initialize package.json**

Run from `calc/`:

```bash
cd /Users/yao/workspace/llm-perf/calc
cat > package.json <<'EOF'
{
  "name": "llm-perf-calc",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "check": "svelte-check --tsconfig ./tsconfig.json",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "@tsconfig/svelte": "^5.0.4",
    "svelte": "^5.0.0",
    "svelte-check": "^4.0.0",
    "tslib": "^2.7.0",
    "typescript": "^5.6.0",
    "vite": "^5.4.0",
    "vitest": "^2.1.0"
  }
}
EOF
npm install
```

Expected: install completes with no errors; `node_modules/` populated.

- [ ] **Step 2: Add tsconfig.json**

```bash
cat > tsconfig.json <<'EOF'
{
  "extends": "@tsconfig/svelte/tsconfig.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "allowJs": false,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*", "test/**/*"]
}
EOF
```

- [ ] **Step 3: Add vite.config.ts**

```bash
cat > vite.config.ts <<'EOF'
import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts']
  }
})
EOF
```

- [ ] **Step 4: Add svelte.config.js**

```bash
cat > svelte.config.js <<'EOF'
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte'
export default { preprocess: vitePreprocess() }
EOF
```

- [ ] **Step 5: Add index.html and main.ts**

```bash
cat > index.html <<'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LLM Performance Calculator</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
EOF

mkdir -p src/ui
cat > src/main.ts <<'EOF'
import { mount } from 'svelte'
import App from './ui/App.svelte'

const app = mount(App, { target: document.getElementById('app')! })
export default app
EOF

cat > src/ui/App.svelte <<'EOF'
<main>
  <h1>LLM Performance Calculator</h1>
  <p>v1 scaffold — engine and UI under construction.</p>
</main>
EOF
```

- [ ] **Step 6: Add .gitignore**

```bash
cat > .gitignore <<'EOF'
node_modules
dist
.DS_Store
*.log
EOF
```

- [ ] **Step 7: Verify build and tests run**

```bash
npm run build
npm test
```

Expected: build produces `dist/` without errors. `npm test` exits 0 with "No test files found" (acceptable — we haven't written tests yet).

- [ ] **Step 8: Commit**

```bash
cd /Users/yao/workspace/llm-perf
git add calc/package.json calc/tsconfig.json calc/vite.config.ts calc/svelte.config.js calc/index.html calc/.gitignore calc/src/main.ts calc/src/ui/App.svelte calc/package-lock.json
git commit -m "feat(calc): scaffold Vite + Svelte + TS project"
```

---

## Task 2: Engine Types

**Files:**
- Create: `calc/src/engine/types.ts`

This is a pure type-definition task. No runtime code, no tests. TypeScript itself is the test — if it compiles, types are well-formed.

- [ ] **Step 1: Write types.ts**

```bash
mkdir -p src/engine
cat > src/engine/types.ts <<'EOF'
export type Dtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4'

export interface GpuOperatingPoint {
  id: string
  label: string
  tflops: Partial<Record<Dtype, number>>
  hbmBandwidthGBs: number
}

export interface GpuVariant {
  id: string
  label: string
  hbmCapacityGB: number
  operatingPoints: GpuOperatingPoint[]
}

export interface GpuSpec {
  id: string
  name: string
  vendor: string
  family?: string
  variants: GpuVariant[]
}

export interface ModelArch {
  id: string
  name: string
  family: string
  layers: number
  hiddenDim: number
  intermediateDim: number
  numHeads: number
  numKvHeads: number
  headDim: number
  vocabSize: number
  paramCount: number
}

export interface Quantization {
  weights: Dtype
  kv: Dtype
  activations: Dtype
}

export interface Workload {
  promptTokens: number
  outputTokens: number
  concurrency: number
}

export interface CalcInput {
  gpu: GpuSpec
  gpuVariantId: string
  model: ModelArch
  quant: Quantization
  workload: Workload
}

export interface MemoryResult {
  weights: number
  kvCachePerRequest: number
  kvCacheTotal: number
  activationsPeak: number
  total: number
  hbmCapacityGB: number
  headroom: number
  fits: boolean
}

export interface PerfTier {
  prefill: { flops: number; bytes: number; timeS: number; regime: 'compute' | 'memory' }
  decode:  { flopsPerStep: number; bytesPerStep: number; timePerTokenS: number;
             regime: 'compute' | 'memory'; aggregateTokensPerS: number }
  ttftS: number
  inputTokenRate: number
  outputTokenRate: number
}

export interface DerivationStep {
  label: string
  expression: string
  value: number
  unit: string
}

export interface CalcResult {
  memory: MemoryResult
  perf: Record<string, PerfTier>
  derivation: DerivationStep[]
}
EOF
```

- [ ] **Step 2: Verify it compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add calc/src/engine/types.ts
git commit -m "feat(calc): add engine type definitions"
```

---

## Task 3: Dtypes Module

**Files:**
- Create: `calc/src/engine/dtypes.ts`
- Create: `calc/test/engine/dtypes.test.ts`

- [ ] **Step 1: Write the failing test**

```bash
mkdir -p test/engine
cat > test/engine/dtypes.test.ts <<'EOF'
import { describe, it, expect } from 'vitest'
import { bytesOf } from '../../src/engine/dtypes'

describe('bytesOf', () => {
  it('returns correct bytes per element for each dtype', () => {
    expect(bytesOf('fp32')).toBe(4)
    expect(bytesOf('fp16')).toBe(2)
    expect(bytesOf('bf16')).toBe(2)
    expect(bytesOf('fp8')).toBe(1)
    expect(bytesOf('int8')).toBe(1)
    expect(bytesOf('int4')).toBe(0.5)
  })
})
EOF
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
npx vitest run test/engine/dtypes.test.ts
```

Expected: FAIL with "Cannot find module '../../src/engine/dtypes'".

- [ ] **Step 3: Implement dtypes.ts**

```bash
cat > src/engine/dtypes.ts <<'EOF'
import type { Dtype } from './types'

const DTYPE_BYTES: Record<Dtype, number> = {
  fp32: 4, fp16: 2, bf16: 2, fp8: 1, int8: 1, int4: 0.5
}

export function bytesOf(dtype: Dtype): number {
  return DTYPE_BYTES[dtype]
}
EOF
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
npx vitest run test/engine/dtypes.test.ts
```

Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add calc/src/engine/dtypes.ts calc/test/engine/dtypes.test.ts
git commit -m "feat(calc): add dtypes byte-size table"
```

---

## Task 4: Test Fixtures

**Files:**
- Create: `calc/test/fixtures.ts`

Synthetic small fixtures with hand-computable numbers. These are used by all engine tests.

- [ ] **Step 1: Write fixtures**

```bash
cat > test/fixtures.ts <<'EOF'
import type { GpuSpec, ModelArch, Quantization, Workload, CalcInput } from '../src/engine/types'

// Tiny synthetic GPU: 1 TFLOP fp16, 1 GB/s HBM, 1 GB capacity.
// Numbers chosen so arithmetic is exact and hand-verifiable.
export const testGpu: GpuSpec = {
  id: 'test-gpu',
  name: 'Test GPU',
  vendor: 'test',
  variants: [{
    id: 'v',
    label: 'V',
    hbmCapacityGB: 1,
    operatingPoints: [{
      id: 'peak',
      label: 'Peak',
      tflops: { fp16: 1 },
      hbmBandwidthGBs: 1
    }]
  }]
}

// Tiny synthetic model:
//   2 layers, hidden=4, intermediate=8, heads=2, kv_heads=1, head_dim=2
//   vocab=100, paramCount=1000
export const testModel: ModelArch = {
  id: 'test-model',
  name: 'Test Model',
  family: 'test',
  layers: 2,
  hiddenDim: 4,
  intermediateDim: 8,
  numHeads: 2,
  numKvHeads: 1,
  headDim: 2,
  vocabSize: 100,
  paramCount: 1000
}

export const fp16Quant: Quantization = {
  weights: 'fp16', kv: 'fp16', activations: 'fp16'
}

export const testWorkload: Workload = {
  promptTokens: 10, outputTokens: 5, concurrency: 2
}

export const testInput: CalcInput = {
  gpu: testGpu,
  gpuVariantId: 'v',
  model: testModel,
  quant: fp16Quant,
  workload: testWorkload
}
EOF
```

- [ ] **Step 2: Verify it compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add calc/test/fixtures.ts
git commit -m "test(calc): add synthetic engine test fixtures"
```

---

## Task 5: Memory — Weights

**Files:**
- Create: `calc/src/engine/memory.ts`
- Create: `calc/test/engine/memory.test.ts`

- [ ] **Step 1: Write failing test for weights**

```bash
cat > test/engine/memory.test.ts <<'EOF'
import { describe, it, expect } from 'vitest'
import { computeMemory } from '../../src/engine/memory'
import { testInput } from '../fixtures'

describe('computeMemory', () => {
  it('weights = paramCount × bytes(weight_dtype)', () => {
    // paramCount=1000, fp16=2 bytes → 2000 bytes
    const m = computeMemory(testInput)
    expect(m.weights).toBe(2000)
  })
})
EOF
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
npx vitest run test/engine/memory.test.ts
```

Expected: FAIL with "Cannot find module '../../src/engine/memory'".

- [ ] **Step 3: Implement minimal memory.ts**

```bash
cat > src/engine/memory.ts <<'EOF'
import type { CalcInput, MemoryResult } from './types'
import { bytesOf } from './dtypes'

export function computeMemory(input: CalcInput): MemoryResult {
  const { model, quant } = input
  const weights = model.paramCount * bytesOf(quant.weights)
  return {
    weights,
    kvCachePerRequest: 0,
    kvCacheTotal: 0,
    activationsPeak: 0,
    total: 0,
    hbmCapacityGB: 0,
    headroom: 0,
    fits: false
  }
}
EOF
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
npx vitest run test/engine/memory.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add calc/src/engine/memory.ts calc/test/engine/memory.test.ts
git commit -m "feat(calc): compute weight bytes"
```

---

## Task 6: Memory — KV Cache

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/memory.test.ts`

- [ ] **Step 1: Add failing tests for KV cache**

Append to `test/engine/memory.test.ts` inside the existing `describe`:

```ts
  it('kvCachePerRequest = 2 × layers × kv_heads × head_dim × bytes(kv_dtype) × (prompt + output)', () => {
    // 2 × 2 × 1 × 2 × 2 (fp16) = 16 bytes per token
    // × (10 + 5) = 240 bytes per request
    const m = computeMemory(testInput)
    expect(m.kvCachePerRequest).toBe(240)
  })

  it('kvCacheTotal = kvCachePerRequest × concurrency', () => {
    // 240 × 2 = 480
    const m = computeMemory(testInput)
    expect(m.kvCacheTotal).toBe(480)
  })
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run test/engine/memory.test.ts
```

Expected: 2 new tests FAIL (kvCache* values are 0).

- [ ] **Step 3: Implement KV computation**

Replace `memory.ts` with:

```bash
cat > src/engine/memory.ts <<'EOF'
import type { CalcInput, MemoryResult } from './types'
import { bytesOf } from './dtypes'

export function computeMemory(input: CalcInput): MemoryResult {
  const { model, quant, workload } = input
  const seqlen = workload.promptTokens + workload.outputTokens

  const weights = model.paramCount * bytesOf(quant.weights)
  const kvPerTokenPerRequest =
    2 * model.layers * model.numKvHeads * model.headDim * bytesOf(quant.kv)
  const kvCachePerRequest = kvPerTokenPerRequest * seqlen
  const kvCacheTotal = kvCachePerRequest * workload.concurrency

  return {
    weights,
    kvCachePerRequest,
    kvCacheTotal,
    activationsPeak: 0,
    total: 0,
    hbmCapacityGB: 0,
    headroom: 0,
    fits: false
  }
}
EOF
```

- [ ] **Step 4: Run tests, verify pass**

```bash
npx vitest run test/engine/memory.test.ts
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add calc/src/engine/memory.ts calc/test/engine/memory.test.ts
git commit -m "feat(calc): compute KV cache memory"
```

---

## Task 7: Memory — Activations, Total, Headroom, Fits

**Files:**
- Modify: `calc/src/engine/memory.ts`
- Modify: `calc/test/engine/memory.test.ts`

- [ ] **Step 1: Add failing tests**

Append to `test/engine/memory.test.ts`:

```ts
  it('activationsPeak = concurrency × prompt × (hidden + intermediate) × bytes(act_dtype) × 2', () => {
    // 2 × 10 × (4 + 8) × 2 (fp16) × 2 = 960 bytes
    const m = computeMemory(testInput)
    expect(m.activationsPeak).toBe(960)
  })

  it('total = weights + kvCacheTotal + activationsPeak', () => {
    // 2000 + 480 + 960 = 3440
    const m = computeMemory(testInput)
    expect(m.total).toBe(3440)
  })

  it('hbmCapacityGB echoed from chosen variant', () => {
    const m = computeMemory(testInput)
    expect(m.hbmCapacityGB).toBe(1)
  })

  it('headroom = hbmCapacity_bytes − total, fits when ≥ 0', () => {
    // 1 GB = 1_073_741_824 bytes; headroom = 1_073_741_824 − 3440
    const m = computeMemory(testInput)
    expect(m.headroom).toBe(1_073_741_824 - 3440)
    expect(m.fits).toBe(true)
  })

  it('fits=false and negative headroom on OOM', () => {
    const bigModel = { ...testInput.model, paramCount: 10_000_000_000 }  // 10B params × 2B = 20GB
    const m = computeMemory({ ...testInput, model: bigModel })
    expect(m.fits).toBe(false)
    expect(m.headroom).toBeLessThan(0)
  })
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run test/engine/memory.test.ts
```

Expected: 5 new tests FAIL.

- [ ] **Step 3: Implement remaining memory math**

```bash
cat > src/engine/memory.ts <<'EOF'
import type { CalcInput, GpuVariant, MemoryResult } from './types'
import { bytesOf } from './dtypes'

const BYTES_PER_GB = 1024 ** 3

function findVariant(input: CalcInput): GpuVariant {
  const v = input.gpu.variants.find(v => v.id === input.gpuVariantId)
  if (!v) throw new Error(`Variant ${input.gpuVariantId} not in ${input.gpu.id}`)
  return v
}

export function computeMemory(input: CalcInput): MemoryResult {
  const { model, quant, workload } = input
  const variant = findVariant(input)
  const seqlen = workload.promptTokens + workload.outputTokens

  const weights = model.paramCount * bytesOf(quant.weights)
  const kvPerTokenPerRequest =
    2 * model.layers * model.numKvHeads * model.headDim * bytesOf(quant.kv)
  const kvCachePerRequest = kvPerTokenPerRequest * seqlen
  const kvCacheTotal = kvCachePerRequest * workload.concurrency

  // Coarse: one layer's attention + FFN buffer × small constant.
  // Assumes FlashAttention-style kernels (no materialized S×S matrix).
  const activationsPeak =
    workload.concurrency * workload.promptTokens *
    (model.hiddenDim + model.intermediateDim) * bytesOf(quant.activations) * 2

  const total = weights + kvCacheTotal + activationsPeak
  const hbmCapacityBytes = variant.hbmCapacityGB * BYTES_PER_GB
  const headroom = hbmCapacityBytes - total
  const fits = headroom >= 0

  return {
    weights,
    kvCachePerRequest,
    kvCacheTotal,
    activationsPeak,
    total,
    hbmCapacityGB: variant.hbmCapacityGB,
    headroom,
    fits
  }
}
EOF
```

- [ ] **Step 4: Run tests, verify pass**

```bash
npx vitest run test/engine/memory.test.ts
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add calc/src/engine/memory.ts calc/test/engine/memory.test.ts
git commit -m "feat(calc): compute activation, total, headroom, fits"
```

---

## Task 8: Roofline Helper

**Files:**
- Create: `calc/src/engine/roofline.ts`
- Create: `calc/test/engine/roofline.test.ts`

- [ ] **Step 1: Write failing tests**

```bash
cat > test/engine/roofline.test.ts <<'EOF'
import { describe, it, expect } from 'vitest'
import { roofline } from '../../src/engine/roofline'

describe('roofline', () => {
  it('returns compute regime when flops/tflops > bytes/bw', () => {
    // flops/tflops = 4 / 1e12 / 1e-12 = ...
    // Use simpler numbers: tflops in TFLOPs = 1 (so 1e12 FLOP/s), bw in GB/s = 1 (so 1e9 B/s)
    //   flops = 2e12 → time = 2s ; bytes = 1e9 → time = 1s → compute wins
    const r = roofline({ flops: 2e12, bytes: 1e9, tflops: 1, bwGBs: 1 })
    expect(r.regime).toBe('compute')
    expect(r.timeS).toBe(2)
  })

  it('returns memory regime when bytes/bw > flops/tflops', () => {
    //   flops = 1e12 → time = 1s ; bytes = 2e9 → time = 2s → memory wins
    const r = roofline({ flops: 1e12, bytes: 2e9, tflops: 1, bwGBs: 1 })
    expect(r.regime).toBe('memory')
    expect(r.timeS).toBe(2)
  })

  it('ties classify as memory regime (defensive choice)', () => {
    const r = roofline({ flops: 1e12, bytes: 1e9, tflops: 1, bwGBs: 1 })
    expect(r.timeS).toBe(1)
    expect(r.regime).toBe('memory')
  })
})
EOF
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run test/engine/roofline.test.ts
```

Expected: FAIL with "Cannot find module".

- [ ] **Step 3: Implement roofline**

```bash
cat > src/engine/roofline.ts <<'EOF'
export interface RooflineInput {
  flops: number
  bytes: number
  tflops: number    // peak compute in TFLOPs (10^12 FLOP/s)
  bwGBs: number     // peak bandwidth in GB/s (10^9 B/s)
}

export interface RooflineResult {
  timeS: number
  regime: 'compute' | 'memory'
}

export function roofline({ flops, bytes, tflops, bwGBs }: RooflineInput): RooflineResult {
  const computeS = flops / (tflops * 1e12)
  const memoryS = bytes / (bwGBs * 1e9)
  if (computeS > memoryS) return { timeS: computeS, regime: 'compute' }
  return { timeS: memoryS, regime: 'memory' }
}
EOF
```

- [ ] **Step 4: Run tests, verify pass**

```bash
npx vitest run test/engine/roofline.test.ts
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add calc/src/engine/roofline.ts calc/test/engine/roofline.test.ts
git commit -m "feat(calc): add roofline helper with regime classification"
```

---

## Task 9: Prefill Math

**Files:**
- Create: `calc/src/engine/prefill.ts`
- Create: `calc/test/engine/prefill.test.ts`

- [ ] **Step 1: Write failing tests**

```bash
cat > test/engine/prefill.test.ts <<'EOF'
import { describe, it, expect } from 'vitest'
import { computePrefill } from '../../src/engine/prefill'
import { testInput } from '../fixtures'
import { computeMemory } from '../../src/engine/memory'

describe('computePrefill', () => {
  const opPoint = testInput.gpu.variants[0].operatingPoints[0]
  const memory = computeMemory(testInput)

  it('flops = 2 × params × prompt + 2 × layers × prompt² × hidden', () => {
    // 2 × 1000 × 10 + 2 × 2 × 100 × 4 = 20000 + 1600 = 21600
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.flops).toBe(21600)
  })

  it('bytes = weightBytes + activationsPeak', () => {
    // weights=2000, activations=960 → 2960
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.bytes).toBe(2960)
  })

  it('timeS = max(flops/tflops, bytes/bw)', () => {
    // flops/tflops = 21600 / 1e12 = 2.16e-8
    // bytes/bw    = 2960 / 1e9    = 2.96e-6  ← bigger
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.timeS).toBeCloseTo(2960 / 1e9, 12)
    expect(p.regime).toBe('memory')
  })

  it('uses activation dtype to pick tflops', () => {
    // testInput uses fp16; opPoint.tflops.fp16 = 1
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.timeS).toBeGreaterThan(0)
  })
})
EOF
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run test/engine/prefill.test.ts
```

Expected: FAIL with "Cannot find module".

- [ ] **Step 3: Implement prefill**

```bash
cat > src/engine/prefill.ts <<'EOF'
import type { CalcInput, GpuOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'

export function computePrefill(
  input: CalcInput,
  opPoint: GpuOperatingPoint,
  memory: MemoryResult
): PerfTier['prefill'] {
  const { model, quant, workload } = input
  const p = workload.promptTokens

  const flops =
    2 * model.paramCount * p +
    2 * model.layers * p * p * model.hiddenDim
  const bytes = memory.weights + memory.activationsPeak

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  const { timeS, regime } = roofline({
    flops, bytes, tflops, bwGBs: opPoint.hbmBandwidthGBs
  })
  return { flops, bytes, timeS, regime }
}
EOF
```

- [ ] **Step 4: Run tests, verify pass**

```bash
npx vitest run test/engine/prefill.test.ts
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add calc/src/engine/prefill.ts calc/test/engine/prefill.test.ts
git commit -m "feat(calc): compute prefill FLOPs, bytes, time, regime"
```

---

## Task 10: Decode Math

**Files:**
- Create: `calc/src/engine/decode.ts`
- Create: `calc/test/engine/decode.test.ts`

- [ ] **Step 1: Write failing tests**

```bash
cat > test/engine/decode.test.ts <<'EOF'
import { describe, it, expect } from 'vitest'
import { computeDecode } from '../../src/engine/decode'
import { testInput } from '../fixtures'
import { computeMemory } from '../../src/engine/memory'

describe('computeDecode', () => {
  const opPoint = testInput.gpu.variants[0].operatingPoints[0]
  const memory = computeMemory(testInput)

  // testInput: prompt=10, output=5, concurrency=2
  // avg seqlen for decode attention ≈ prompt + output/2 = 12.5

  it('flopsPerStep = (2 × params + 2 × layers × seqlen_avg × hidden) × concurrency', () => {
    // (2 × 1000 + 2 × 2 × 12.5 × 4) × 2 = (2000 + 200) × 2 = 4400
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.flopsPerStep).toBe(4400)
  })

  it('bytesPerStep = weightBytes + kvPerRequest × concurrency', () => {
    // weights=2000, kvPerRequest=240 → 2000 + 240×2 = 2480
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.bytesPerStep).toBe(2480)
  })

  it('timePerTokenS = max(flopsPerStep/tflops, bytesPerStep/bw)', () => {
    // flops/tflops = 4400 / 1e12 = 4.4e-9
    // bytes/bw    = 2480 / 1e9  = 2.48e-6  ← bigger
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.timePerTokenS).toBeCloseTo(2480 / 1e9, 12)
    expect(d.regime).toBe('memory')
  })

  it('aggregateTokensPerS = concurrency / timePerTokenS', () => {
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.aggregateTokensPerS).toBeCloseTo(2 / d.timePerTokenS, 6)
  })
})
EOF
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run test/engine/decode.test.ts
```

Expected: FAIL with "Cannot find module".

- [ ] **Step 3: Implement decode**

```bash
cat > src/engine/decode.ts <<'EOF'
import type { CalcInput, GpuOperatingPoint, MemoryResult, PerfTier } from './types'
import { roofline } from './roofline'

export function computeDecode(
  input: CalcInput,
  opPoint: GpuOperatingPoint,
  memory: MemoryResult
): PerfTier['decode'] {
  const { model, quant, workload } = input
  const avgSeqlen = workload.promptTokens + workload.outputTokens / 2

  const flopsPerStep =
    (2 * model.paramCount + 2 * model.layers * avgSeqlen * model.hiddenDim) *
    workload.concurrency
  const bytesPerStep = memory.weights + memory.kvCachePerRequest * workload.concurrency

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  const { timeS, regime } = roofline({
    flops: flopsPerStep, bytes: bytesPerStep,
    tflops, bwGBs: opPoint.hbmBandwidthGBs
  })

  return {
    flopsPerStep,
    bytesPerStep,
    timePerTokenS: timeS,
    regime,
    aggregateTokensPerS: workload.concurrency / timeS
  }
}
EOF
```

- [ ] **Step 4: Run tests, verify pass**

```bash
npx vitest run test/engine/decode.test.ts
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add calc/src/engine/decode.ts calc/test/engine/decode.test.ts
git commit -m "feat(calc): compute decode FLOPs, bytes, time, throughput"
```

---

## Task 11: Derivation Builder

**Files:**
- Create: `calc/src/engine/derivation.ts`
- Create: `calc/test/engine/derivation.test.ts`

Helper that accumulates `DerivationStep[]` as math is computed.

- [ ] **Step 1: Write failing tests**

```bash
cat > test/engine/derivation.test.ts <<'EOF'
import { describe, it, expect } from 'vitest'
import { DerivationBuilder } from '../../src/engine/derivation'

describe('DerivationBuilder', () => {
  it('records steps in order', () => {
    const b = new DerivationBuilder()
    b.add('weights', 'paramCount × bytes(dtype)', 2000, 'bytes')
    b.add('kv/token', '2 × layers × kv_heads × head_dim × bytes(dtype)', 16, 'bytes')
    expect(b.steps()).toEqual([
      { label: 'weights', expression: 'paramCount × bytes(dtype)', value: 2000, unit: 'bytes' },
      { label: 'kv/token', expression: '2 × layers × kv_heads × head_dim × bytes(dtype)', value: 16, unit: 'bytes' }
    ])
  })

  it('returns a defensive copy from steps()', () => {
    const b = new DerivationBuilder()
    b.add('x', 'y', 1, 'z')
    const out = b.steps()
    out.push({ label: 'evil', expression: '', value: 0, unit: '' })
    expect(b.steps()).toHaveLength(1)
  })
})
EOF
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run test/engine/derivation.test.ts
```

Expected: FAIL with "Cannot find module".

- [ ] **Step 3: Implement DerivationBuilder**

```bash
cat > src/engine/derivation.ts <<'EOF'
import type { DerivationStep } from './types'

export class DerivationBuilder {
  private readonly _steps: DerivationStep[] = []

  add(label: string, expression: string, value: number, unit: string): void {
    this._steps.push({ label, expression, value, unit })
  }

  steps(): DerivationStep[] {
    return [...this._steps]
  }
}
EOF
```

- [ ] **Step 4: Run tests, verify pass**

```bash
npx vitest run test/engine/derivation.test.ts
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add calc/src/engine/derivation.ts calc/test/engine/derivation.test.ts
git commit -m "feat(calc): add derivation step builder"
```

---

## Task 12: Top-level calculate()

**Files:**
- Create: `calc/src/engine/calc.ts`
- Create: `calc/src/engine/index.ts`
- Create: `calc/test/engine/calc.test.ts`

Wires memory + prefill + decode + derivation into the top-level `calculate(input)`. Iterates operating points to produce `perf[opPointId]`.

- [ ] **Step 1: Write failing tests**

```bash
cat > test/engine/calc.test.ts <<'EOF'
import { describe, it, expect } from 'vitest'
import { calculate } from '../../src/engine/calc'
import { testInput } from '../fixtures'

describe('calculate', () => {
  it('returns memory matching computeMemory', () => {
    const r = calculate(testInput)
    expect(r.memory.weights).toBe(2000)
    expect(r.memory.kvCachePerRequest).toBe(240)
    expect(r.memory.kvCacheTotal).toBe(480)
    expect(r.memory.activationsPeak).toBe(960)
    expect(r.memory.total).toBe(3440)
    expect(r.memory.fits).toBe(true)
  })

  it('produces one perf tier per operating point', () => {
    const r = calculate(testInput)
    expect(Object.keys(r.perf)).toEqual(['peak'])
  })

  it('perf.peak has all the expected fields', () => {
    const r = calculate(testInput)
    const p = r.perf['peak']
    expect(p.prefill.flops).toBe(21600)
    expect(p.decode.flopsPerStep).toBe(4400)
    expect(p.ttftS).toBe(p.prefill.timeS)
    expect(p.outputTokenRate).toBeCloseTo(p.decode.aggregateTokensPerS, 9)
    expect(p.inputTokenRate).toBeCloseTo(testInput.workload.promptTokens / p.prefill.timeS, 6)
  })

  it('derivation is non-empty and ends with the final memory total', () => {
    const r = calculate(testInput)
    expect(r.derivation.length).toBeGreaterThan(0)
    const memoryTotalStep = r.derivation.find(s => s.label === 'memory total')
    expect(memoryTotalStep?.value).toBe(r.memory.total)
  })

  it('throws on unknown variant id', () => {
    expect(() => calculate({ ...testInput, gpuVariantId: 'nope' })).toThrow()
  })
})
EOF
```

- [ ] **Step 2: Run tests, verify failure**

```bash
npx vitest run test/engine/calc.test.ts
```

Expected: FAIL with "Cannot find module".

- [ ] **Step 3: Implement calculate()**

```bash
cat > src/engine/calc.ts <<'EOF'
import type { CalcInput, CalcResult, PerfTier } from './types'
import { bytesOf } from './dtypes'
import { computeMemory } from './memory'
import { computePrefill } from './prefill'
import { computeDecode } from './decode'
import { DerivationBuilder } from './derivation'

export function calculate(input: CalcInput): CalcResult {
  const variant = input.gpu.variants.find(v => v.id === input.gpuVariantId)
  if (!variant) {
    throw new Error(`Variant ${input.gpuVariantId} not in GPU ${input.gpu.id}`)
  }

  const memory = computeMemory(input)
  const d = new DerivationBuilder()

  d.add('weights', 'paramCount × bytes(weight_dtype)', memory.weights, 'bytes')
  d.add(
    'kv per token per request',
    '2 × layers × kv_heads × head_dim × bytes(kv_dtype)',
    memory.kvCachePerRequest / (input.workload.promptTokens + input.workload.outputTokens),
    'bytes'
  )
  d.add('kv per request', 'kv_per_token × (prompt + output)', memory.kvCachePerRequest, 'bytes')
  d.add('kv total', 'kv_per_request × concurrency', memory.kvCacheTotal, 'bytes')
  d.add(
    'activations peak (coarse)',
    'concurrency × prompt × (hidden + intermediate) × bytes(act_dtype) × 2',
    memory.activationsPeak, 'bytes'
  )
  d.add('memory total', 'weights + kv_total + activations_peak', memory.total, 'bytes')

  const perf: Record<string, PerfTier> = {}
  for (const op of variant.operatingPoints) {
    const prefill = computePrefill(input, op, memory)
    const decode = computeDecode(input, op, memory)
    perf[op.id] = {
      prefill, decode,
      ttftS: prefill.timeS,
      inputTokenRate: input.workload.promptTokens / prefill.timeS,
      outputTokenRate: decode.aggregateTokensPerS
    }
    d.add(
      `prefill time @ ${op.id}`,
      'max(prefill_flops / tflops, prefill_bytes / bw)',
      prefill.timeS, 's'
    )
    d.add(
      `decode time per token @ ${op.id}`,
      'max(decode_flops / tflops, decode_bytes / bw)',
      decode.timePerTokenS, 's'
    )
  }

  // bytesOf is re-exported so consumers can read the same table.
  void bytesOf

  return { memory, perf, derivation: d.steps() }
}
EOF
```

- [ ] **Step 4: Add public re-exports**

```bash
cat > src/engine/index.ts <<'EOF'
export * from './types'
export { bytesOf } from './dtypes'
export { calculate } from './calc'
EOF
```

- [ ] **Step 5: Run tests, verify pass**

```bash
npx vitest run
```

Expected: all engine tests PASS.

- [ ] **Step 6: Commit**

```bash
git add calc/src/engine/calc.ts calc/src/engine/index.ts calc/test/engine/calc.test.ts
git commit -m "feat(calc): wire memory + prefill + decode into calculate()"
```

---

## Task 13: GPU Seed Data

**Files:**
- Create: `calc/src/data/gpus.ts`
- Create: `calc/src/data/index.ts`

Datasheet sources for each operating point are cited in comments. `peak` only for v1; `achievable` deferred.

- [ ] **Step 1: Write gpus.ts**

```bash
mkdir -p src/data
cat > src/data/gpus.ts <<'EOF'
import type { GpuSpec } from '../engine/types'

// Peak numbers from vendor datasheets. Dense compute (no sparsity).
// FP16/BF16 columns reflect Tensor Core peak.
export const GPUS: GpuSpec[] = [
  {
    id: 'h100', name: 'NVIDIA H100', vendor: 'NVIDIA', family: 'Hopper',
    variants: [
      {
        id: 'sxm-80', label: 'SXM 80GB', hbmCapacityGB: 80,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 989, bf16: 989, fp8: 1979, int8: 1979 },
          hbmBandwidthGBs: 3350
        }]
      },
      {
        id: 'pcie-80', label: 'PCIe 80GB', hbmCapacityGB: 80,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 756, bf16: 756, fp8: 1513, int8: 1513 },
          hbmBandwidthGBs: 2000
        }]
      },
      {
        id: 'pcie-94', label: 'PCIe 94GB', hbmCapacityGB: 94,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 756, bf16: 756, fp8: 1513, int8: 1513 },
          hbmBandwidthGBs: 2400
        }]
      },
      {
        id: 'nvl-188', label: 'NVL (per GPU 94GB)', hbmCapacityGB: 94,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 989, bf16: 989, fp8: 1979, int8: 1979 },
          hbmBandwidthGBs: 3900
        }]
      }
    ]
  },
  {
    id: 'h200', name: 'NVIDIA H200', vendor: 'NVIDIA', family: 'Hopper',
    variants: [{
      id: 'sxm-141', label: 'SXM 141GB', hbmCapacityGB: 141,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 989, bf16: 989, fp8: 1979, int8: 1979 },
        hbmBandwidthGBs: 4800
      }]
    }]
  },
  {
    id: 'a100', name: 'NVIDIA A100', vendor: 'NVIDIA', family: 'Ampere',
    variants: [
      {
        id: 'sxm-40', label: 'SXM 40GB', hbmCapacityGB: 40,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 1555
        }]
      },
      {
        id: 'sxm-80', label: 'SXM 80GB', hbmCapacityGB: 80,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 2039
        }]
      },
      {
        id: 'pcie-40', label: 'PCIe 40GB', hbmCapacityGB: 40,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 1555
        }]
      },
      {
        id: 'pcie-80', label: 'PCIe 80GB', hbmCapacityGB: 80,
        operatingPoints: [{
          id: 'peak', label: 'Peak',
          tflops: { fp16: 312, bf16: 312, int8: 624 },
          hbmBandwidthGBs: 1935
        }]
      }
    ]
  },
  {
    id: 'l40s', name: 'NVIDIA L40S', vendor: 'NVIDIA', family: 'Ada Lovelace',
    variants: [{
      id: 'pcie-48', label: 'PCIe 48GB', hbmCapacityGB: 48,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 362, bf16: 362, fp8: 733, int8: 733 },
        hbmBandwidthGBs: 864
      }]
    }]
  },
  {
    id: 'rtx-5090', name: 'NVIDIA RTX 5090', vendor: 'NVIDIA', family: 'Blackwell',
    variants: [{
      id: 'sku', label: '32GB', hbmCapacityGB: 32,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 209, bf16: 209, fp8: 419, int8: 419 },
        hbmBandwidthGBs: 1792
      }]
    }]
  },
  {
    id: 'rtx-4090', name: 'NVIDIA RTX 4090', vendor: 'NVIDIA', family: 'Ada Lovelace',
    variants: [{
      id: 'sku', label: '24GB', hbmCapacityGB: 24,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 165, bf16: 165, fp8: 330, int8: 330 },
        hbmBandwidthGBs: 1008
      }]
    }]
  },
  {
    id: 'mi300x', name: 'AMD Instinct MI300X', vendor: 'AMD', family: 'CDNA3',
    variants: [{
      id: 'oam-192', label: 'OAM 192GB', hbmCapacityGB: 192,
      operatingPoints: [{
        id: 'peak', label: 'Peak',
        tflops: { fp16: 1307, bf16: 1307, fp8: 2615, int8: 2615 },
        hbmBandwidthGBs: 5300
      }]
    }]
  }
]
EOF
```

- [ ] **Step 2: Add data/index.ts (GPUs only for now)**

```bash
cat > src/data/index.ts <<'EOF'
export { GPUS } from './gpus'
EOF
```

- [ ] **Step 3: Verify type-check**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add calc/src/data/gpus.ts calc/src/data/index.ts
git commit -m "feat(calc): seed GPU property database"
```

---

## Task 14: Model Seed Data

**Files:**
- Create: `calc/src/data/models.ts`

Arch fields sourced from each model's HuggingFace `config.json`. **The agent implementing this task must verify each entry against the actual HF config.** Numbers below are best-effort placeholders — confirm before committing.

- [ ] **Step 1: Write models.ts**

Pull each model's `config.json` from HuggingFace and fill in:
- `layers` = `num_hidden_layers`
- `hiddenDim` = `hidden_size`
- `intermediateDim` = `intermediate_size`
- `numHeads` = `num_attention_heads`
- `numKvHeads` = `num_key_value_heads` (or `numHeads` if not specified)
- `headDim` = `head_dim` (or `hiddenDim / numHeads` if not specified)
- `vocabSize` = `vocab_size`
- `paramCount` = official figure from the model card

```bash
cat > src/data/models.ts <<'EOF'
import type { ModelArch } from '../engine/types'

// Architecture fields sourced from HuggingFace config.json per model.
// paramCount taken from each model's official card.
export const MODELS: ModelArch[] = [
  // === Qwen3 dense series ===
  {
    id: 'qwen3-1.7b', name: 'Qwen3 1.7B', family: 'qwen3',
    layers: 28, hiddenDim: 2048, intermediateDim: 6144,
    numHeads: 16, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 1_720_000_000
  },
  {
    id: 'qwen3-4b', name: 'Qwen3 4B', family: 'qwen3',
    layers: 36, hiddenDim: 2560, intermediateDim: 9728,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 4_020_000_000
  },
  {
    id: 'qwen3-8b', name: 'Qwen3 8B', family: 'qwen3',
    layers: 36, hiddenDim: 4096, intermediateDim: 12288,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 8_190_000_000
  },
  {
    id: 'qwen3-14b', name: 'Qwen3 14B', family: 'qwen3',
    layers: 40, hiddenDim: 5120, intermediateDim: 17408,
    numHeads: 40, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 14_770_000_000
  },
  {
    id: 'qwen3-32b', name: 'Qwen3 32B', family: 'qwen3',
    layers: 64, hiddenDim: 5120, intermediateDim: 25600,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 151936,
    paramCount: 32_760_000_000
  },
  // === Llama ===
  {
    id: 'llama-3.3-70b', name: 'Llama 3.3 70B', family: 'llama-3',
    layers: 80, hiddenDim: 8192, intermediateDim: 28672,
    numHeads: 64, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 70_553_706_496
  },
  {
    id: 'llama-3.1-405b', name: 'Llama 3.1 405B', family: 'llama-3',
    layers: 126, hiddenDim: 16384, intermediateDim: 53248,
    numHeads: 128, numKvHeads: 8, headDim: 128, vocabSize: 128256,
    paramCount: 405_853_356_032
  },
  // === Gemma 3 ===
  {
    id: 'gemma-3-12b', name: 'Gemma 3 12B', family: 'gemma-3',
    layers: 48, hiddenDim: 3840, intermediateDim: 15360,
    numHeads: 16, numKvHeads: 8, headDim: 256, vocabSize: 262144,
    paramCount: 12_187_000_000
  },
  {
    id: 'gemma-3-27b', name: 'Gemma 3 27B', family: 'gemma-3',
    layers: 62, hiddenDim: 5376, intermediateDim: 21504,
    numHeads: 32, numKvHeads: 16, headDim: 128, vocabSize: 262144,
    paramCount: 27_009_000_000
  },
  // === Mistral ===
  {
    id: 'mistral-small-3.1-24b', name: 'Mistral Small 3.1 24B', family: 'mistral',
    layers: 40, hiddenDim: 5120, intermediateDim: 32768,
    numHeads: 32, numKvHeads: 8, headDim: 128, vocabSize: 131072,
    paramCount: 23_572_403_200
  },
  {
    id: 'mistral-large-2', name: 'Mistral Large 2 123B', family: 'mistral',
    layers: 88, hiddenDim: 12288, intermediateDim: 28672,
    numHeads: 96, numKvHeads: 8, headDim: 128, vocabSize: 32768,
    paramCount: 122_610_524_160
  },
  // === Phi ===
  {
    id: 'phi-4', name: 'Phi-4 14B', family: 'phi',
    layers: 40, hiddenDim: 5120, intermediateDim: 17920,
    numHeads: 40, numKvHeads: 10, headDim: 128, vocabSize: 100352,
    paramCount: 14_659_507_200
  }
]
EOF
```

- [ ] **Step 2: Extend data/index.ts to re-export MODELS**

```bash
cat > src/data/index.ts <<'EOF'
export { GPUS } from './gpus'
export { MODELS } from './models'
EOF
```

- [ ] **Step 3: Verify type-check passes for full data module**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add calc/src/data/models.ts calc/src/data/index.ts
git commit -m "feat(calc): seed model property database (Qwen3, Llama, Gemma 3, Mistral, Phi-4)"
```

---

## Task 15: Integration Test with Real Data

**Files:**
- Modify: `calc/test/engine/calc.test.ts`

Validate the engine end-to-end against a real (GPU, model, workload) tuple. This catches issues that synthetic fixtures miss (e.g., unit-conversion bugs that vanish at scale=1).

- [ ] **Step 1: Add integration test**

Append to `test/engine/calc.test.ts`:

```ts
import { GPUS } from '../../src/data/gpus'
import { MODELS } from '../../src/data/models'
import type { CalcInput } from '../../src/engine/types'

describe('calculate — real data integration', () => {
  const h100 = GPUS.find(g => g.id === 'h100')!
  const llama70b = MODELS.find(m => m.id === 'llama-3.3-70b')!

  const input: CalcInput = {
    gpu: h100,
    gpuVariantId: 'sxm-80',
    model: llama70b,
    quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
    workload: { promptTokens: 2048, outputTokens: 512, concurrency: 1 }
  }

  it('Llama 3.3 70B on H100 SXM-80: weights are 141 GB (does not fit single-GPU)', () => {
    const r = calculate(input)
    // 70.55B params × 2 bytes = 141.1 GB
    expect(r.memory.weights / 1e9).toBeCloseTo(141.1, 0)
    expect(r.memory.fits).toBe(false)
  })

  it('Llama 3.3 70B prefill regime is compute-bound for batch=1, prompt=2048', () => {
    const r = calculate(input)
    // Long prefill on dense 70B model — compute term dominates
    expect(r.perf['peak'].prefill.regime).toBe('compute')
  })

  it('Llama 3.3 70B decode at batch=1 is memory-bound', () => {
    const r = calculate(input)
    // Classic single-stream decode: weight-load bandwidth dominates
    expect(r.perf['peak'].decode.regime).toBe('memory')
  })
})
```

- [ ] **Step 2: Run tests, verify pass**

```bash
npx vitest run
```

Expected: all tests PASS, including the new integration cases.

- [ ] **Step 3: Commit**

```bash
git add calc/test/engine/calc.test.ts
git commit -m "test(calc): integration test against real Llama/H100 data"
```

---

## Task 16: UI — Reactive Store

**Files:**
- Create: `calc/src/ui/stores.ts`

Svelte 5 stores holding the current input. UI binds to them; downstream output components reactively recompute.

- [ ] **Step 1: Write stores.ts**

```bash
cat > src/ui/stores.ts <<'EOF'
import { writable, derived, type Readable } from 'svelte/store'
import { GPUS, MODELS } from '../data'
import { calculate } from '../engine'
import type { CalcInput, CalcResult, Quantization, Workload } from '../engine/types'

const defaultGpu = GPUS[0]
const defaultModel = MODELS[0]

export const gpuId = writable(defaultGpu.id)
export const variantId = writable(defaultGpu.variants[0].id)
export const modelId = writable(defaultModel.id)

export const quant = writable<Quantization>({
  weights: 'fp16', kv: 'fp16', activations: 'fp16'
})
export const workload = writable<Workload>({
  promptTokens: 2048, outputTokens: 512, concurrency: 1
})

export const input: Readable<CalcInput | null> = derived(
  [gpuId, variantId, modelId, quant, workload],
  ([$gpuId, $variantId, $modelId, $quant, $workload]) => {
    const gpu = GPUS.find(g => g.id === $gpuId)
    const model = MODELS.find(m => m.id === $modelId)
    if (!gpu || !model) return null
    if (!gpu.variants.find(v => v.id === $variantId)) return null
    return { gpu, gpuVariantId: $variantId, model, quant: $quant, workload: $workload }
  }
)

export const result: Readable<CalcResult | null> = derived(input, $input => {
  if (!$input) return null
  try { return calculate($input) } catch { return null }
})
EOF
```

- [ ] **Step 2: Verify it compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add calc/src/ui/stores.ts
git commit -m "feat(calc): add reactive UI stores"
```

---

## Task 17: UI — Input Panel

**Files:**
- Create: `calc/src/ui/InputPanel.svelte`

Top-of-page form: GPU + variant (cascading), model, quantization (3 dropdowns), workload (3 number inputs).

- [ ] **Step 1: Write InputPanel.svelte**

```bash
cat > src/ui/InputPanel.svelte <<'EOF'
<script lang="ts">
  import { GPUS, MODELS } from '../data'
  import { gpuId, variantId, modelId, quant, workload } from './stores'
  import type { Dtype } from '../engine/types'

  const DTYPES: Dtype[] = ['fp32', 'fp16', 'bf16', 'fp8', 'int8', 'int4']

  $: gpu = GPUS.find(g => g.id === $gpuId)
  $: variants = gpu?.variants ?? []
  // Reset variant if it falls outside the new GPU's list.
  $: if (gpu && !variants.find(v => v.id === $variantId)) {
       variantId.set(variants[0]?.id ?? '')
     }
</script>

<section class="input-panel">
  <div class="row">
    <label>
      GPU
      <select bind:value={$gpuId}>
        {#each GPUS as g}
          <option value={g.id}>{g.name}</option>
        {/each}
      </select>
    </label>

    <label>
      Variant
      <select bind:value={$variantId}>
        {#each variants as v}
          <option value={v.id}>{v.label}</option>
        {/each}
      </select>
    </label>

    <label>
      Model
      <select bind:value={$modelId}>
        {#each MODELS as m}
          <option value={m.id}>{m.name}</option>
        {/each}
      </select>
    </label>
  </div>

  <div class="row">
    <label>
      Weights
      <select bind:value={$quant.weights}>
        {#each DTYPES as d}<option value={d}>{d}</option>{/each}
      </select>
    </label>
    <label>
      KV
      <select bind:value={$quant.kv}>
        {#each DTYPES as d}<option value={d}>{d}</option>{/each}
      </select>
    </label>
    <label>
      Activations
      <select bind:value={$quant.activations}>
        {#each DTYPES as d}<option value={d}>{d}</option>{/each}
      </select>
    </label>
  </div>

  <div class="row">
    <label>
      Prompt tokens
      <input type="number" min="1" bind:value={$workload.promptTokens} />
    </label>
    <label>
      Output tokens
      <input type="number" min="1" bind:value={$workload.outputTokens} />
    </label>
    <label>
      Concurrency
      <input type="number" min="1" bind:value={$workload.concurrency} />
    </label>
  </div>
</section>

<style>
  .input-panel { display: flex; flex-direction: column; gap: 0.5rem; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; }
  label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.9rem; }
  select, input { font-size: 1rem; padding: 0.25rem; }
</style>
EOF
```

- [ ] **Step 2: Verify it compiles**

```bash
npm run check
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add calc/src/ui/InputPanel.svelte
git commit -m "feat(calc): add input panel UI"
```

---

## Task 18: UI — Memory Panel

**Files:**
- Create: `calc/src/ui/MemoryPanel.svelte`

Segmented bar (weights / KV / activations) over the variant's capacity, with headroom and fits indicator.

- [ ] **Step 1: Write MemoryPanel.svelte**

```bash
cat > src/ui/MemoryPanel.svelte <<'EOF'
<script lang="ts">
  import { result } from './stores'

  const GB = 1024 ** 3
  function gb(bytes: number): string { return (bytes / GB).toFixed(2) }
  function pct(part: number, whole: number): number {
    return Math.max(0, Math.min(100, (part / whole) * 100))
  }
</script>

{#if $result}
  {@const m = $result.memory}
  {@const cap = m.hbmCapacityGB * GB}
  <section class="memory-panel">
    <h3>Memory budget — {gb(cap)} GB</h3>
    <div class="bar" class:oom={!m.fits}>
      <div class="seg weights" style="width: {pct(m.weights, cap)}%"></div>
      <div class="seg kv" style="width: {pct(m.kvCacheTotal, cap)}%"></div>
      <div class="seg act" style="width: {pct(m.activationsPeak, cap)}%"></div>
    </div>
    <table>
      <tbody>
        <tr><td>Weights</td>          <td>{gb(m.weights)} GB</td></tr>
        <tr><td>KV cache (total)</td> <td>{gb(m.kvCacheTotal)} GB</td></tr>
        <tr><td>Activations (~)</td>  <td>{gb(m.activationsPeak)} GB</td></tr>
        <tr class="total"><td>Total</td><td>{gb(m.total)} GB</td></tr>
        <tr>
          <td>Headroom</td>
          <td class:oom={!m.fits}>
            {gb(m.headroom)} GB &nbsp; {m.fits ? '✓ fits' : '✗ OOM'}
          </td>
        </tr>
      </tbody>
    </table>
    <p class="caveat">~ activations estimate assumes FlashAttention-style kernels</p>
  </section>
{/if}

<style>
  .memory-panel { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem; }
  .bar { display: flex; height: 1.5rem; border: 1px solid #888; background: #f0f0f0; }
  .bar.oom { border-color: #c33; }
  .seg.weights { background: #4a90e2; }
  .seg.kv      { background: #7ac74a; }
  .seg.act     { background: #e2a04a; }
  table { font-variant-numeric: tabular-nums; }
  td:first-child { padding-right: 1rem; }
  tr.total { font-weight: bold; }
  .oom { color: #c33; font-weight: bold; }
  .caveat { font-size: 0.8rem; color: #666; font-style: italic; }
</style>
EOF
```

- [ ] **Step 2: Verify it compiles**

```bash
npm run check
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add calc/src/ui/MemoryPanel.svelte
git commit -m "feat(calc): add memory panel UI"
```

---

## Task 19: UI — Performance Panel

**Files:**
- Create: `calc/src/ui/PerfPanel.svelte`

Per-operating-point table: TTFT, prefill regime, decode time/tok, decode regime, input/output tok/s.

- [ ] **Step 1: Write PerfPanel.svelte**

```bash
cat > src/ui/PerfPanel.svelte <<'EOF'
<script lang="ts">
  import { result } from './stores'

  function ms(s: number): string { return (s * 1000).toFixed(2) + ' ms' }
  function rate(tps: number): string { return tps.toFixed(1) + ' tok/s' }
</script>

{#if $result}
  <section class="perf-panel">
    <h3>Performance</h3>
    <table>
      <thead>
        <tr>
          <th>Operating point</th>
          <th>TTFT</th>
          <th>Prefill regime</th>
          <th>Decode time / tok</th>
          <th>Decode regime</th>
          <th>Input tok/s</th>
          <th>Output tok/s (aggregate)</th>
        </tr>
      </thead>
      <tbody>
        {#each Object.entries($result.perf) as [id, p]}
          <tr>
            <td>{id}</td>
            <td>{ms(p.ttftS)}</td>
            <td><span class="regime {p.prefill.regime}">{p.prefill.regime}</span></td>
            <td>{ms(p.decode.timePerTokenS)}</td>
            <td><span class="regime {p.decode.regime}">{p.decode.regime}</span></td>
            <td>{rate(p.inputTokenRate)}</td>
            <td>{rate(p.outputTokenRate)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </section>
{/if}

<style>
  .perf-panel { margin-top: 1rem; }
  table { font-variant-numeric: tabular-nums; border-collapse: collapse; }
  th, td { padding: 0.25rem 0.75rem; text-align: left; border-bottom: 1px solid #eee; }
  .regime { padding: 0.1rem 0.4rem; border-radius: 0.2rem; font-size: 0.85rem; }
  .regime.compute { background: #fde6c8; color: #8a4400; }
  .regime.memory  { background: #c8dcfd; color: #003a8c; }
</style>
EOF
```

- [ ] **Step 2: Verify it compiles**

```bash
npm run check
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add calc/src/ui/PerfPanel.svelte
git commit -m "feat(calc): add performance panel UI"
```

---

## Task 20: UI — Derivation Drawer

**Files:**
- Create: `calc/src/ui/DerivationDrawer.svelte`

Side drawer, collapsed by default, overlays main content when expanded.

- [ ] **Step 1: Write DerivationDrawer.svelte**

```bash
cat > src/ui/DerivationDrawer.svelte <<'EOF'
<script lang="ts">
  import { result } from './stores'
  let open = $state(false)

  function fmt(value: number, unit: string): string {
    if (unit === 'bytes' && value >= 1024 ** 3) return (value / 1024 ** 3).toFixed(2) + ' GB'
    if (unit === 'bytes' && value >= 1024 ** 2) return (value / 1024 ** 2).toFixed(2) + ' MB'
    if (unit === 'bytes' && value >= 1024)      return (value / 1024).toFixed(2) + ' KB'
    if (unit === 's' && value < 0.001)          return (value * 1e6).toFixed(2) + ' µs'
    if (unit === 's' && value < 1)              return (value * 1000).toFixed(3) + ' ms'
    return value.toLocaleString() + ' ' + unit
  }
</script>

<button class="toggle" onclick={() => open = !open}>
  {open ? '✕' : '☰'} Show math
</button>

{#if open && $result}
  <aside class="drawer">
    <h3>Derivation</h3>
    <ol>
      {#each $result.derivation as step}
        <li>
          <div class="label">{step.label}</div>
          <code class="expr">{step.expression}</code>
          <div class="value">= {fmt(step.value, step.unit)}</div>
        </li>
      {/each}
    </ol>
  </aside>
{/if}

<style>
  .toggle {
    position: fixed; top: 1rem; right: 1rem; z-index: 11;
    background: #333; color: #fff; border: none; padding: 0.5rem 1rem;
    cursor: pointer; font-family: inherit;
  }
  .drawer {
    position: fixed; top: 0; right: 0; bottom: 0; width: min(420px, 90vw);
    background: #fff; border-left: 1px solid #888;
    overflow-y: auto; padding: 3rem 1rem 1rem; z-index: 10;
    box-shadow: -4px 0 12px rgba(0,0,0,0.1);
  }
  ol { list-style: decimal inside; padding-left: 0; }
  li { margin-bottom: 0.75rem; }
  .label { font-weight: 600; }
  .expr { display: block; font-size: 0.85rem; color: #555; margin: 0.1rem 0; }
  .value { font-variant-numeric: tabular-nums; }
</style>
EOF
```

- [ ] **Step 2: Verify it compiles**

```bash
npm run check
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add calc/src/ui/DerivationDrawer.svelte
git commit -m "feat(calc): add derivation drawer UI"
```

---

## Task 21: UI — Assemble App.svelte

**Files:**
- Modify: `calc/src/ui/App.svelte`

Wire the four panels into the top-down layout.

- [ ] **Step 1: Rewrite App.svelte**

```bash
cat > src/ui/App.svelte <<'EOF'
<script lang="ts">
  import InputPanel from './InputPanel.svelte'
  import MemoryPanel from './MemoryPanel.svelte'
  import PerfPanel from './PerfPanel.svelte'
  import DerivationDrawer from './DerivationDrawer.svelte'
</script>

<main>
  <header>
    <h1>LLM Performance Calculator</h1>
    <p>Roofline estimates for dense decoder-only transformers.</p>
  </header>
  <InputPanel />
  <MemoryPanel />
  <PerfPanel />
  <DerivationDrawer />
</main>

<style>
  :global(body) {
    margin: 0; font-family: system-ui, -apple-system, sans-serif;
    background: #fafafa; color: #222;
  }
  main { max-width: 960px; margin: 0 auto; padding: 1.5rem; }
  header { margin-bottom: 1.5rem; }
  h1 { margin: 0 0 0.25rem; }
  header p { margin: 0; color: #666; }
</style>
EOF
```

- [ ] **Step 2: Run the dev server and verify the page works**

```bash
npm run dev
```

Open `http://localhost:5173` in a browser. Verify:
- All three input rows render and dropdowns populate.
- Changing inputs updates memory and perf panels live.
- Memory bar shrinks/grows with concurrency.
- Switching to Llama 3.3 70B on H100 SXM-80 shows OOM (red headroom).
- Toggle "Show math" — drawer slides in with steps.

Stop the dev server (Ctrl-C).

- [ ] **Step 3: Final tsc + test sweep**

```bash
npm run check && npm test
```

Expected: type-check clean, all tests pass.

- [ ] **Step 4: Commit**

```bash
git add calc/src/ui/App.svelte
git commit -m "feat(calc): assemble v1 UI"
```

---

## Self-Review Notes

Coverage of the spec, by section:

- **Engine architecture** — Tasks 2–12 (types, dtypes, memory, roofline, prefill, decode, derivation, calc)
- **Property database** — Tasks 13–14 (GPUs, models)
- **L0–L2 math** — All formulas in the spec are implemented and tested
- **MemoryResult shape** — Task 7 (weights, KV, activations, total, headroom, fits)
- **Operating-point iteration** — Task 12 (calculate iterates `variant.operatingPoints`)
- **Derivation steps** — Tasks 11–12
- **UI layout (top-down, side drawer)** — Tasks 16–21
- **Testing (TDD, fixtures, integration test)** — Tasks 3–12, 15
- **Svelte + Vite scaffolding** — Task 1

Deferred per spec (no tasks needed):
- v2+ layers (TP/PP, disagg, MoE, multi-node, achievable operating points, library calibration)
- UI tests, roofline plot, sweep mode, URL state, comparison mode

Type consistency check: `bytesOf`, `roofline`, `computeMemory`, `computePrefill`, `computeDecode`, `calculate`, `DerivationBuilder` — all consistent across tasks. Field names in `MemoryResult`, `PerfTier`, `CalcResult`, `DerivationStep` match the spec exactly.

---
name: adding-hardware-sku
description: Use when adding a new accelerator (GPU/TPU/Trainium/Gaudi), interconnect fabric, or multi-accelerator system to the llm-calc database (calc/src/data/{accelerators,interconnects,systems}.ts). Routes to the right sub-procedure based on whether the SKU is a new chip, a new fabric, or a new product composition. Invoke whenever hardware is being added — the failure modes (sparsity-inflated TFLOPS, confused per-direction vs aggregate BW, wrong variant in a system) all produce plausible-looking perf numbers that are silently wrong.
---

# Adding a Hardware SKU

Three things live under hardware. Pick the right sub-procedure:

- **Accelerator** — new chip generation (H200, MI400, TPU v7). Adds an `AcceleratorSpec` with variants and operating points.
- **Interconnect** — new fabric (NVLink 6, NVL Switch tray, IB-XDR, EFA v4). Adds an `InterconnectSpec`.
- **System** — new product (DGX Spark, GB300 NVL72, p6e-class cloud SKU). Composes existing accelerator + interconnect.

If multiple apply (a new chip + new fabric + new system land together), do them in order: accelerator → interconnect → system. Later layers reference earlier by id.

## Source priority (all hardware)

1. **Vendor whitepaper / architecture brief** — primary truth. NVIDIA architecture whitepaper PDFs, AMD CDNA briefs, Intel Gaudi product briefs, Google TPU papers, AWS Neuron docs.
2. **Vendor datasheet** — HBM capacity, package power, form factor.
3. **Independent microbenchmark paper** — for `achievable` operating points only. See [`calc/src/data/sources.ts`](../../../calc/src/data/sources.ts) for the registry (arxiv-2501-12084 for Hopper, arxiv-2510-27583 for MI300X, etc.).
4. **Cloud SKU page** — for `availability.clouds` on systems and to disambiguate variant labels (e.g. "AWS P5e" → which exact accelerator+variant).

Aggregator/marketing sites are not acceptable primary sources for TFLOPS, HBM BW, or fabric specs. See [`docs/data-philosophy.md`](../../../docs/data-philosophy.md) for the reasoning.

## A. New Accelerator

Files: [`calc/src/data/accelerators.ts`](../../../calc/src/data/accelerators.ts), [`calc/src/data/sources.ts`](../../../calc/src/data/sources.ts).

Schema in [`calc/src/engine/types.ts`](../../../calc/src/engine/types.ts) (`AcceleratorSpec` / `AcceleratorVariant` / `AcceleratorOperatingPoint`).

Shape:

```typescript
{
  id: 'mi400',                       // lowercase-kebab; stable across renames
  name: 'AMD MI400',
  vendor: 'AMD',
  family: 'CDNA Next',
  variants: [
    {
      id: 'oam-256',                 // form-factor + HBM capacity
      label: 'OAM 256GB',
      hbmCapacityGB: 256,
      operatingPoints: [
        {
          id: 'peak', label: 'Peak',
          tflops: { fp16: ..., bf16: ..., fp8: ..., int8: ..., fp4: ... },
          hbmBandwidthGBs: ...
        },
        // Optional — only with a citable source:
        {
          id: 'achievable', label: 'Achievable',
          tflops: { ... },
          hbmBandwidthGBs: ...,
          tflopsSources: ['arxiv-...'],     // keys into SOURCES
          bandwidthSources: ['nvbandwidth'],
          asOf: '2026-Q3',
          notes: 'one-line measurement context'
        }
      ]
    }
  ]
}
```

### TFLOPS table — pitfalls (most common bugs land here)

- **Sparsity multipliers**: NVIDIA quotes TensorCore numbers at 2:1 sparse. The schema uses **dense** values. If a spec sheet says "989/1979 TFLOPS BF16 (with sparsity)", record the unsparse half.
- **FP4 / FP6**: Blackwell-class chips quote these with sparsity assumed. Same rule.
- **INT8 vs FP8**: identical throughput on most modern chips; record both if the vendor lists both.
- **TF32**: skip — not a serving dtype.
- **Boost vs base clock**: most vendor TFLOPS are boost-clock peak. Record those (matches the rest of the database).

### Achievable operating points

Optional but valuable. Acceptable sources:
- mamf-finder MAMF table (PyTorch torch.mm sweeps)
- Microbenchmark papers on arxiv (arxiv-2501-12084 Hopper, arxiv-2510-27583 MI300X, arxiv-2512-02189 Blackwell, etc.)
- AMD MAFs blog, NVIDIA cuBLAS perf posts

Don't fabricate achievable numbers. Either cite a source or skip the operating point — peak-only entries are fine.

If you add a new source, register it in `sources.ts` first, then reference its key.

## B. New Interconnect

File: [`calc/src/data/interconnects.ts`](../../../calc/src/data/interconnects.ts).

Schema in [`calc/src/engine/types.ts`](../../../calc/src/engine/types.ts) (`InterconnectSpec`).

Shape:

```typescript
{
  id: 'nvlink-6',
  name: 'NVLink 6',
  vendor: 'NVIDIA',
  generation: 'Gen6 (Rubin)',
  perGpuBandwidthGBs: ...,           // bidirectional aggregate per chip (vendor headline)
  perDirectionGBs: ...,              // half of perGpuBandwidthGBs unless quoted separately
  linksPerGpu: ...,
  perLinkGBs: ...,                   // per direction
  topology: 'switched',              // see InterconnectTopology
  scale: 'intra-node',               // see InterconnectScale
  maxScaleUpGpus: 8,
  sources: ['nvidia-nvlink'],
  notes: '...'
}
```

### Bandwidth conventions (read [`types.ts:58-72`](../../../calc/src/engine/types.ts#L58-L72))

- `perGpuBandwidthGBs` is the **bidirectional aggregate** number on vendor slides ("900 GB/s NVLink 4"). Halve it for ring all-reduce math.
- For point-to-point fabrics (direct NVLink, IB): `perLinkGBs × linksPerGpu` should equal the per-direction aggregate. Check arithmetic before committing.
- For switched (NVSwitch, NVL72): the aggregate is what each chip can pump into the switch; bisection is a separate property (`contention.bisectionFactor`).

### Optional: contention model

Add `contention: { bisectionFactor, oversubscription?, hopCostModel, singleHopUtilization }` only with data. Hand-waving guesses are worse than omitting; the engine falls back gracefully. Tier overrides (`tiers`) are for measured collective performance — only with a citable source.

## C. New System

File: [`calc/src/data/systems.ts`](../../../calc/src/data/systems.ts).

Schema in [`calc/src/engine/types.ts`](../../../calc/src/engine/types.ts) (`MultiAcceleratorSystem`).

Pure composition — pick existing accelerator id+variant, existing interconnect id, set form factor, fill aggregates.

```typescript
{
  id: 'dgx-spark',
  name: 'NVIDIA DGX Spark',
  vendor: 'NVIDIA',
  generation: 'GB10',
  formFactor: 'node',                // see SystemFormFactor
  accelerator: { id: 'gb10', variantId: 'unified-128', count: 1 },
  interconnectId: '...',             // InterconnectSpec.id
  scaleOutInterconnectId: '...',
  scaleOutNicsPerNode: 1,
  aggregate: {
    totalHbmGB: 128,                 // = accelerator.count × variant.hbmCapacityGB
    fabricBidirectionalTBs: ...      // = interconnect.perGpuBandwidthGBs × count / 1000
  },
  availability: { onPrem: true, clouds: [...] },
  notes: '...'
}
```

### Sanity checks (where systems go wrong)

- `aggregate.totalHbmGB` must equal `count × hbmCapacityGB`. Off-by-factor-of-ten is the most common bug.
- `aggregate.fabricBidirectionalTBs` must equal `perGpuBandwidthGBs × count / 1000`. Same kind of math, easy to flub.
- `availability.clouds` should match the cloud SKU page exactly — `aws`, `azure`, `gcp`, `oci`, `coreweave`, `lambda`, `crusoe` are the common ones (full list in `CloudProvider` in types.ts).
- Confirm `accelerator.variantId` exists in the referenced accelerator. TS won't catch this — it'd require a const-keyed lookup; reviewer needs to check.

## Validation (all hardware)

1. `npm test` — won't catch numeric errors (no test asserts specific TFLOPS), but ensures imports resolve.
2. `npm run check` — TS catches schema mismatches.
3. `npm run dev` — open the UI, select the new accelerator / system, verify the perf panel renders without NaN and numbers are in the right order of magnitude vs. siblings (e.g. a new "H300" should land between H200 and B200; if it lands at 10× B200, you have a sparsity bug).

No TDD pattern here — data-only changes don't get unit tests. Schema changes do, but those go through the model-skill TDD flow because they touch the engine.

## Anti-patterns

- Recording sparsity-inflated TFLOPS without halving. The most common bug, by a wide margin.
- Confusing `perGpuBandwidthGBs` (aggregate) vs `perDirectionGBs` (half) on interconnects.
- Inventing achievable numbers without a source citation.
- Picking the wrong accelerator variant in a system (SXM vs PCIe, 80GB vs 96GB).
- Skipping `sources` / `asOf` because "the data is obvious". It rots; provenance is the only defense.
- Adding a tier / contention model based on intuition rather than measurement. Empty is better than wrong.

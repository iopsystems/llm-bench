---
name: verifying-achievable-perf-numbers
description: Use whenever working with citable hardware performance numbers — TFLOPS per dtype, sustained HBM/memory bandwidth, kernel-level throughput — across any of these four contexts: (a) initially sourcing them from papers, vendor blogs, or community benchmarks; (b) editing or filling in citation metadata fields (publication dates, source slugs, methodology notes) on existing entries; (c) auditing whether a claimed perf number's citation actually holds up; (d) adding entries to a sources/citation registry. Triggers on phrases like "achievable peak", "sustained throughput", "microbenchmark", "achievable bandwidth", "max-achievable FLOPs", "TFLOPS achievable", and on any edit that writes into fields named `tflopsSources`, `bandwidthSources`, `asOf`, `sources`, or methodology `notes` on operating-point-like data. Apply this skill whenever you are about to write a date, URL, or methodology claim into a perf citation field — even if you think you remember the value — because the failure mode is inferring from adjacent knowledge instead of verifying the actual source. Research-agent summaries and "I remember roughly when that shipped" inferences are systematically wrong about hardware variants and dates; bad citations look defensible and propagate.
---

# Verifying Achievable Performance Numbers

## What this skill is for

You're trying to fill in achievable hardware performance numbers for a calculator, perf model, or planning doc — TFLOPS the silicon actually delivers under real workloads, sustained memory bandwidth from real kernels, etc. Every number gets a citation, and the citations need to hold up.

## Why this skill exists

Three failure modes burn through this kind of work if you don't watch for them:

1. **Variant confusion.** Most published Hopper microbenchmark papers test **H800 PCIe** — the China export part — and the numbers get cited as "H100 measurements." Same arch family, different SKU. Same trap with Ada (RTX 4090 vs RTX 6000 Ada vs L40S share architecture but not specs), Blackwell (B100 vs B200 vs GB200), and CDNA3 (MI300X vs MI325X vs MI300A). The advertised peak for these variants can differ by 30%+.

2. **Peak vs sustained.** Vendors report "max-achievable" under different conditions. AMD's MI300X blog gives 708 BF16 TFLOPS at 1207 MHz sustained on the 750W variant. Other sources give 890 BF16 TFLOPS at boost clocks. Both are real numbers and they mean different things. A roofline calculator needs to pick one and label it.

3. **Paraphrase decay.** Research-agent summaries of papers consistently lose per-variant precision. "The paper reports H100 SXM at 756 TFLOPS" is a thing you'll see in an agent summary — and it's wrong. The paper tested H800 PCIe; 756 happens to be the H100 PCIe datasheet number, which the agent grabbed from a comparison column.

The skill's core rule: **never cite a source you have not read the actual text of.** Paraphrases can match the methodology and still be wrong on the variant. A bad citation does worse than no citation, because it looks defensible.

## Workflow

### 1. Pin down what you're sourcing

Be precise before you search. Underspecified targets produce conflicting "facts":

- **GPU model + variant**: not "H100" — "H100 SXM5 80GB" or "H100 PCIe 80GB".
- **Operating mode**: stock TDP, boost clocks, sustained clocks, power-capped to N watts.
- **Dtype**: tensor-core FP16 vs shader FP16 vs FP16-accumulate-FP32; FP8 (E4M3 or E5M2); INT8; INT4.
- **What you're measuring**: matmul TFLOPS (GEMM), GEMV throughput, sustained HBM read, sustained HBM copy, NVLink BW, PCIe BW.

A question like "what's the achievable FP16 for H100?" cannot be answered without disambiguating the variant — at least four H100 SKUs exist with materially different numbers.

### 2. Pick candidate sources, ranked by trustworthiness

1. **Peer-reviewed microbenchmark papers** (typically arXiv). Search terms: "dissecting [architecture]", "microbenchmarking [architecture]", "[architecture] performance analysis".
2. **Vendor publications with disclosed methodology** — AMD's ROCm blog "Measuring Max-Achievable FLOPs", NVIDIA cuBLAS/CUTLASS performance blogs. They name library versions, problem sizes, clock conditions.
3. **Crowd-sourced harnesses with public result tables** — `stas00/ml-engineering` mamf-finder, `nod-ai/rocm-gemm-benchmark`.
4. **HPC center reports** (ORNL, ANL, NERSC, LBNL) — usually peer-reviewed.
5. **Technical analysis blogs** (Chips and Cheese, SemiAnalysis) — useful for cross-checks, not as primary citations.

Reject as primary sources:
- Reddit / forum / YouTube benchmarks without published methodology
- Vendor marketing slides that don't disclose conditions
- MLPerf results as a primary roofline source (downstream — useful for end-to-end sanity, not raw compute/BW)

### 3. Verify each candidate by reading the source

This is the non-negotiable part. For each candidate, before citing:

- **Fetch the actual text.** Use WebFetch on the HTML version of arXiv papers — they typically live at `arxiv.org/html/<id>` or `arxiv.org/html/<id>v<N>`. PDF parsing through WebFetch often returns garbled output; HTML is more reliable. If a paper has no HTML version, try a literature mirror or ask the user.
- **Find the explicit hardware identifier in the paper's methodology section.** Look for tables captioned "GPU specifications", "Experimental setup", or similar. The paper should explicitly name what was tested — "H800 PCIe", "MI300X 750W variant", "A100 SXM4 80GB". The abstract often glosses this over.
- **Find the specific number with its conditions.** What clock speed? Which library/version? What problem size? Papers often report multiple numbers per dtype (different problem sizes, different precisions of accumulator); pick the one most representative of the target workload (usually large-shape GEMM for LLM matmul).

If the HTML isn't available and PDF parsing fails, ask the user — don't guess. Or try a different source.

### 4. Reject mismatches honestly

If the paper measured a different variant than your target, **reject it as a direct citation**. Two paths forward:

- **Substitute with an explicit caveat in `notes`.** If the variants are silicon-equivalent (e.g., H800 PCIe ≈ H100 PCIe with restricted NVLink), it can be reasonable to use the numbers — but the notes field must say so. Example: `"Measured on H800 PCIe (same silicon as H100 PCIe, export-restricted NVLink); wgmma + global memory tests"`. Don't claim the source measured the target directly.
- **Keep searching.** The honest answer most of the time.

Substitutions that look reasonable and often aren't:
- H800 → H100 SXM (different silicon, different specs)
- B200 → B100 (different silicon configuration)
- A100 SXM → A100 PCIe (different bandwidth, sometimes different boost behavior)
- MI300X 750W → MI300X 1000W (different sustained clocks)
- "Apple M3 Ultra" → which M3 Ultra config? (GPU core counts vary by SKU)

### 5. Record provenance in the data model

For projects following the llm-calc schema (`src/data/sources.ts` + `tflopsSources` / `bandwidthSources` on operating points), each verified citation produces:

**(a) Entry in `src/data/sources.ts`:**

```ts
'<slug>': {
  title: 'Exact title from the source',
  url: 'https://...'
}
```

Slug conventions: `arxiv-YYMM-NNNNN` for papers, `<vendor>-<topic>` for vendor blogs (e.g., `amd-rocm-mafs`), `<tool>-<date>` for community harnesses.

**(b) Operating-point fields in the GPU data file:**

```ts
{
  id: 'achievable', label: 'Achievable',
  tflops: { ... },                        // actual measured numbers
  hbmBandwidthGBs: ...,                   // peak if not separately measured
  tflopsSources: ['<slug>'],              // citations for compute
  bandwidthSources: ['<slug>'],           // citations for BW; omit if not measured
  asOf: 'YYYY-MM',                        // when the measurement was published
  notes: 'Methodology + caveats in one sentence'
}
```

Use the per-axis citation fields. If only compute is sourced and bandwidth stays at peak, set `tflopsSources` and leave `bandwidthSources` undefined. The UI and CLI render this correctly — they'll label compute citations under "TFLOPS:" and not falsely imply bandwidth was measured.

For other projects: the principle is the same — record source key, what variant/conditions were measured, dates, and which axis (compute vs bandwidth) each citation backs.

### 6. Be honest about gaps

If you can't find a verified source for a particular GPU/variant, **say so** in your final summary. Do not:

- Invent a percentage and call it achievable
- Reuse another variant's number without flagging the substitution
- Cite a source you only saw paraphrased

Acceptable outcomes:
- No achievable tier for that variant (just keep peak)
- Achievable tier with `notes: "estimated derate; no verified microbenchmark source — placeholder"` so future contributors know to replace it
- Achievable tier sourced from a near-relative with the substitution called out in notes

## Common failure modes — concrete examples

- **"The Hopper paper says H100 fp16 is 729 TFLOPS."** Check the experimental-setup table. Almost certainly the paper tested H800 PCIe; you're looking at the H100 PCIe datasheet column or a number that maps to H100 PCIe via silicon equivalence.

- **"AMD says MI300X gets 890 BF16 TFLOPS."** Confirm conditions. AMD's official blog gives 708 at sustained 1207 MHz on the 750W variant. The 890 number is from a different paper at different conditions. They are not interchangeable.

- **"This RTX 4090 number is from a benchmark."** Whose benchmark? Forum and blog posts without published methodology aren't citable — they're hearsay. Either run `mamf-finder.py` yourself or skip the achievable tier.

- **"NVIDIA's cuBLAS blog reports the H100 SXM5 number."** NVIDIA's blogs frequently report *relative speedups* (X× over A100) rather than absolute TFLOPS. Re-read for absolutes; they're often not there.

- **"The paper measures 95% of peak."** Peak under which condition? Sparsity-enabled tensor-core peak? Boost-clock peak? Sustained-clock peak? "95% of peak" without naming the peak baseline is unfalsifiable.

## Output

A successful invocation produces:

1. **A short verification summary** naming, for each target:
   - What was found and verified
   - What was rejected and why
   - What gaps remain (and recommended next steps)

2. **Concrete code changes** to the project's source registry and data file (with per-axis citations populated correctly — `tflopsSources` and `bandwidthSources` separately, set only when actually measured).

3. **One line of honest self-assessment** — what would I want a second reviewer to double-check?

The goal is not to maximize citation count. The goal is that every number in the calculator can be traced to an actual sentence in an actual source — and that gaps stay gaps until they can be filled honestly.

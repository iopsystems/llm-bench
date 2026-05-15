# Data Philosophy

The "why" behind how the llm-calc database is built. The "how" (procedures, schemas, field mappings) lives in [`skills/adding-a-model`](../.claude/skills/adding-a-model/SKILL.md) and [`skills/adding-hardware-sku`](../.claude/skills/adding-hardware-sku/SKILL.md).

## Source hierarchy: vendor primary, paper authoritative for novel architecture

**Models**: HuggingFace `config.json` is the file the model actually loads from. If it's wrong, the model doesn't run — so vendors keep it accurate. Marketing materials, blog posts, and aggregator sites all have weaker enforcement loops behind their correctness.

**Hardware**: vendor whitepapers and datasheets. Independent microbenchmark papers (arxiv) are the source for *achievable* operating points — peer-reviewed measurement, not vendor marketing.

The pattern: prefer the artifact with the **strongest enforcement loop** behind its correctness.

## Why we record provenance (`sources`, `asOf`)

Specs age. NVIDIA revises driver-realized TFLOPS after a CUDA release. AMD's MI300X HBM BW estimate moved between 5.3 and 5.2 TB/s across product briefs. Without a date and a source key, no reviewer can tell whether an old entry is wrong or just stale.

The cost of recording is one line. The cost of *not* recording shows up six months later when someone asks "where did this number come from?" and the answer is "I don't know."

## Peak vs. achievable: one is marketing, one is delivered

**Peak**: what the vendor will sell you. Quoted under maximum-favorable conditions — sparsity, boost clock, ideal memory access patterns.

**Achievable**: what a well-tuned dense kernel actually hits. Lands at 30–70% of peak for FLOPS, 80–95% for HBM BW. Cited from microbenchmark sources, not estimated.

Both belong in the database; the UI lets users pick. Don't conflate them, and don't fabricate achievable numbers when no measurement exists — peak-only is honest.

## "One workload owns the fabric"

The interconnect math assumes the entire fabric is dedicated to the workload being modeled — no cross-tenant contention, no other jobs sharing NVSwitch or the IB spine. Stated once in [`calc/src/engine/types.ts`](../calc/src/engine/types.ts) under `InterconnectSpec`; not repeated elsewhere.

Real shared clusters degrade further. The calc is an **upper bound** for fabric throughput. Multi-tenant modeling is a different problem and outside this tool's scope.

## Extrapolation: OK at the edges, not in the body

**Trained context window**: calc runs past `maxContext` and the UI shows a soft warning. The math is linear in context, so extrapolation is meaningful — just less accurate. Accepted because the alternative (hard-blocking) is annoying and the user knows what they're doing.

**TFLOPS at an unsupported dtype**: not extrapolated. If a chip's `tflops` table omits FP4, the calc returns no number for FP4-on-that-chip. Don't fabricate.

The rule: extrapolate where the underlying physics is continuous (sequence length, batch size). Don't extrapolate where the vendor hasn't shipped the capability (dtype support, fabric existence, MoE variants the model doesn't have).

## TDD for models, data-import for hardware

**Models get unit tests.** The schema has 13+ fields and 8 attention variants; typos in `numKvHeads` or `intermediateDim` produce silently-wrong KV cache numbers that look plausible until the roofline disagrees with reality. A test that asserts `kvCachePerRequest` at a known prompt length catches these instantly. Worth the few minutes.

**Hardware doesn't get tests.** The schema is narrow (TFLOPS table, HBM, fabric BW); errors show up immediately as wrong order-of-magnitude perf numbers in the UI. `npm run check` covers the TS shape; eyeballing the UI catches the values. Adding a test per accelerator is busywork.

## Aggregator sites are not primary sources

TechPowerUp, comparison spec sites, and LLM-generated summaries are convenient but not authoritative. Their numbers come from one of: (a) the same vendor source we should be using directly, (b) a stale snapshot of that source, (c) someone else's secondary citation that we can't verify.

Going to the vendor source costs one extra click and removes the middleman.

**Exception: cross-check.** If a vendor whitepaper and an aggregator agree, confidence rises slightly. If they disagree, trust the vendor — and look closer, because someone made an error.

## Schema additions raise the bar

Adding a new attention variant, a new operating-point dimension, or a new architecture type is a design change, not a data update. The cost is touching `types.ts`, `memory.ts`, `prefill.ts`, `decode.ts`, plus tests for each. The cost of *not* adding a needed variant is shoehorning into an approximate one and producing silently-wrong numbers forever.

When in doubt, brainstorm with the user before extending the schema. Once a variant ships, removing it is harder than adding it cleanly the first time.

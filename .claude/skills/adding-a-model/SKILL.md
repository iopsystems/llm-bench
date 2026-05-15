---
name: adding-a-model
description: Use when adding a new model (or model family) to the llm-calc database (calc/src/data/models.ts). Walks through sourcing fields from HuggingFace config.json, reading papers for novel attention variants, writing tests first, and registering the entry. Invoke this whenever a model is being added or updated — even for a single SKU, since the failure modes (wrong KV bytes, mis-sized attention, silently broken roofline) all look fine until someone runs the calc against a real workload.
---

# Adding a Model

## Source priority

1. **HuggingFace `config.json`** — primary truth for architecture. Fetch from `https://huggingface.co/<org>/<repo>/raw/main/config.json`. This is the file the model actually loads from; mismatches break inference, so vendors keep it accurate.
2. **HF model card README** — for active parameter count on MoE, family/release notes, anything not in config.
3. **Paper / technical report** — required for any novel attention variant or sparsity scheme (MLA, DSA, CSA, linear attention, delta-net). config.json names the fields but doesn't tell you what the geometry means.
4. **Vendor blog / launch post** — supplementary; verify any numeric claim against config.json before trusting.

Aggregators (TechPowerUp, marketing comparison sites, LLM-summarized model cards) are not acceptable primary sources. Use them only to cross-check after vendor sources agree. See [`docs/data-philosophy.md`](../../../docs/data-philosophy.md) for the reasoning.

## ModelArch field mapping

Schema in [`calc/src/engine/types.ts`](../../../calc/src/engine/types.ts) (`ModelArch`).

`config.json` key → `ModelArch` field:

| HF key | Field | Notes |
|---|---|---|
| `num_hidden_layers` | `layers` | |
| `hidden_size` | `hiddenDim` | |
| `intermediate_size` | `intermediateDim` | For MoE: **per-expert** FFN inner dim (small, e.g. 768), not dense-FFN width |
| `num_attention_heads` | `numHeads` | |
| `num_key_value_heads` | `numKvHeads` | Defaults to `numHeads` if absent (no GQA) |
| `head_dim` | `headDim` | If absent, compute `hidden_size / num_attention_heads` |
| `vocab_size` | `vocabSize` | |
| `max_position_embeddings` | `maxContext` | Trained ceiling; calc extrapolates past it with a soft warning |
| `num_nextn_predict_layers` | `numNextnLayers` | Multi-Token Prediction depth; 0 for non-MTP models |

`paramCount` — take from the model card, not derived. Vendors quote total parameters including embeddings and norms; recompute is unreliable.

## Architecture (dense vs MoE)

Detect MoE by presence of `num_experts` / `num_local_experts` in config:

```typescript
architecture: {
  type: 'moe',
  numExperts: <num_experts>,
  numExpertsActive: <num_experts_per_tok>,
  numSharedExperts: <num_shared_experts ?? 0>,
  activeParamCount: <from model card>  // routed-active + shared aggregate
}
```

Otherwise: `architecture: { type: 'dense' }`.

## Attention variant — when to use which

Walk through these in order. First match wins.

- **`full`** — default; MHA or GQA. config.json has no special attention keys beyond the standard ones.
- **`sliding`** — every layer uses a sliding window (Mistral 7B, Mixtral). Config has `sliding_window` and applies it globally.
- **`hybrid`** — interleaved sliding + global (Gemma 3, Phi-4 with sliding pattern). Config has `sliding_window_pattern` or the paper specifies a per-layer pattern.
- **`mla`** — DeepSeek V2/V3 base. Config has `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`. Read the V2 paper if you haven't.
- **`mla-dsa`** — DeepSeek V3.2. MLA fields + sparse-attention indexer. Adds `topK`.
- **`linear-mla-hybrid`** — Kimi-Linear (linear attention + MLA). Per-layer counts must sum to `model.layers`.
- **`csa-hca-hybrid`** — DeepSeek V4-Flash/Pro. Sliding + CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention).
- **`delta-hybrid`** — Qwen3.5 (Gated DeltaNet + Gated Attention).

If the model fits none of these, **stop**. Don't shoehorn into the closest variant — add a new variant to `AttentionConfig` in `types.ts` with engine integration in `memory.ts`, `prefill.ts`, `decode.ts`. Brainstorm with the user first; new attention is a meaningful design change, not a data update.

## Process

1. Pull `config.json`. Pull model card. Note `paramCount` and `activeParamCount` (if MoE).
2. Identify attention variant. Read the paper if it's not vanilla.
3. Find the **most similar existing entry** in [`calc/src/data/models.ts`](../../../calc/src/data/models.ts) (same family, same attention variant). Copy it as a template — saves more time than typing from scratch and inherits proven shape.
4. **Write a test first** in [`calc/test/engine/calc.test.ts`](../../../calc/test/engine/calc.test.ts). Pick one or two assertions that catch field mis-entry:
   - `paramCount` within a few % of expected (compute layer params manually — embeddings + per-layer attention+FFN — and check)
   - `kvCachePerRequest` at a known prompt length (verifies attention variant fields)
   - For MoE: `activeParamCount` aligns with model card
5. Run the test → red.
6. Add the entry to the `MODELS` array. Run tests → green.
7. `npm run check` (svelte-check / TS).
8. `npm run dev`, click the model in the UI, verify the perf panel renders without NaN and the memory bar looks sane against its siblings.
9. Commit.

## Reading novel attention papers

WebFetch struggles with PDFs. Two options:
- Ask the user to paste the relevant section inline.
- Dispatch a research subagent with the paper URL and a focused question: "extract per-layer attention dims, KV state size, compression ratio M, indexer head count" — list exactly the fields you need.

Don't guess. A misread MLA paper produces silently-wrong KV cache numbers for the rest of the model's life in the database.

## Anti-patterns

- Trusting aggregator sites or LLM-generated summaries for architecture numbers.
- Deriving `paramCount` from a formula instead of taking it from the model card.
- Setting `numNextnLayers: 0` reflexively; DeepSeek V3 onwards has it.
- Adding the entry before the test — the test is what catches typos in `numKvHeads` or `intermediateDim`.
- Picking the "closest" attention variant when the model is actually novel. Either it matches exactly or it gets a new variant.
- Skipping the UI check; a NaN in the perf panel is the fastest signal something is wrong.

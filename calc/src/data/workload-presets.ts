// Public-benchmark workload presets surfaced in the Calc/Sim Workload picker.
// Each preset carries sourced median (promptTokens, outputTokens) values so the
// user can pick "HumanEval" instead of hand-entering numbers. Values are
// tokenized against the Llama-3 reference tokenizer; assume ±10–20% variance
// on other tokenizers.

export interface WorkloadPreset {
  id: string                    // slug; URL-safe; must be unique within the registry
  name: string                  // display name in the dropdown
  group: 'code-gen' | 'other'   // for <optgroup> rendering
  promptTokens: number          // sourced median, positive integer
  outputTokens: number          // sourced median, positive integer
  sourceUrl: string             // citation URL (HF dataset card or canonical paper)
  sourceAccessedAt: string      // YYYY-MM-DD when the source was fetched
  description: string           // ≤100 chars; used as <option title>
}

// Pure helper — exported for testing. Returns the id of the preset whose
// promptTokens AND outputTokens both exactly match the provided workload,
// else the sentinel string 'custom'. The picker's reactive selection uses
// this. Note: return type is `string` rather than `WorkloadPreset['id'] |
// 'custom'` because TS can't narrow a runtime-data array's ids without an
// `as const` trick. Callers must treat any non-'custom' return as a live
// registry id — never free text.
export function matchPreset(
  workload: { promptTokens: number; outputTokens: number },
  presets: WorkloadPreset[]
): string {
  const m = presets.find(
    p => p.promptTokens === workload.promptTokens
      && p.outputTokens === workload.outputTokens
  )
  return m?.id ?? 'custom'
}

export const WORKLOAD_PRESETS: WorkloadPreset[] = [
  // HumanEval values from "Towards Green AI" (arXiv 2602.05712) Tables 1 & 3,
  // CodeLlama-7B zero-shot row — mean prompt and mean generated tokens. Using
  // CodeLlama (Llama-family BPE) as the closest publicly-tabulated proxy for
  // Llama-3 tokenization; within the ±10–20% tokenizer-variance disclaimer.
  // Output 207 is shaped by the canonical max_new_tokens=300 cap, which is
  // the realistic served-request distribution (not raw canonical solution).
  {
    id: 'humaneval',
    name: 'HumanEval (0-shot)',
    group: 'code-gen',
    promptTokens: 163,
    outputTokens: 207,
    sourceUrl: 'https://arxiv.org/abs/2602.05712',
    sourceAccessedAt: '2026-06-08',
    description: '164 Python problems; zero-shot, output median reflects CodeLlama-7B + max_new_tokens=300 harness',
  },
  // Sarathi-Serve (arXiv 2403.02310) Table 2 medians for openchat_sharegpt4,
  // the collated ShareGPT trace serving papers benchmark against. Multi-turn
  // chat as actually served (prompt includes prior turns), so values run
  // higher than a first-turn slice. P50 prompt / P50 output as published.
  {
    id: 'chat-typical',
    name: 'Chat (ShareGPT median)',
    group: 'other',
    promptTokens: 1730,
    outputTokens: 415,
    sourceUrl: 'https://arxiv.org/abs/2403.02310',
    sourceAccessedAt: '2026-06-08',
    description: 'ShareGPT served-chat median per Sarathi-Serve Table 2 (openchat_sharegpt4, P50)',
  },
  // LongBench gov_report: HF card reports avg 8,734 *words* (Python split,
  // not tokens — deliberate to avoid tokenizer drift). Llama-3 BPE on English
  // prose ≈ 1.35 tokens/word → ≈11,790 prompt tokens. Output = 512 per the
  // repo's LongBench/config/dataset2maxlen.json cap, which is the realistic
  // served distribution since summarization saturates the cap.
  {
    id: 'longbench-gov-report',
    name: 'LongBench (gov_report summary)',
    group: 'other',
    promptTokens: 11790,
    outputTokens: 512,
    sourceUrl: 'https://huggingface.co/datasets/THUDM/LongBench',
    sourceAccessedAt: '2026-06-08',
    description: 'Government-report summarization subtask; long prompt, 512-tok output cap',
  },
]

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
  // MBPP: CASTILLO (arXiv 2505.16881) Table 1 reports median 131 / mean 153.5
  // prompt tokens under Llama-3.2-1B tokenization — closest publicly-tabulated
  // proxy for Llama-3 BPE. Output: MBPP paper reports mean 6.8 LoC per solution
  // (~100 chars / ~30 tokens raw), but served outputs include reasoning/
  // commentary; ~120 tokens is the typical served distribution.
  {
    id: 'mbpp',
    name: 'MBPP',
    group: 'code-gen',
    promptTokens: 131,
    outputTokens: 120,
    sourceUrl: 'https://arxiv.org/abs/2505.16881',
    sourceAccessedAt: '2026-06-08',
    description: '974 Python problems; prompt median per CASTILLO Table 1, output approx from MBPP 6.8 LoC',
  },
  // LiveCodeBench code_generation_lite: LeetCode/AtCoder/Codeforces problems
  // with description + I/O examples + constraints + starter code. Problem
  // statements run ~500-1500 tokens; with format scaffolding the prompt sits
  // around 1.5k. Output: LCB default max_tokens=2000 (lcb_runner parser.py);
  // realistic solutions saturate near ~800 tokens of code+reasoning.
  {
    id: 'livecodebench',
    name: 'LiveCodeBench (code_gen_lite)',
    group: 'code-gen',
    promptTokens: 1500,
    outputTokens: 800,
    sourceUrl: 'https://huggingface.co/datasets/livecodebench/code_generation_lite',
    sourceAccessedAt: '2026-06-08',
    description: 'code_generation_lite subset; approximate — prompt incl. starter code, output near 2k cap',
  },
  // SWE-Bench Verified (Oracle retrieval): HF SWE-bench_oracle text field is
  // 2.59k–1.94M chars; SWE-bench paper §5.1 example clocks 20,882 tokens for
  // one instance. Mean for Verified (curated 500-instance subset) sits around
  // 12k tokens — Oracle context is the retrieved-file slice, not the full
  // codebase. Patch field 277-17.4k chars → median ≈1000 output tokens.
  {
    id: 'swe-bench-verified',
    name: 'SWE-Bench Verified (Oracle)',
    group: 'code-gen',
    promptTokens: 12000,
    outputTokens: 1000,
    sourceUrl: 'https://huggingface.co/datasets/princeton-nlp/SWE-bench_oracle',
    sourceAccessedAt: '2026-06-08',
    description: 'Oracle-retrieval setting; approximate — HF char range + paper §5.1 reference point',
  },
  // MMLU: HF cais/mmlu card reports question length 41–243 chars. Each item
  // is question + 4 options (A/B/C/D) + brief header. Per-item zero-shot
  // prompt: ~140 chars question + ~120 chars options + header ≈ ~50 tokens
  // (char-count ÷ 4 BPE rule-of-thumb). Output: single letter answer with
  // minimal framing → ~3 tokens.
  {
    id: 'mmlu',
    name: 'MMLU (0-shot, per item)',
    group: 'other',
    promptTokens: 50,
    outputTokens: 3,
    sourceUrl: 'https://huggingface.co/datasets/cais/mmlu',
    sourceAccessedAt: '2026-06-08',
    description: 'Zero-shot per-question; approximate — HF char range ÷ 4 BPE rule-of-thumb, single-letter answer',
  },
  // AlpacaEval: 805 single-turn instructions from helpful_base + koala. No
  // tokenized stats on HF card, but CASTILLO Table 1 reports Alpaca median 49
  // prompt tokens (Llama-3.2-1B); AlpacaEval is a curated subset of similar
  // distribution. Output: AlpacaEval lifts max_new_tokens from 300 → 2000;
  // typical judged responses run ~200 tokens (length-bias analysis baseline).
  {
    id: 'alpaca-eval',
    name: 'AlpacaEval',
    group: 'other',
    promptTokens: 50,
    outputTokens: 200,
    sourceUrl: 'https://github.com/tatsu-lab/alpaca_eval',
    sourceAccessedAt: '2026-06-08',
    description: '805 single-turn instructions; approximate — prompt from CASTILLO Alpaca median, output typical',
  },
]

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

export const WORKLOAD_PRESETS: WorkloadPreset[] = []

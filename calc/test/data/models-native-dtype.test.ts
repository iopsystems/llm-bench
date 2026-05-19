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

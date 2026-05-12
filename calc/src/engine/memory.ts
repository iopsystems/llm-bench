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

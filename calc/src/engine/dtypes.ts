import type { Dtype } from './types'

const DTYPE_BYTES: Record<Dtype, number> = {
  fp32: 4, fp16: 2, bf16: 2, fp8: 1, int8: 1, int4: 0.5
}

export function bytesOf(dtype: Dtype): number {
  return DTYPE_BYTES[dtype]
}

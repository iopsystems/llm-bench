export interface RooflineInput {
  flops: number
  bytes: number
  tflops: number           // peak compute in TFLOPs (10^12 FLOP/s)
  bwGBs: number            // peak bandwidth in GB/s (10^9 B/s)
  commsBytes?: number      // undefined for single-accelerator calls
  interconnectBwGBs?: number  // required if commsBytes is set
}

export interface RooflineResult {
  timeS: number
  regime: 'compute' | 'memory' | 'comms'
}

export function roofline({ flops, bytes, tflops, bwGBs }: RooflineInput): RooflineResult {
  const computeS = flops / (tflops * 1e12)
  const memoryS = bytes / (bwGBs * 1e9)
  if (computeS > memoryS) return { timeS: computeS, regime: 'compute' }
  return { timeS: memoryS, regime: 'memory' }
}

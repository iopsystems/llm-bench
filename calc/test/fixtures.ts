import type { AcceleratorSpec, ModelArch, Quantization, Workload, CalcInput } from '../src/engine/types'

// Tiny synthetic accelerator: 1 TFLOP fp16, 1 GB/s HBM, 1 GB capacity.
// Numbers chosen so arithmetic is exact and hand-verifiable.
export const testAccelerator: AcceleratorSpec = {
  id: 'test-accel',
  name: 'Test Accelerator',
  vendor: 'test',
  variants: [{
    id: 'v',
    label: 'V',
    hbmCapacityGB: 1,
    operatingPoints: [{
      id: 'peak',
      label: 'Peak',
      tflops: { fp16: 1 },
      hbmBandwidthGBs: 1
    }]
  }]
}

// Tiny synthetic model:
//   2 layers, hidden=4, intermediate=8, heads=2, kv_heads=1, head_dim=2
//   vocab=100, paramCount=1000
export const testModel: ModelArch = {
  id: 'test-model',
  name: 'Test Model',
  family: 'test',
  layers: 2,
  hiddenDim: 4,
  intermediateDim: 8,
  numHeads: 2,
  numKvHeads: 1,
  headDim: 2,
  vocabSize: 100,
  paramCount: 1000,
  attention: { type: 'full' },
  architecture: { type: 'dense' }
}

export const fp16Quant: Quantization = {
  weights: 'fp16', kv: 'fp16', activations: 'fp16'
}

export const testWorkload: Workload = {
  promptTokens: 10, outputTokens: 5, concurrency: 2
}

export const testInput: CalcInput = {
  accelerator: testAccelerator,
  acceleratorVariantId: 'v',
  model: testModel,
  quant: fp16Quant,
  workload: testWorkload
}

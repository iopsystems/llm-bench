/**
 * What-if: chips connected only over RDMA/RoCE (no NVLink scale-up).
 *
 * Replaces each 8-chip system's scale-up fabric with a slow scale-out fabric
 * (IB-HDR / IB-NDR / IB-XDR / AWS EFA v3) and reports when decode flips
 * comms-bound.
 *
 * Run: npx tsx scripts/rdma-decode-survey.ts
 */

import { ACCELERATORS } from '../src/data/accelerators'
import { MODELS } from '../src/data/models'
import { SYSTEMS } from '../src/data/systems'
import { INTERCONNECTS } from '../src/data/interconnects'
import { defaultParallelism } from '../src/engine/parallelism'
import { calculate } from '../src/engine/calc'
import type { CalcInput, MultiAcceleratorSystem, Quantization, Workload } from '../src/engine/types'

const QUANT: Quantization = { weights: 'fp16', kv: 'fp16', activations: 'fp16' }

const WORKLOADS: Workload[] = [
  { promptTokens: 1024, outputTokens: 256, concurrency: 1   },
  { promptTokens: 1024, outputTokens: 256, concurrency: 32  },
  { promptTokens: 1024, outputTokens: 256, concurrency: 256 },
  { promptTokens: 1024, outputTokens: 256, concurrency: 1024 },
]

// Fabrics to test — order from slow to fast, finishing with the original scale-up.
const FABRICS = ['ib-hdr', 'ib-ndr', 'ib-xdr', 'aws-efa-v3']

const baseSystem = SYSTEMS.find(s => s.id === 'hgx-h100-8')!
const h100 = ACCELERATORS.find(a => a.id === 'h100')!

interface Row {
  fabric: string
  fabricGBs: number
  model: string
  concurrency: number
  prefillRegime: string
  decodeRegime: string
  decodeBytesGB: number
  decodeCommsGB: number
}

const rows: Row[] = []

for (const fabricId of FABRICS) {
  const fabric = INTERCONNECTS.find(i => i.id === fabricId)
  if (!fabric) continue
  const fabricBW = fabric.perDirectionGBs ?? fabric.perGpuBandwidthGBs / 2

  // Build a hypothetical system: HGX H100 chassis but with the slow fabric.
  const hypoSystem: MultiAcceleratorSystem = {
    ...baseSystem,
    id: `hgx-h100-8-${fabricId}`,
    name: `HGX H100 8× via ${fabric.name}`,
    interconnectId: fabricId,
  }

  for (const model of MODELS) {
    for (const workload of WORKLOADS) {
      const pc = defaultParallelism(hypoSystem, model)
      const input: CalcInput = {
        accelerator: h100,
        acceleratorVariantId: 'sxm-80',
        model,
        quant: QUANT,
        workload,
        multiDevice: {
          system: hypoSystem,
          parallelism: pc.parallelism,
          parallelismDegrees: pc.parallelismDegrees,
        },
      }

      let result
      try { result = calculate(input) } catch { continue }
      const peak = result.perf['peak']
      if (!peak) continue

      // Derive the comms bytes for the decode pass directly so we can show them.
      // Decode B = concurrency for a single-token-per-step pass.
      // TP all-reduce dominates: 2 × L × 2 × ((N-1)/N) × B × hidden × bytes
      const N = pc.parallelismDegrees.tp ?? 1
      const decodeCommsBytes = N > 1
        ? 2 * model.layers * 2 * ((N - 1) / N) * workload.concurrency * model.hiddenDim * 2
        : 0

      rows.push({
        fabric: fabric.name,
        fabricGBs: fabricBW,
        model: model.name,
        concurrency: workload.concurrency,
        prefillRegime: peak.prefill.regime,
        decodeRegime: peak.decode.regime,
        decodeBytesGB: peak.decode.bytesPerStep / 1e9,
        decodeCommsGB: decodeCommsBytes / 1e9,
      })
    }
  }
}

console.log(`\nHypothetical: HGX H100 chassis (TP=8) wired over slow scale-out fabric.\n`)
console.log('H100 HBM bandwidth: 3350 GB/s')
console.log('NVLink-4 (real):    450 GB/s per direction')
console.log('Fabrics tested:')
for (const id of FABRICS) {
  const f = INTERCONNECTS.find(i => i.id === id)!
  console.log(`  ${f.name.padEnd(22)} ${(f.perDirectionGBs ?? f.perGpuBandwidthGBs / 2).toString().padStart(5)} GB/s per direction`)
}

// === Decode comms-bound rate by fabric ===
console.log('\nDecode comms-bound rate (of all models tested, by fabric × concurrency):')
const fabrics = Array.from(new Set(rows.map(r => r.fabric)))
const concurrencies = Array.from(new Set(rows.map(r => r.concurrency))).sort((a, b) => a - b)
console.log('Fabric'.padEnd(22) + concurrencies.map(c => `c=${c}`.padStart(10)).join(''))
for (const f of fabrics) {
  const cells = concurrencies.map(c => {
    const subset = rows.filter(r => r.fabric === f && r.concurrency === c)
    const commsBound = subset.filter(r => r.decodeRegime === 'comms').length
    return `${commsBound}/${subset.length}`.padStart(10)
  })
  console.log(f.padEnd(22) + cells.join(''))
}

// === First decode-comms-bound case per fabric (smallest concurrency) ===
console.log('\nFirst decode-comms-bound model (lowest concurrency that flips):')
for (const f of fabrics) {
  for (const c of concurrencies) {
    const hits = rows.filter(r => r.fabric === f && r.concurrency === c && r.decodeRegime === 'comms')
    if (hits.length > 0) {
      const first = hits[0]
      console.log(`  ${f.padEnd(22)} c=${c}: ${first.model} (decode bytes ${first.decodeBytesGB.toFixed(2)} GB, comms ${first.decodeCommsGB.toFixed(2)} GB)`)
      break
    }
  }
}

// === Specific examples: Llama 70B + DeepSeek V3 + Qwen3 32B across fabrics ===
console.log('\nTracking specific models across fabrics (decode regime @ c=256):')
const examples = ['Llama 3.3 70B', 'DeepSeek-V3', 'Qwen3 32B', 'Qwen3-30B-A3B', 'Mixtral 8x7B v0.1']
console.log('Model'.padEnd(28) + fabrics.map(f => f.padStart(18)).join(''))
for (const m of examples) {
  const cells = fabrics.map(f => {
    const row = rows.find(r => r.model === m && r.fabric === f && r.concurrency === 256)
    return (row?.decodeRegime ?? '?').padStart(18)
  })
  console.log(m.padEnd(28) + cells.join(''))
}

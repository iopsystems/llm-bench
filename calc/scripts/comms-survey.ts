/**
 * Survey: which (system, model, workload) combos land in the 'comms' regime
 * under the default parallelism heuristic?
 *
 * Iterates over every system × model × a few representative workloads. For each
 * combo: applies defaultParallelism, runs calculate(), and reports prefill /
 * decode regime. Filters output to highlight comms-bound cases.
 *
 * Run: npx tsx scripts/comms-survey.ts
 */

import { ACCELERATORS } from '../src/data/accelerators'
import { MODELS } from '../src/data/models'
import { SYSTEMS } from '../src/data/systems'
import { defaultParallelism } from '../src/engine/parallelism'
import { calculate } from '../src/engine/calc'
import type { CalcInput, Quantization, Workload } from '../src/engine/types'

const QUANT: Quantization = { weights: 'fp16', kv: 'fp16', activations: 'fp16' }

// A spread of workloads that exercise different regimes:
//   short prompt + low concurrency: decode-dominated, comms is per-token small
//   long prompt + low concurrency:  prefill comms scales with prompt × concurrency
//   short prompt + high concurrency: decode comms scales with concurrency
//   very long prompt: stress the long-context regime
const WORKLOADS: Workload[] = [
  { promptTokens: 1024,   outputTokens: 256, concurrency: 1    },
  { promptTokens: 8192,   outputTokens: 512, concurrency: 1    },
  { promptTokens: 32768,  outputTokens: 512, concurrency: 1    },
  { promptTokens: 131072, outputTokens: 512, concurrency: 1    },
  { promptTokens: 1024,   outputTokens: 256, concurrency: 32   },
  { promptTokens: 8192,   outputTokens: 512, concurrency: 32   },
  { promptTokens: 1024,   outputTokens: 256, concurrency: 256  },
  { promptTokens: 1024,   outputTokens: 256, concurrency: 1024 },
  { promptTokens: 8192,   outputTokens: 512, concurrency: 1024 },
]

interface Hit {
  system: string
  model: string
  workload: string
  parallelism: string
  prefillRegime: string
  decodeRegime: string
  fitsPerRank: boolean
  decodeBytesGB: number
  commsTimeSPerStep?: number
}

const hits: Hit[] = []
const allRows: Hit[] = []

for (const system of SYSTEMS) {
  const accelerator = ACCELERATORS.find(a => a.id === system.accelerator.id)
  if (!accelerator) continue

  for (const model of MODELS) {
    const pc = defaultParallelism(system, model)

    for (const workload of WORKLOADS) {
      const input: CalcInput = {
        accelerator,
        acceleratorVariantId: system.accelerator.variantId,
        model,
        quant: QUANT,
        workload,
        multiDevice: {
          system,
          parallelism: pc.parallelism,
          parallelismDegrees: pc.parallelismDegrees,
        },
      }

      let result
      try {
        result = calculate(input)
      } catch {
        continue
      }

      const peak = result.perf['peak']
      if (!peak) continue

      const wkLabel = `p=${workload.promptTokens / 1024 < 1 ? workload.promptTokens : (workload.promptTokens / 1024 + 'k')} c=${workload.concurrency}`
      const parallelismLabel = Object.entries(pc.parallelismDegrees)
        .map(([k, v]) => `${k.toUpperCase()}=${v}`).join('×')

      const row: Hit = {
        system: system.name,
        model: model.name,
        workload: wkLabel,
        parallelism: parallelismLabel,
        prefillRegime: peak.prefill.regime,
        decodeRegime: peak.decode.regime,
        fitsPerRank: result.memory.perRank?.fits ?? result.memory.fits,
        decodeBytesGB: peak.decode.bytesPerStep / 1e9,
      }
      allRows.push(row)

      if (peak.prefill.regime === 'comms' || peak.decode.regime === 'comms') {
        hits.push(row)
      }
    }
  }
}

console.log(`\nSurveyed ${SYSTEMS.length} systems × ${MODELS.length} models × ${WORKLOADS.length} workloads = ${allRows.length} viable combinations`)

// === Overall regime distribution ===
const dist = { 'prefill compute': 0, 'prefill memory': 0, 'prefill comms': 0,
               'decode compute': 0,  'decode memory': 0,  'decode comms': 0 }
for (const r of allRows) {
  dist[`prefill ${r.prefillRegime}` as keyof typeof dist]++
  dist[`decode ${r.decodeRegime}` as keyof typeof dist]++
}
console.log('\nRegime distribution (count / total):')
for (const [k, v] of Object.entries(dist)) {
  const pct = ((v / allRows.length) * 100).toFixed(0)
  console.log(`  ${k.padEnd(20)} ${String(v).padStart(5)}  (${pct}%)`)
}

// === Comms-bound by workload bucket ===
console.log('\nComms-bound by workload (any phase):')
const byWorkload = new Map<string, number>()
for (const h of hits) {
  byWorkload.set(h.workload, (byWorkload.get(h.workload) ?? 0) + 1)
}
for (const [k, v] of Array.from(byWorkload.entries()).sort()) {
  const totalForWorkload = allRows.filter(r => r.workload === k).length
  console.log(`  ${k.padEnd(15)} ${String(v).padStart(4)} / ${totalForWorkload}`)
}

// === Comms-bound by system ===
console.log('\nComms-bound by system (any phase, fits per-rank):')
const bySys = new Map<string, number>()
const fittingHits = hits.filter(h => h.fitsPerRank)
for (const h of fittingHits) {
  bySys.set(h.system, (bySys.get(h.system) ?? 0) + 1)
}
for (const [k, v] of Array.from(bySys.entries()).sort((a, b) => b[1] - a[1])) {
  const totalForSys = allRows.filter(r => r.system === k && r.fitsPerRank).length
  console.log(`  ${k.padEnd(40)} ${String(v).padStart(4)} / ${totalForSys}`)
}

// === The interesting category: DECODE comms-bound and fits per-rank ===
// (Prefill comms at high concurrency is expected — that's what TP all-reduce
// does. Decode comms-bound means even single-token-per-step generation is
// limited by the fabric, which is the diagnostic the calc is meant to surface.)
const decodeComms = hits.filter(h => h.decodeRegime === 'comms' && h.fitsPerRank)
console.log(`\nDecode comms-bound (fits per-rank): ${decodeComms.length} cases\n`)
if (decodeComms.length > 0) {
  console.log('System'.padEnd(40) + 'Model'.padEnd(28) + 'Workload'.padEnd(15) + 'Parallelism'.padEnd(22) + 'Decode')
  console.log('-'.repeat(130))
  for (const h of decodeComms.slice(0, 60)) {
    console.log(
      h.system.padEnd(40) +
      h.model.padEnd(28) +
      h.workload.padEnd(15) +
      h.parallelism.padEnd(22) +
      h.decodeRegime
    )
  }
  if (decodeComms.length > 60) console.log(`  ... ${decodeComms.length - 60} more`)
}

import { describe, it, expect } from 'vitest'
import { calculate } from '../../src/engine/calc'
import { bytesOf } from '../../src/engine/dtypes'
import { ACCELERATORS, MODELS } from '../../src/data'
import { SYSTEMS } from '../../src/data/systems'
import { INTERCONNECTS } from '../../src/data/interconnects'
import type { CalcInput, DerivationStep } from '../../src/engine/types'

// Audit: every drawer row's printed formula must reproduce its printed value
// using only other drawer rows and the selected SKU's rates. Run across the
// whole catalog, single- and multi-device, so a formula/value mismatch in any
// attention variant or parallelism path fails loudly.

const h100 = ACCELERATORS.find(a => a.id === 'h100')!
const hgxH100 = SYSTEMS.find(s => s.id === 'hgx-h100-8')!

function row(steps: DerivationStep[], label: string): DerivationStep {
  const s = steps.find(x => x.label === label)
  expect(s, `missing derivation row: ${label}`).toBeDefined()
  return s!
}

function maybeRow(steps: DerivationStep[], label: string): DerivationStep | undefined {
  return steps.find(x => x.label === label)
}

// Relative comparison: values span ~20 orders of magnitude across rows.
function expectClose(actual: number, expected: number, ctx: string) {
  if (expected === 0) {
    expect(actual, ctx).toBe(0)
  } else {
    expect(actual / expected, ctx).toBeCloseTo(1, 9)
  }
}

function auditConfig(input: CalcInput, ctx: string) {
  const r = calculate(input)
  const d = r.derivation
  const { model, quant, workload } = input
  const seqlen = workload.promptTokens + workload.outputTokens

  // --- memory rows ---
  expectClose(
    row(d, 'weights').value,
    model.paramCount * bytesOf(quant.weights),
    `${ctx}: weights = paramCount × bytes(weight_dtype)`
  )
  const kvPerToken = row(d, 'kv per token per request')
  const kvPerRequest = row(d, 'kv per request')
  expectClose(
    kvPerToken.value * seqlen, kvPerRequest.value,
    `${ctx}: kv_per_request = kv_per_token × (prompt + output)`
  )
  // Exact closed-form kv-per-token expressions must be literally evaluable.
  if (kvPerToken.expression === '2 × layers × kv_heads × head_dim × bytes(kv_dtype)') {
    expectClose(
      kvPerToken.value,
      2 * model.layers * model.numKvHeads * model.headDim * bytesOf(quant.kv),
      `${ctx}: full-attention kv-per-token closed form`
    )
  }
  if (kvPerToken.expression === '(kv_lora_rank + rope_dim) × bytes(kv_dtype) × layers') {
    const att = model.attention
    if (att.type !== 'mla' && att.type !== 'mla-dsa') {
      throw new Error(`${ctx}: MLA expression printed for ${att.type}`)
    }
    expectClose(
      kvPerToken.value,
      (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(quant.kv) * model.layers,
      `${ctx}: MLA kv-per-token closed form`
    )
  }
  if (kvPerToken.expression === '2 × attn_layers × kv_heads × head_dim × bytes(kv_dtype)') {
    const att = model.attention
    if (att.type !== 'partial') throw new Error(`${ctx}: partial expression printed for ${att.type}`)
    expectClose(
      kvPerToken.value,
      2 * att.numFullLayers * model.numKvHeads * model.headDim * bytesOf(quant.kv),
      `${ctx}: partial-attention kv-per-token closed form`
    )
  }
  const kvTotal = row(d, 'kv total')
  expectClose(
    kvTotal.value, kvPerRequest.value * workload.concurrency,
    `${ctx}: kv_total = kv_per_request × concurrency`
  )
  const prefillAct = row(d, 'activations peak (prefill, coarse)')
  expectClose(
    prefillAct.value,
    workload.concurrency * workload.promptTokens *
      (model.hiddenDim + model.intermediateDim) * bytesOf(quant.activations) * 2,
    `${ctx}: prefill activations closed form`
  )
  const decodeAct = row(d, 'activations peak (decode, coarse)')
  expectClose(
    decodeAct.value,
    workload.concurrency * (model.hiddenDim + model.intermediateDim) * bytesOf(quant.activations) * 2,
    `${ctx}: decode activations closed form`
  )
  expectClose(
    row(d, 'prefill side total').value,
    row(d, 'weights').value + kvTotal.value + prefillAct.value,
    `${ctx}: prefill side total = weights + kv_total + prefill_activations`
  )
  expectClose(
    row(d, 'decode side total').value,
    row(d, 'weights').value + kvTotal.value + decodeAct.value,
    `${ctx}: decode side total = weights + kv_total + decode_activations`
  )
  expectClose(
    row(d, 'memory total').value, row(d, 'prefill side total').value,
    `${ctx}: memory total mirrors prefill side`
  )

  // --- volume rows match perf ---
  const prefillFlops = row(d, 'prefill flops')
  const prefillBytes = row(d, 'prefill bytes (hbm)')
  const decodeFlops = row(d, 'decode flops per step')
  const decodeBytes = row(d, 'decode bytes per step')
  const anyTier = Object.values(r.perf)[0]
  expectClose(prefillFlops.value, anyTier.prefill.flops, `${ctx}: prefill flops row`)
  expectClose(prefillBytes.value, anyTier.prefill.bytes, `${ctx}: prefill bytes row`)
  expectClose(decodeFlops.value, anyTier.decode.flopsPerStep, `${ctx}: decode flops row`)
  expectClose(decodeBytes.value, anyTier.decode.bytesPerStep, `${ctx}: decode bytes row`)
  expectClose(
    prefillBytes.value,
    row(d, 'weights').value + prefillAct.value,
    `${ctx}: prefill bytes = weights + prefill_activations`
  )

  const prefillComms = maybeRow(d, 'prefill comms bytes')
  const decodeComms = maybeRow(d, 'decode comms bytes')
  const icRow = maybeRow(d, 'interconnect bw (per direction)')
  if (input.multiDevice) {
    expect(prefillComms, `${ctx}: prefill comms row present`).toBeDefined()
    expect(decodeComms, `${ctx}: decode comms row present`).toBeDefined()
    expect(icRow, `${ctx}: interconnect bw row present`).toBeDefined()
    const ic = INTERCONNECTS.find(i => i.id === input.multiDevice!.system.interconnectId)!
    expectClose(
      icRow!.value, ic.perDirectionGBs ?? ic.perGpuBandwidthGBs / 2,
      `${ctx}: interconnect bw row value`
    )
  } else {
    expect(prefillComms, `${ctx}: no comms row single-device`).toBeUndefined()
    expect(icRow, `${ctx}: no interconnect row single-device`).toBeUndefined()
  }

  // --- time rows reproduce from volume rows + the tier's rates ---
  const variant = input.accelerator.variants.find(v => v.id === input.acceleratorVariantId)!
  for (const op of variant.operatingPoints) {
    const tier = r.perf[op.id]
    if (!tier) continue
    const t = op.tflops[quant.activations]! * 1e12
    const bw = op.hbmBandwidthGBs * 1e9
    const icBw = icRow ? icRow.value * 1e9 : undefined

    const prefillTimeRow = row(d, `prefill time @ ${op.id}`)
    const expectedPrefill = Math.max(
      prefillFlops.value / t,
      prefillBytes.value / bw,
      icBw !== undefined && prefillComms ? prefillComms.value / icBw : 0
    )
    expectClose(prefillTimeRow.value, expectedPrefill, `${ctx}: prefill time @ ${op.id}`)
    expectClose(prefillTimeRow.value, tier.prefill.timeS, `${ctx}: prefill time row = perf @ ${op.id}`)

    const decodeTimeRow = row(d, `decode time per token @ ${op.id}`)
    const expectedDecodeBase = Math.max(
      decodeFlops.value / t,
      decodeBytes.value / bw,
      icBw !== undefined && decodeComms ? decodeComms.value / icBw : 0
    )
    const expectedDecode = expectedDecodeBase / (1 + model.numNextnLayers)
    expectClose(decodeTimeRow.value, expectedDecode, `${ctx}: decode time @ ${op.id}`)
    expectClose(decodeTimeRow.value, tier.decode.timePerTokenS, `${ctx}: decode time row = perf @ ${op.id}`)
    // The MTP division must be visible in the formula whenever it's applied.
    if (model.numNextnLayers > 0) {
      expect(decodeTimeRow.expression, `${ctx}: decode expr shows MTP`).toContain('1 + mtp_depth')
    }
  }
}

describe('derivation audit — formulas reproduce values across the catalog', () => {
  it('single-device: every model on H100 SXM-80', () => {
    for (const m of MODELS) {
      auditConfig({
        accelerator: h100,
        acceleratorVariantId: 'sxm-80',
        model: m,
        quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
        workload: { promptTokens: 8192, outputTokens: 512, concurrency: 4 }
      }, m.id)
    }
  })

  it('multi-device: every model on HGX H100-8 (TP=8, +EP=8 for MoE)', () => {
    for (const m of MODELS) {
      const moe = m.architecture.type === 'moe'
      auditConfig({
        accelerator: h100,
        acceleratorVariantId: 'sxm-80',
        model: m,
        quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' },
        workload: { promptTokens: 8192, outputTokens: 512, concurrency: 64 },
        multiDevice: {
          system: hgxH100,
          parallelism: moe ? ['tp', 'ep'] : ['tp'],
          parallelismDegrees: moe ? { tp: 8, ep: 8 } : { tp: 8 }
        }
      }, `${m.id} (multi)`)
    }
  })
})

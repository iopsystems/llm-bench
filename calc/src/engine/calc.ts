import type { CalcInput, CalcResult, ModelArch, PerfTier } from './types'
import { bytesOf } from './dtypes'
import { computeMemory } from './memory'
import { computePrefill } from './prefill'
import { computeDecode } from './decode'
import { DerivationBuilder } from './derivation'
import { INTERCONNECTS } from '../data/interconnects'
import { pairOpPoints } from './opPoints'

// Derivation-display formula for the kv-per-token row. Must mirror
// kvBytesPerTokenPerLayer × attendedSeqlenSummedOverLayers in memory.ts: the
// printed expression has to reproduce the printed value. Variants whose
// per-token cost depends on seqlen (windows, compression, recurrent state)
// get an honest "effective" description instead of a fake closed form.
function kvPerTokenExpression(model: ModelArch): string {
  switch (model.attention.type) {
    case 'full':
      return '2 × layers × kv_heads × head_dim × bytes(kv_dtype)'
    case 'mla':
    case 'mla-dsa':
      return '(kv_lora_rank + rope_dim) × bytes(kv_dtype) × layers'
    case 'partial':
      return '2 × attn_layers × kv_heads × head_dim × bytes(kv_dtype)'
    default:
      return 'kv_cache_per_request ÷ (prompt + output) — effective avg; window/state-dependent for this attention variant'
  }
}

export function calculate(input: CalcInput): CalcResult {
  // Resolve both sides. Decode side falls back to prefill when fields absent.
  const prefillVariant = input.accelerator.variants.find(v => v.id === input.acceleratorVariantId)
  if (!prefillVariant) {
    throw new Error(`Variant ${input.acceleratorVariantId} not in ${input.accelerator.id}`)
  }
  const decodeAccelerator = input.decodeAccelerator ?? input.accelerator
  const decodeVariantId = input.decodeAcceleratorVariantId ?? input.acceleratorVariantId
  const decodeVariant = decodeAccelerator.variants.find(v => v.id === decodeVariantId)
  if (!decodeVariant) {
    throw new Error(`Variant ${decodeVariantId} not in ${decodeAccelerator.id}`)
  }

  // Validate activations dtype against both sides' operating points up front, so
  // the error message names the accelerator and supported alternatives rather
  // than leaking engine vocabulary.
  for (const op of prefillVariant.operatingPoints) {
    if (op.tflops[input.quant.activations] === undefined) {
      const supported = Object.keys(op.tflops).join(', ')
      throw new Error(
        `${input.accelerator.name} ${prefillVariant.label} has no ${input.quant.activations} ` +
        `compute throughput. Try: ${supported}.`
      )
    }
  }
  for (const op of decodeVariant.operatingPoints) {
    if (op.tflops[input.quant.activations] === undefined) {
      const supported = Object.keys(op.tflops).join(', ')
      throw new Error(
        `${decodeAccelerator.name} ${decodeVariant.label} has no ${input.quant.activations} ` +
        `compute throughput. Try: ${supported}.`
      )
    }
  }

  const memory = computeMemory(input)
  const d = new DerivationBuilder()

  d.add('weights', 'paramCount × bytes(weight_dtype)', memory.weights, 'bytes')
  d.add(
    'kv per token per request',
    kvPerTokenExpression(input.model),
    memory.kvCachePerRequest / (input.workload.promptTokens + input.workload.outputTokens),
    'bytes'
  )
  d.add('kv per request', 'kv_per_token × (prompt + output)', memory.kvCachePerRequest, 'bytes')
  d.add('kv total', 'kv_per_request × concurrency', memory.kvCacheTotal, 'bytes')
  d.add(
    'activations peak (prefill, coarse)',
    'concurrency × prompt × (hidden + intermediate) × bytes(act_dtype) × 2',
    memory.activationsPeak, 'bytes'
  )
  d.add(
    'activations peak (decode, coarse)',
    'concurrency × 1 × (hidden + intermediate) × bytes(act_dtype) × 2',
    memory.decodeActivationsPeak, 'bytes'
  )
  d.add('prefill side total', 'weights + kv_total + prefill_activations', memory.prefillSide.total, 'bytes')
  d.add('decode side total',  'weights + kv_total + decode_activations',  memory.decodeSide.total,  'bytes')
  // Backward-compat row: mirrors prefillSide so existing UI/tests that key off
  // "memory total" keep working unchanged.
  d.add('memory total', 'prefill_side.total', memory.total, 'bytes')

  // Disaggregated serving: KV cache ships from prefill cluster to decode cluster
  // over a separate fabric. Adds a one-shot transfer time to TTFT.
  // For integrated serving (single cluster) this is 0.
  let kvTransferS = 0
  if (input.disaggKvTransferFabricId) {
    const fab = INTERCONNECTS.find(i => i.id === input.disaggKvTransferFabricId)
    if (fab) {
      const bw = fab.perDirectionGBs ?? fab.perGpuBandwidthGBs / 2
      kvTransferS = memory.kvCachePerRequest / (bw * 1e9)
    }
  }
  // Production-standard: prefill node emits the first decoded token locally
  // while KV transfer streams in parallel. Defaults true when disagg is on.
  const firstTokenOnPrefill = input.disaggFirstTokenOnPrefill ?? true

  const perf: Record<string, PerfTier> = {}
  // flops/bytes/comms volumes are op-point-independent (only the rates differ
  // per tier), so emit those rows once, ahead of the per-tier time rows.
  let volumeRowsAdded = false
  for (const pair of pairOpPoints(prefillVariant, decodeVariant)) {
    // computePrefill uses input.multiDevice (prefill side); computeDecode uses
    // input.decodeMultiDevice ?? input.multiDevice (decode side, falls back).
    const prefill = computePrefill(input, pair.prefillOp, memory)
    const decode  = computeDecode(input, pair.decodeOp,  memory)

    // TTFT case-B (firstTokenOnPrefill=true): the first decode step runs on the
    // PREFILL cluster, so the step time must reflect prefill-side TFLOPS/HBM —
    // NOT the decode cluster's. Recompute decode on the prefill op + force
    // multiDevice to the prefill cluster's parallelism config.
    let firstStepOnPrefillS = decode.timePerTokenS  // fallback covers symmetric case
    if (kvTransferS > 0 && firstTokenOnPrefill) {
      const onPrefill = computeDecode(input, pair.prefillOp, memory, input.multiDevice)
      firstStepOnPrefillS = onPrefill.timePerTokenS
    }

    // firstTokenOnPrefill=true:  ttft = prefill + first decode on prefill cluster
    //                            (KV transfer streams in parallel, hidden).
    // firstTokenOnPrefill=false: ttft = prefill + full KV transfer (worst case).
    const ttftS = kvTransferS > 0 && firstTokenOnPrefill
      ? prefill.timeS + firstStepOnPrefillS
      : prefill.timeS + kvTransferS

    perf[pair.id] = {
      prefill, decode,
      ttftS,
      kvTransferS,
      // inputTokenRate stays on prefill (cluster throughput); ttftS already
      // captures the disagg overhead (user-facing latency to first token).
      inputTokenRate: input.workload.promptTokens / prefill.timeS,
      outputTokenRate: decode.aggregateTokensPerS,
      // Provenance: echo from the prefill-side op point (the historically
      // canonical source of these fields; decode-side may have its own asOf/notes
      // that we don't surface yet to avoid breaking PerfTier shape).
      ...(pair.prefillOp.tflopsSources && { tflopsSources: pair.prefillOp.tflopsSources }),
      ...(pair.prefillOp.bandwidthSources && { bandwidthSources: pair.prefillOp.bandwidthSources }),
      ...(pair.prefillOp.asOf && { asOf: pair.prefillOp.asOf }),
      ...(pair.prefillOp.notes && { notes: pair.prefillOp.notes })
    }
    if (!volumeRowsAdded) {
      volumeRowsAdded = true
      d.add('prefill flops', '2 × active_params × prompt + attention + state terms', prefill.flops, 'FLOPs')
      d.add('prefill bytes (hbm)', 'weights + prefill_activations', prefill.bytes, 'bytes')
      if (prefill.commsBytes !== undefined) {
        d.add(
          'prefill comms bytes',
          'TP all-reduce + PP sends + EP all-to-all volume, B = prompt × concurrency',
          prefill.commsBytes, 'bytes'
        )
      }
      d.add('decode flops per step', '(2 × active_params + attention + state terms) × concurrency', decode.flopsPerStep, 'FLOPs')
      d.add('decode bytes per step', 'active_params × bytes(weight_dtype) + kv_total + state write-back', decode.bytesPerStep, 'bytes')
      if (decode.commsBytes !== undefined) {
        d.add(
          'decode comms bytes',
          'TP all-reduce + PP sends + EP all-to-all volume, B = concurrency',
          decode.commsBytes, 'bytes'
        )
      }
      // The time formulas divide comms bytes by this; surface it so the
      // drawer is self-contained (tflops/hbm_bw live on the SKU spec sheet,
      // but the fabric rate appears nowhere else in the UI).
      if (input.multiDevice) {
        const ic = INTERCONNECTS.find(i => i.id === input.multiDevice!.system.interconnectId)
        if (ic) {
          d.add(
            'interconnect bw (per direction)',
            'system fabric per-direction bandwidth (bidirectional ÷ 2 unless specified)',
            ic.perDirectionGBs ?? ic.perGpuBandwidthGBs / 2, 'GB/s'
          )
        }
      }
    }
    const prefillTimeExpr = prefill.commsBytes !== undefined
      ? 'max(prefill_flops ÷ tflops, prefill_bytes ÷ hbm_bw, prefill_comms_bytes ÷ interconnect_bw)'
      : 'max(prefill_flops ÷ tflops, prefill_bytes ÷ hbm_bw)'
    d.add(`prefill time @ ${pair.id}`, prefillTimeExpr, prefill.timeS, 's')
    if (kvTransferS > 0) {
      d.add(
        `kv transfer time @ ${pair.id}`,
        firstTokenOnPrefill
          ? 'kv_cache_per_request / disagg_fabric_bw (overlapped with first decode)'
          : 'kv_cache_per_request / disagg_fabric_bw',
        kvTransferS, 's'
      )
    }
    const decodeTimeExpr = (decode.commsBytes !== undefined
      ? 'max(decode_flops ÷ tflops, decode_bytes ÷ hbm_bw, decode_comms_bytes ÷ interconnect_bw)'
      : 'max(decode_flops ÷ tflops, decode_bytes ÷ hbm_bw)')
      + (input.model.numNextnLayers > 0 ? ' ÷ (1 + mtp_depth)' : '')
    d.add(`decode time per token @ ${pair.id}`, decodeTimeExpr, decode.timePerTokenS, 's')
  }

  // bytesOf re-exported so consumers can read the same dtype table.
  void bytesOf

  return { memory, perf, derivation: d.steps() }
}

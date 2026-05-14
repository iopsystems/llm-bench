import type { Dtype, ModelArch, MultiAcceleratorSystem, ParallelismMode } from './types'
import { bytesOf } from './dtypes'

export interface RankDivisors {
  weights: number
  kv: number
  activations: number
  replicas: number
}

export function perRankMemoryDivisors(
  parallelism: ParallelismMode['id'][],
  degrees: Partial<Record<ParallelismMode['id'], number>>,
  model: ModelArch
): RankDivisors {
  const tp = parallelism.includes('tp') ? (degrees.tp ?? 1) : 1
  const pp = parallelism.includes('pp') ? (degrees.pp ?? 1) : 1
  const ep = parallelism.includes('ep') ? (degrees.ep ?? 1) : 1
  const dp = parallelism.includes('dp') ? (degrees.dp ?? 1) : 1

  // Weights: TP shards weight matrices, PP shards layers, EP shards routed-expert
  // weights (first-cut approximates as full N divisor for MoE), DP replicates.
  const weightsDivisor = tp * pp * (model.architecture.type === 'moe' && ep > 1 ? ep : 1)

  // KV cache: TP shards heads (capped at numKvHeads), PP per-stage, EP/DP replicated.
  const kvShard = Math.min(tp, model.numKvHeads)
  const kvDivisor = kvShard * pp

  // Activations: TP shards them; PP/EP/DP don't (per-stage forward, replicated).
  const activationsDivisor = tp

  return {
    weights: weightsDivisor,
    kv: kvDivisor,
    activations: activationsDivisor,
    replicas: dp
  }
}

export function commsBytesPerStep(
  parallelism: ParallelismMode['id'][],
  degrees: Partial<Record<ParallelismMode['id'], number>>,
  model: ModelArch,
  B: number,
  activationDtype: Dtype
): number {
  const tp = parallelism.includes('tp') ? (degrees.tp ?? 1) : 1
  const pp = parallelism.includes('pp') ? (degrees.pp ?? 1) : 1
  const ep = parallelism.includes('ep') ? (degrees.ep ?? 1) : 1

  const d = model.hiddenDim
  const L = model.layers
  const bytes = bytesOf(activationDtype)
  let total = 0

  // TP: two all-reduces per layer (ring algorithm): 2 × (N-1)/N × B × d × bytes each
  if (tp > 1) {
    total += 2 * L * 2 * ((tp - 1) / tp) * B * d * bytes
  }
  // PP: (N-1) point-to-point sends per forward pass
  if (pp > 1) {
    total += (pp - 1) * B * d * bytes
  }
  // EP: all-to-all per MoE layer (forward gather + scatter); zero for dense
  if (ep > 1 && model.architecture.type === 'moe') {
    total += 2 * L * (1 - 1 / ep) * B * d * bytes
  }

  return total
}

export interface ParallelismConfig {
  parallelism: ParallelismMode['id'][]
  parallelismDegrees: Partial<Record<ParallelismMode['id'], number>>
}

export function defaultParallelism(
  system: MultiAcceleratorSystem,
  model: ModelArch
): ParallelismConfig {
  const N = system.accelerator.count
  const isMoE = model.architecture.type === 'moe'

  const tp = Math.min(N, 8)
  const pp = N > 8 ? Math.ceil(N / 8) : 1

  const parallelism: ParallelismMode['id'][] = ['tp']
  const degrees: Partial<Record<ParallelismMode['id'], number>> = { tp }

  if (pp > 1) {
    parallelism.push('pp')
    degrees.pp = pp
  }
  if (isMoE) {
    parallelism.push('ep')
    degrees.ep = N
  }
  return { parallelism, parallelismDegrees: degrees }
}

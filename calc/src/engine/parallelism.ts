import type { ModelArch, ParallelismMode } from './types'

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

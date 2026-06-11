// Display ordering for the model and SKU pickers. Pure functions, no UI deps,
// so the ordering rules can be unit-tested in isolation. Data files keep their
// human-readable grouping; this is the single place ordering is decided.
//
// Models: publisher → (within publisher) newer first → larger paramCount first.
// SKUs:   vendor → (AMD only) product line → single accelerators before
//         systems → newer first → larger first (accelerator count desc, then
//         HBM desc).
//
// AMD spans datacenter (Instinct) and client/workstation (Radeon RX / PRO /
// AI PRO) product lines that pure recency interleaves confusingly. For AMD we
// group by product line first, in the order Instinct → Radeon AI PRO →
// Radeon PRO → Radeon RX; other vendors are left on pure recency.
//
// Publisher/vendor groups are themselves ordered by the recency of their
// newest entry (the publisher who shipped most recently floats to the top),
// ties broken by name ascending for stable output.

import type { ModelArch, AcceleratorSpec, MultiAcceleratorSystem } from '../engine/types'

// ISO `YYYY-MM` compares correctly as a string; slice guards stray day parts.
const ym = (s: string) => s.slice(0, 7)

// AMD product-line rank, derived from the SKU name. Lower sorts first.
// Non-AMD vendors all map to 0, leaving their ordering on pure recency.
// Order: Instinct (datacenter) → Radeon AI PRO → Radeon PRO → Radeon RX.
// Check the more specific "Radeon AI PRO" / "Radeon PRO" before "Radeon" alone.
function amdLineRank(vendor: string, name: string): number {
  if (vendor !== 'AMD') return 0
  if (name.includes('Instinct')) return 0
  if (name.includes('Radeon AI PRO')) return 1
  if (name.includes('Radeon PRO')) return 2
  if (name.includes('Radeon RX')) return 3
  return 4 // any future/other AMD line sorts after the known ones
}

export interface ModelGroup {
  publisher: string
  models: ModelArch[]
}

export function orderModels(models: ModelArch[]): ModelGroup[] {
  const byPublisher = new Map<string, ModelArch[]>()
  for (const m of models) {
    const arr = byPublisher.get(m.publisher) ?? []
    arr.push(m)
    byPublisher.set(m.publisher, arr)
  }

  const groups: ModelGroup[] = []
  for (const [publisher, ms] of byPublisher) {
    ms.sort((a, b) =>
      ym(b.releaseDate).localeCompare(ym(a.releaseDate)) ||  // newer first
      b.paramCount - a.paramCount                            // larger first
    )
    groups.push({ publisher, models: ms })
  }

  groups.sort((a, b) => {
    const newestA = ym(a.models[0].releaseDate)
    const newestB = ym(b.models[0].releaseDate)
    return newestB.localeCompare(newestA) || a.publisher.localeCompare(b.publisher)
  })
  return groups
}

export type SkuEntry =
  | { kind: 'single'; id: string; publisher: string; name: string }
  | { kind: 'system'; id: string; publisher: string; name: string; count: number }

export interface SkuGroup {
  publisher: string
  entries: SkuEntry[]
}

interface SkuRow {
  entry: SkuEntry
  vendor: string
  releaseDate: string
  lineRank: number   // product-line rank (AMD only; 0 for other vendors)
  typeRank: number   // 0 = single accelerator, 1 = multi-accelerator system
  count: number      // accelerators in the SKU (1 for a single chip)
  hbmGB: number      // headline memory: max variant HBM (single) / total (system)
}

export function orderSkus(
  accelerators: AcceleratorSpec[],
  systems: MultiAcceleratorSystem[],
): SkuGroup[] {
  const rows: SkuRow[] = []

  for (const a of accelerators) {
    const hbmGB = a.variants.reduce((mx, v) => Math.max(mx, v.hbmCapacityGB), 0)
    rows.push({
      entry: { kind: 'single', id: a.id, publisher: a.vendor, name: a.name },
      vendor: a.vendor, releaseDate: a.releaseDate,
      lineRank: amdLineRank(a.vendor, a.name),
      typeRank: 0, count: 1, hbmGB,
    })
  }
  for (const s of systems) {
    rows.push({
      entry: { kind: 'system', id: s.id, publisher: s.vendor, name: s.name, count: s.accelerator.count },
      vendor: s.vendor, releaseDate: s.releaseDate,
      lineRank: amdLineRank(s.vendor, s.name),
      typeRank: 1, count: s.accelerator.count, hbmGB: s.aggregate.totalHbmGB,
    })
  }

  const byVendor = new Map<string, SkuRow[]>()
  for (const r of rows) {
    const arr = byVendor.get(r.vendor) ?? []
    arr.push(r)
    byVendor.set(r.vendor, arr)
  }

  const groups: { publisher: string; rows: SkuRow[] }[] = []
  for (const [vendor, vrows] of byVendor) {
    vrows.sort((a, b) =>
      a.lineRank - b.lineRank ||                              // product line (AMD only)
      a.typeRank - b.typeRank ||                              // singles before systems
      ym(b.releaseDate).localeCompare(ym(a.releaseDate)) ||   // newer first
      b.count - a.count ||                                    // bigger cluster first
      b.hbmGB - a.hbmGB                                       // more memory first
    )
    groups.push({ publisher: vendor, rows: vrows })
  }

  // Order vendor groups by recency of their newest SKU, ties by name asc.
  const newest = (rs: SkuRow[]) =>
    rs.reduce((mx, r) => (ym(r.releaseDate) > mx ? ym(r.releaseDate) : mx), '')
  groups.sort((a, b) =>
    newest(b.rows).localeCompare(newest(a.rows)) ||
    a.publisher.localeCompare(b.publisher)
  )

  return groups.map(g => ({ publisher: g.publisher, entries: g.rows.map(r => r.entry) }))
}

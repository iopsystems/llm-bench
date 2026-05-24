// Unit-aware token-count parsing. "k" and "M" are binary (1024, 1024²) to
// match HuggingFace context-window conventions (8k=8192, 128k=131072, 1M=1048576).
export function parseTokenCount(s: string): number | null {
  const m = s.trim().match(/^(\d+(?:\.\d+)?)\s*([kKmM]?)$/)
  if (!m) return null   // unparseable (incl. negative — regex has no sign)
  const n = parseFloat(m[1])
  const unit = m[2].toLowerCase()
  let v: number
  if (unit === 'k') v = n * 1024
  else if (unit === 'm') v = n * 1024 * 1024
  else v = n
  v = Math.round(v)
  if (!Number.isFinite(v) || v < 1) return null
  return v
}

export function formatTokenCount(n: number): string {
  if (n >= 1024 * 1024 && n % (1024 * 1024) === 0) return `${n / (1024 * 1024)}M`
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`
  return `${n}`
}

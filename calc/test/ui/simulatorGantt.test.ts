import { describe, it, expect } from 'vitest'
import { computeGanttGeometry, type GanttInput } from '../../src/ui/simulatorGantt'

// Helper: round to 9 decimals to dodge floating-point noise in assertions.
const r = (n: number) => Math.round(n * 1e9) / 1e9

describe('computeGanttGeometry — non-disagg (case A)', () => {
  const input: GanttInput = {
    prefillS: 0.287,
    kvTransferS: 0,
    tpotS: 0.042,
    outputTokens: 512,
    firstTokenOnPrefill: true,
    ttftS: 0.287,
    prefillRegime: 'compute',
    decodeRegime: 'memory',
  }
  const geom = computeGanttGeometry(input)

  it('produces two segments: prefill + decode', () => {
    expect(geom.segments).toHaveLength(2)
    expect(geom.segments[0]).toMatchObject({ kind: 'prefill', x: 0, width: 0.287, regime: 'compute' })
    expect(geom.segments[1]).toMatchObject({ kind: 'decode',  x: 0.287, regime: 'memory' })
  })
  it('no KV-xfer overlay', () => {
    expect(geom.kvOverlay).toBeUndefined()
  })
  it('marker at ttftS', () => {
    expect(geom.markerX).toBe(0.287)
  })
  it('totalS = ttftS + tpotS * (outputTokens-1)', () => {
    expect(r(geom.totalS)).toBe(r(0.287 + 0.042 * 511))
    expect(r(geom.segments[1].x + geom.segments[1].width)).toBe(r(geom.totalS))
  })
})

describe('computeGanttGeometry — disagg sequential (case C, firstTokenOnPrefill=false)', () => {
  const input: GanttInput = {
    prefillS: 0.287,
    kvTransferS: 0.213,
    tpotS: 0.042,
    outputTokens: 512,
    firstTokenOnPrefill: false,
    ttftS: 0.287 + 0.213,
    prefillRegime: 'compute',
    decodeRegime: 'memory',
  }
  const geom = computeGanttGeometry(input)

  it('produces three serial segments: prefill, kv-xfer, decode', () => {
    expect(geom.segments).toHaveLength(3)
    expect(geom.segments[0]).toMatchObject({ kind: 'prefill', x: 0, width: 0.287, regime: 'compute' })
    expect(geom.segments[1]).toMatchObject({ kind: 'kv-xfer', x: 0.287, width: 0.213, regime: 'comms' })
    expect(geom.segments[2]).toMatchObject({ kind: 'decode',  x: 0.500, regime: 'memory' })
  })
  it('no overlay (KV is its own segment here)', () => {
    expect(geom.kvOverlay).toBeUndefined()
  })
  it('marker at the kv-xfer/decode boundary', () => {
    expect(geom.markerX).toBe(0.5)
  })
  it('decode segment ends at totalS = ttftS + tpotS * (outputTokens-1)', () => {
    expect(r(geom.totalS)).toBe(r(0.5 + 0.042 * 511))
    expect(r(geom.segments[2].x + geom.segments[2].width)).toBe(r(geom.totalS))
  })
})

describe('computeGanttGeometry — disagg overlap (case B, firstTokenOnPrefill=true)', () => {
  const input: GanttInput = {
    prefillS: 0.287,
    kvTransferS: 0.213,
    tpotS: 0.042,
    outputTokens: 512,
    firstTokenOnPrefill: true,
    ttftS: 0.287 + 0.042,
    prefillRegime: 'compute',
    decodeRegime: 'memory',
  }
  const geom = computeGanttGeometry(input)

  it('produces two main segments (prefill + decode) — KV-xfer is overlay, not a segment', () => {
    expect(geom.segments).toHaveLength(2)
    expect(geom.segments[0]).toMatchObject({ kind: 'prefill', x: 0, width: 0.287, regime: 'compute' })
    expect(geom.segments[1]).toMatchObject({ kind: 'decode',  x: 0.287, regime: 'memory' })
  })
  it('emits a KV-xfer overlay spanning [prefillS, prefillS + kvTransferS]', () => {
    expect(geom.kvOverlay).toEqual({ x: 0.287, width: 0.213 })
  })
  it('marker at ttftS = prefillS + tpotS', () => {
    expect(r(geom.markerX)).toBe(r(0.287 + 0.042))
  })
  it('totalS = ttftS + tpotS * (outputTokens-1) = prefillS + tpotS * outputTokens', () => {
    expect(r(geom.totalS)).toBe(r(0.287 + 0.042 * 512))
  })
})

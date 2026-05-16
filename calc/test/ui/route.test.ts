import { describe, it, expect } from 'vitest'
import { parseRoute, serializeRoute, type Route } from '../../src/ui/route'

describe('parseRoute', () => {
  it('empty hash → calc', () => {
    expect(parseRoute('')).toEqual({ tab: 'calc' })
    expect(parseRoute('#')).toEqual({ tab: 'calc' })
  })
  it('calc with payload', () => {
    expect(parseRoute('#calc?a=h100&m=llama-3.3-70b')).toEqual({ tab: 'calc' })
  })
  it('bare info → list', () => {
    expect(parseRoute('#info')).toEqual({ tab: 'info' })
  })
  it('info model detail', () => {
    expect(parseRoute('#info/model/deepseek-v3'))
      .toEqual({ tab: 'info', detail: { kind: 'model', id: 'deepseek-v3' } })
  })
  it('info sku detail', () => {
    expect(parseRoute('#info/sku/hgx-h100-8'))
      .toEqual({ tab: 'info', detail: { kind: 'sku', id: 'hgx-h100-8' } })
  })
  it('unknown path → calc', () => {
    expect(parseRoute('#bogus')).toEqual({ tab: 'calc' })
  })
})

describe('serializeRoute', () => {
  it('calc → #calc', () => {
    expect(serializeRoute({ tab: 'calc' })).toBe('#calc')
  })
  it('info list', () => {
    expect(serializeRoute({ tab: 'info' })).toBe('#info')
  })
  it('info detail round-trips', () => {
    const r: Route = { tab: 'info', detail: { kind: 'model', id: 'x.y' } }
    expect(parseRoute(serializeRoute(r))).toEqual(r)
  })
})

describe('serializeRoute(calc) preserves an explicit payload', () => {
  it('keeps payload when given', () => {
    expect(serializeRoute({ tab: 'calc' }, 'a=h100')).toBe('#calc?a=h100')
  })
})

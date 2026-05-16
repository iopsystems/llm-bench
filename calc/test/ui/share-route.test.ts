import { describe, it, expect } from 'vitest'
import { calcPayloadFromHash } from '../../src/ui/share'

describe('calcPayloadFromHash', () => {
  it('extracts payload after calc?', () => {
    expect(calcPayloadFromHash('#calc?a=h100&m=x')).toBe('a=h100&m=x')
  })
  it('legacy bare payload (no calc prefix) still works', () => {
    expect(calcPayloadFromHash('#a=h100&m=x')).toBe('a=h100&m=x')
  })
  it('info routes carry no calc payload', () => {
    expect(calcPayloadFromHash('#info/model/deepseek-v3')).toBe('')
    expect(calcPayloadFromHash('#info')).toBe('')
  })
  it('empty hash → empty', () => {
    expect(calcPayloadFromHash('')).toBe('')
    expect(calcPayloadFromHash('#calc')).toBe('')
  })
})

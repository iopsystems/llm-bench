import { describe, it, expect } from 'vitest'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const bin = resolve(here, '..', 'bin', 'llm-calc.mjs')

function run(args: string[]): { stdout: string; stderr: string; status: number } {
  const r = spawnSync('node', [bin, ...args], { encoding: 'utf8', timeout: 30_000 })
  return {
    stdout: r.stdout ?? '',
    stderr: r.stderr ?? '',
    status: r.status ?? 1
  }
}

describe('llm-calc CLI', () => {
  it('list gpus: contains h100 and exits 0', () => {
    const { stdout, status } = run(['list', 'gpus'])
    expect(status).toBe(0)
    expect(stdout).toContain('h100')
  })

  it('calc JSON: produces valid JSON and exits 0', () => {
    const { stdout, status } = run([
      '-g', 'h100', '-V', 'sxm-80', '-m', 'llama-3.3-70b',
      '-p', '2048', '-o', '512', '-c', '1',
      '-w', 'fp16', '-k', 'fp16', '-a', 'fp16'
    ])
    expect(status).toBe(0)
    expect(() => JSON.parse(stdout.trim())).not.toThrow()
    const result = JSON.parse(stdout.trim())
    expect(result).toHaveProperty('memory')
    expect(result).toHaveProperty('perf')
  })

  it('calc table: output contains TTFT and exits 0', () => {
    const { stdout, status } = run([
      '-g', 'h100', '-V', 'sxm-80', '-m', 'llama-3.3-70b',
      '-p', '2048', '-o', '512', '-c', '1',
      '--format', 'table'
    ])
    expect(status).toBe(0)
    expect(stdout).toContain('TTFT')
  })
})

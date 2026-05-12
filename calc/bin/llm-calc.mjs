#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const cli = resolve(here, '..', 'src', 'cli.ts')
// Silence DEP0205 — tsx 4.21 still calls module.register(); revisit when tsx ships a fix.
const env = { ...process.env, NODE_OPTIONS: `${process.env.NODE_OPTIONS ?? ''} --disable-warning=DEP0205`.trim() }
const result = spawnSync('npx', ['tsx', cli, ...process.argv.slice(2)], { stdio: 'inherit', env })
process.exit(result.status ?? 1)

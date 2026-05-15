#!/usr/bin/env node
// Verifies the adding-a-model SKILL.md stays in sync with the ModelArch /
// AttentionConfig / ArchitectureConfig schemas in calc/src/engine/types.ts.
// Used by both the Claude PostToolUse hook and the optional git pre-commit hook.

import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = resolve(__dirname, '../..')
const TYPES_PATH = resolve(REPO_ROOT, 'calc/src/engine/types.ts')
const SKILL_PATH = resolve(REPO_ROOT, '.claude/skills/adding-a-model/SKILL.md')

// Pull discriminants from a discriminated union: every `type: 'X'` inside
// the named type alias body. Body ends at the next top-level declaration.
export function extractDiscriminants(typesSrc, typeName) {
  const startRe = new RegExp(`export type ${typeName}\\s*=`)
  const startMatch = typesSrc.match(startRe)
  if (!startMatch) return []
  const startIdx = startMatch.index + startMatch[0].length
  const rest = typesSrc.slice(startIdx)
  const endMatch = rest.match(/\n(export |interface |const |type )/)
  const body = endMatch ? rest.slice(0, endMatch.index) : rest
  return Array.from(body.matchAll(/\btype:\s*'([^']+)'/g)).map(m => m[1])
}

// Top-level field names from an interface body. Nested braces are stripped
// so `nested: { hidden: number }` doesn't leak `hidden`.
export function extractInterfaceFields(typesSrc, interfaceName) {
  const startRe = new RegExp(`export interface ${interfaceName}\\s*\\{`)
  const startMatch = typesSrc.match(startRe)
  if (!startMatch) return []
  const bodyStart = startMatch.index + startMatch[0].length
  let i = bodyStart
  let depth = 1
  while (i < typesSrc.length && depth > 0) {
    if (typesSrc[i] === '{') depth++
    else if (typesSrc[i] === '}') depth--
    i++
  }
  const body = typesSrc.slice(bodyStart, i - 1)
  let cleaned = ''
  let d = 0
  for (const c of body) {
    if (c === '{') d++
    else if (c === '}') d--
    else if (d === 0) cleaned += c
  }
  return Array.from(cleaned.matchAll(/^\s*([a-zA-Z_$][\w$]*)\??:/gm)).map(m => m[1])
}

function findSection(md, headingPrefix) {
  const start = md.indexOf(headingPrefix)
  if (start === -1) return ''
  const after = md.slice(start + headingPrefix.length)
  const endMatch = after.match(/\n## /)
  return after.slice(0, endMatch ? endMatch.index : after.length)
}

export function extractSkillAttentionVariants(skillMd) {
  const section = findSection(skillMd, '## Attention variant')
  return Array.from(section.matchAll(/-\s+\*\*`([a-z][\w-]*)`\*\*/g)).map(m => m[1])
}

export function extractSkillArchitectureVariants(skillMd) {
  const section = findSection(skillMd, '## Architecture')
  return Array.from(new Set(
    Array.from(section.matchAll(/type:\s*'([a-z][\w-]*)'/g)).map(m => m[1])
  ))
}

export function extractSkillModelArchFields(skillMd) {
  const section = findSection(skillMd, '## ModelArch field mapping')
  return Array.from(new Set(
    Array.from(section.matchAll(/`([a-zA-Z_$][\w$]*)`/g)).map(m => m[1])
  ))
}

// Fields intentionally absent from the HF → ModelArch table — they're
// metadata or get their own section in the skill.
const FIELDS_NOT_IN_TABLE = new Set([
  'id', 'name', 'family',
  'attention', 'architecture',
  'paramCount',
])

export function runCheck() {
  const typesSrc = readFileSync(TYPES_PATH, 'utf8')
  const skillMd = readFileSync(SKILL_PATH, 'utf8')

  const failures = []

  const attnTypes = new Set(extractDiscriminants(typesSrc, 'AttentionConfig'))
  const attnSkill = new Set(extractSkillAttentionVariants(skillMd))
  const attnMissing = [...attnTypes].filter(x => !attnSkill.has(x))
  const attnExtra = [...attnSkill].filter(x => !attnTypes.has(x))
  if (attnMissing.length || attnExtra.length) {
    failures.push({ label: 'AttentionConfig discriminants', missing: attnMissing, extra: attnExtra })
  }

  const archTypes = new Set(extractDiscriminants(typesSrc, 'ArchitectureConfig'))
  const archSkill = new Set(extractSkillArchitectureVariants(skillMd))
  const archMissing = [...archTypes].filter(x => !archSkill.has(x))
  const archExtra = [...archSkill].filter(x => !archTypes.has(x))
  if (archMissing.length || archExtra.length) {
    failures.push({ label: 'ArchitectureConfig discriminants', missing: archMissing, extra: archExtra })
  }

  const required = extractInterfaceFields(typesSrc, 'ModelArch')
    .filter(f => !FIELDS_NOT_IN_TABLE.has(f))
  const fieldSkill = new Set(extractSkillModelArchFields(skillMd))
  const fieldMissing = required.filter(f => !fieldSkill.has(f))
  if (fieldMissing.length) {
    failures.push({ label: 'ModelArch fields (in field-mapping table)', missing: fieldMissing, extra: [] })
  }

  return failures
}

function main() {
  const failures = runCheck()
  if (failures.length === 0) {
    console.log('✓ adding-a-model SKILL.md in sync with types.ts')
    process.exit(0)
  }
  console.error('✗ adding-a-model SKILL.md is out of sync with types.ts:\n')
  for (const f of failures) {
    console.error(`  ${f.label}:`)
    if (f.missing.length) console.error(`    In types.ts but not in SKILL.md: ${f.missing.join(', ')}`)
    if (f.extra.length)   console.error(`    In SKILL.md but not in types.ts: ${f.extra.join(', ')}`)
  }
  console.error(`\n  Schema: ${TYPES_PATH}`)
  console.error(`  Skill:  ${SKILL_PATH}`)
  console.error('\nUpdate both so the skill stays an accurate guide for future contributors.')
  process.exit(2)
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main()
}

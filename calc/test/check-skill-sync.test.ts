import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  extractDiscriminants,
  extractInterfaceFields,
  extractSkillAttentionVariants,
  extractSkillArchitectureVariants,
  extractSkillModelArchFields,
  // @ts-expect-error — plain .mjs, no .d.ts
} from '../../.claude/hooks/check-skill-sync.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = resolve(__dirname, '../..')

describe('extractDiscriminants', () => {
  it('parses a TS union of { type: "X" } literals', () => {
    const src = `
export type Foo =
  | { type: 'alpha' }
  | { type: 'beta'; extra: number }
export interface Other { x: number }
`
    expect(extractDiscriminants(src, 'Foo')).toEqual(['alpha', 'beta'])
  })

  it('returns [] when the named type is absent', () => {
    expect(extractDiscriminants('', 'Foo')).toEqual([])
  })

  it('stops at the next top-level declaration', () => {
    const src = `
export type A = | { type: 'first' }
export type B = | { type: 'second' }
`
    expect(extractDiscriminants(src, 'A')).toEqual(['first'])
  })
})

describe('extractInterfaceFields', () => {
  it('returns top-level field names only', () => {
    const src = `
export interface Bar {
  id: string
  count?: number
  nested: { hidden: number }
  arr: Array<{ x: number }>
}
`
    expect(extractInterfaceFields(src, 'Bar')).toEqual(['id', 'count', 'nested', 'arr'])
  })

  it('returns [] for unknown interfaces', () => {
    expect(extractInterfaceFields('', 'Bar')).toEqual([])
  })
})

describe('extractSkillAttentionVariants', () => {
  it('parses bullet list of **`name`** items in the Attention section', () => {
    const md = `
## Attention variant — when to use which

- **\`full\`** — vanilla MHA/GQA
- **\`sliding\`** — sliding window
- **\`mla\`** — DeepSeek V2/V3

## Next section
`
    expect(extractSkillAttentionVariants(md)).toEqual(['full', 'sliding', 'mla'])
  })
})

describe('extractSkillArchitectureVariants', () => {
  it('extracts `type: "X"` from the Architecture section', () => {
    const md = `
## Architecture (dense vs MoE)

\`\`\`typescript
architecture: { type: 'moe', ... }
\`\`\`
Otherwise: \`architecture: { type: 'dense' }\`.
`
    expect(extractSkillArchitectureVariants(md).sort()).toEqual(['dense', 'moe'])
  })
})

describe('integration: current files are in sync', () => {
  const typesSrc = readFileSync(resolve(REPO_ROOT, 'calc/src/engine/types.ts'), 'utf8')
  const skillMd = readFileSync(
    resolve(REPO_ROOT, '.claude/skills/adding-a-model/SKILL.md'), 'utf8'
  )

  it('AttentionConfig discriminants — types.ts matches SKILL.md', () => {
    const inTypes = new Set<string>(extractDiscriminants(typesSrc, 'AttentionConfig'))
    const inSkill = new Set<string>(extractSkillAttentionVariants(skillMd))
    expect([...inTypes].sort()).toEqual([...inSkill].sort())
  })

  it('ArchitectureConfig discriminants — types.ts matches SKILL.md', () => {
    const inTypes = new Set<string>(extractDiscriminants(typesSrc, 'ArchitectureConfig'))
    const inSkill = new Set<string>(extractSkillArchitectureVariants(skillMd))
    expect([...inTypes].sort()).toEqual([...inSkill].sort())
  })

  it('Every required ModelArch field is mentioned in the SKILL.md field table', () => {
    const FIELDS_NOT_IN_TABLE = new Set([
      'id', 'name', 'family',
      'attention', 'architecture',
      'paramCount',
    ])
    const required = (extractInterfaceFields(typesSrc, 'ModelArch') as string[])
      .filter(f => !FIELDS_NOT_IN_TABLE.has(f))
    const inSkill = new Set<string>(extractSkillModelArchFields(skillMd))
    const missing = required.filter(f => !inSkill.has(f))
    expect(missing).toEqual([])
  })
})

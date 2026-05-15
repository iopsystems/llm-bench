# Shared git hooks

Hooks checked into the repo so they can be shared. Not active by default — enable per clone:

```bash
git config core.hooksPath .githooks
```

To disable, unset the config or reset it to `.git/hooks`:

```bash
git config --unset core.hooksPath
```

## Hooks

- **`pre-commit`** — if the commit touches `calc/src/engine/types.ts` (the model schema) or `.claude/skills/adding-a-model/SKILL.md`, runs the skill-sync check at `.claude/hooks/check-skill-sync.mjs`. Blocks the commit on drift.

The same check is invoked by:
- The Claude `PostToolUse` hook in `.claude/settings.json` — fires inside agent sessions.
- The vitest integration test in `calc/test/check-skill-sync.test.ts` — catches drift in CI even when neither hook is enabled.

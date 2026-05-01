---
name: apply-patch-edits
description: Use apply_patch-only edits for AGENTS.md + canonical skills/rules to avoid cross-agent conflicts.
---

# apply-patch-edits

## When to use

Use this skill whenever you need to edit any of:

- `AGENTS.md`
- Canonical agent rules: `artefacts/rules/**`
- Canonical agent skills: `artefacts/skills/**/SKILL.md`

These files are high-churn coordination surfaces; in multi-agent work, in-place manual edits tend to create
cross-conflicts and ambiguous provenance.

## Rules

1. Make changes only via the `apply_patch` tool (atomic, reviewable diffs).
2. Do not edit receiver copies under `.codex/`, `~/.codex/`, `.cursor/`, `.opencode/` directly.
3. After editing canonical assets, sync receivers so installs and tooling stay consistent:

~~~bash
uv run scripts/build_artefacts.py
uv run scripts/sync_agent_assets.py
~~~

If you changed only canonical skills and want a faster sync of receiver skill folders:

~~~bash
uv run scripts/sync_skills_from_artefacts.py --prune
~~~

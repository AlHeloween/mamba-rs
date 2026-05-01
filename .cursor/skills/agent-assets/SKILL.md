---
name: agent-assets
description: Maintain canonical artefacts and install agent receiver scaffolds (.cursor/.codex/~/.codex/.opencode).
---

# agent-assets

## Canonical sources

- Rules: `artefacts/rules/`
- Skills: `artefacts/skills/`

Agent folders are receivers (safe to delete): `.cursor/`, `.codex/`, `~/.codex/`, `.opencode/`.

## Workflow

1. Edit canonical assets under `artefacts/rules/` and/or `artefacts/skills/`.
2. Regenerate derived artefacts and scaffolds:

~~~bash
uv run scripts/build_artefacts.py
~~~

3. Install scaffolds into receivers:

~~~bash
uv run scripts/sync_agent_assets.py
~~~

## Targets

- Install only one receiver:
  - `uv run scripts/sync_agent_assets.py --targets opencode`

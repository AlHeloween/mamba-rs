---
name: patch-tool
description: Apply apply_patch-format patches via adm with ADID backups and per-file ledgers.
---

# patch-tool (adm wrapper for apply_patch)

Use this when you need to apply an `apply_patch`-format patch file in a way that still creates ADID rotated backups and per-file JSONL ledgers.

## Command

- Apply a patch file: `tools/adm.exe --patch-tool <patch_file>` (or `uv run adm --patch-tool <patch_file>` in a repo checkout).
- Dry-run (no writes): `tools/adm.exe --dry-run --patch-tool <patch_file>`

## Notes

- This wrapper calls the bundled `apply_patch.exe` and pre-creates rotated backups for any existing target files.
- It emits per-file entries to `<file>.adid.log.jsonl` with `"command": "--patch-tool"` so you can audit what changed.
- Patch files must start with `*** Begin Patch` and end with `*** End Patch`, with file operations like `*** Update File: ...`, `*** Add File: ...`, and `*** Delete File: ...`.


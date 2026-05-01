---
name: adm-exe
description: Use the ADID Update Manager (adm) executable for declarative updates, verify-all, rollback, and templates.
---

# ADID Update Manager (adm) executable

Use this skill when performing declarative file updates, verification, rollback, or template workflows in a project that uses the ADID framework.

## Critical: Do not create XML descriptors from scratch

**Always use adm commands to get templates and fill them.** Do not hand-craft `updates/*.xml` or invent `<update_md5_*>`, `<content_md5_*>`, or mode syntax from memory.

1. **First:** Run `tools/adm --help` (or `tools/adm.exe --help` on Windows; or `uv run adm --help` when tools/adm not present) to see all commands.
2. **To add a new update:** Create template (`tools/adm.exe --template/--tpl <NAME> [output_dir]` - Generate a timestamped XML descriptor template under `./updates/` by default (use `[TIMESTAMP]_[short_semantic_dominant].xml`; legacy `_update_scaffold.xml` is deprecated). Templates: `all`, `replace`, `overwrite`, `create`, `insert`, `delete`, `pattern-rule`, `binary-overwrite`, `binary-hex-replace`, `refactor-replace-function`), edit the generated descriptor in `updates/` (prefer using the agent `apply_patch` tool for XML edits), then apply (`--apply`). To **replay history** (inspect descriptors in chronological order; information-only by default): use `--replay-updates [dir] [--until TIMESTAMP] [--limit N]` (add `--unified-diff` for exact hunks; add `--execute --workdir DIR --confirm-execute APPLY_IN_WORKDIR_ONLY` to apply only inside an isolated copy). Do not write XML descriptors from scratch. Multiple backups per file are by design and an advantage (rollback, traceability; project stays manageable).
3. **Never** write a new descriptor XML from scratch; you will get tags, MD5, or modes wrong. Use the template, then edit.

ADID workflow and communication rules are in `docs/ADID_Framework_15_3.md`. Prefer that document for structure and epistemic markers.

---

## How to invoke

**Use `tools/adm` (or `tools/adm.exe` on Windows) when the project has it; otherwise `uv run adm`.**

- **Primary:** `tools/adm` (Unix) or `tools/adm.exe` (Windows) when the project has adm installed there (e.g. after `--install-adid-rules`). That executable is stable; if you edit the tool with adm and there is an error, you cannot run adm anymore and the toolchain breaks—the copy in `tools/` avoids that.
- **Fallback:** `uv run adm` with the same subcommands when tools/adm is not present (e.g. in the adm repo before install).

**Rule:** Use `tools/adm` when present—same as AGENTS.md; avoids misunderstanding and toolchain break when editing the tool with adm.

---

## Commands and when to use them

Invoke as: `tools/adm <command>` (or `tools/adm.exe <command>` on Windows; or `uv run adm <command>` when tools/adm not present).

| Command | What it does | When to use |
|--------|----------------|-------------|
| `--help` | Print all commands and options. | **Always run first** if unsure. Do not guess CLI syntax. |
| `--template/--tpl <NAME> [output_dir]` | Generates a timestamped XML descriptor template under `./updates/` by default (uses `[TIMESTAMP]_[short_semantic_dominant].xml`; legacy `_update_scaffold.xml` is deprecated). Templates: `all`, `replace`, `overwrite`, `create`, `insert`, `delete`, `pattern-rule`, `binary-overwrite`, `binary-hex-replace`, `refactor-replace-function`. | **Whenever you need a new update descriptor.** Edit the generated file; do not create XML descriptors from scratch. |
| `--apply <updates.xml>` | Applies all update blocks in the descriptor (atomic writes, backups, ledger). | After editing a templated or existing descriptor; use on the path you edited (e.g. `updates/20260202T040632_update_scaffold.xml`). |
| `--replay-updates [dir] [--until TIMESTAMP] [--limit N]` | Inspects all `*.xml` descriptors in `dir` (default `updates/`) in **chronological order** and prints diff-like intent extracted from XML (no writes). Add `--unified-diff` for exact unified diff hunks (slower). Add `--execute --workdir DIR --confirm-execute APPLY_IN_WORKDIR_ONLY` to apply only inside an isolated copy. | To understand what changed and why, and (rarely) to reproduce state in a dedicated isolated directory. |
| `--compute-md5 <payload_file>` | Prints canonical MD5 and stripped size for a payload. | **Debugging/testing only.** Normal workflow should not manually compute MD5; use `--fix-xml` / `--verify-all-fix-xml` or just `--apply` (auto-normalizes descriptor md5/size + tag names). |
| `--fix-xml <updates.xml> [output.xml]` | Normalizes descriptors and recalculates md5/size tags; can write to a second file. | When descriptor tags are wrong or after manual edits; run before `--apply` if you changed payloads. |
| `--verify-all [root]` | Verifies integrity/syntax under the given root; writes reports to `logs/verify_report_*.{json,md}`. | After applies or to audit; use `src tests adid_tests` for a clean report. |
| `--verify-all-fix-xml` | Same as `--verify-all` but also rewrites descriptor tags in place. | When verify reports MD5/tag issues in descriptors; fixes tags then re-verify. |
| `--rollback <file>` | Restores the file from the latest backup (per-file). | When a single file is corrupted or wrong; do **not** use git restore for one bad edit. |
| `--list-backups <file>` | Lists backups for the file (timestamp, size, MD5, semantics). | To see what rollback will restore or to inspect history. |
| `--list-diff <file> [N]` | Shows unified or hex diffs against up to N backups. | To compare current vs previous versions before rollback. |
| `--emit-boot-log [roots...]` | Runs pytest then verify-all; writes `boot_test.log` and verify reports. | One-shot “boot” check: tests + verification. |
| `--install-adid-rules [target_dir]` | Installs Cursor rules, adm-exe skill, framework doc, and `adm.exe` into the target project. | To onboard a project with ADID (creates `.cursor/`, `docs/`, `tools/adm.exe`). |
| `--snapshot-context [snapshot.json]` | Captures repo context (git head, uv.lock, versions) to a JSON file. | Before major changes; then use `--preflight` to detect drift. |
| `--preflight [snapshot.json] [--strict]` | Compares current tree to snapshot; `--strict` fails on drift. | After changes to ensure nothing unintended changed. |
| `--clean [root] [--all]` | Removes manifests, rotated backups, demo bundles. With `--all`, also removes baseline snapshots (`*.baseline`), JSONL ledgers (`*.adid.log.jsonl`), and strips trailing `ADID_ROLLBACK`/`SDID_ROLLBACK` blocks from text files. | To tidy artifacts under the given root (use `--all` only when you want a full cleanup). |
| `--rg <pattern> <replacement> <file> [-- flags]` | ripgrep-based replacement with backup and ledger. | When you need regex replace but want backups/rollback; use via adm, not raw rg. |
| `--sed <script> <file> [-- flags]` | Runs GNU sed with backup and ledger. | When you need sed but want backups/rollback; use via adm, not raw sed. |
| `--patch-tool <patch_file>` | Applies an `apply_patch`-format patch file with backups + per-file ledger entries. | When you want apply_patch-style edits but still want ADID backups and ledgers. |
| `--move <src> <dst> [--execute] [--no-fix-updates] [--no-fix-refs] [--updates-dir DIR] [roots...]` | Move/rename a file and rewrite occurrences of the old path in `updates/` and other roots (literal, handles both `/` and `\\`). | When you rename/move a file and want to keep `updates/` descriptors + docs/code references consistent. **Dry-run by default**; pass `--execute` to perform writes. |
| `--cmd-runner <args...>` | Pass-through helper to execute `cmd_runner` with the given args (Windows-only). | When you need to start/list/status/tail cmd_runner runs but want it recorded in the same adm progress-log cycle. |
| `--rag settings|list|index|docs|status|delete ...` | Manage local RAG indexes (sqlite) using `adm.json`. | When you want indexed querying of code/docs with the local `sentence_transformers` + `BAAI/bge-m3` embedder. In bundled installs, `adm.exe` forwards this to `adm-rag.exe`. |
| `--query <index_name> <request...>` | Query a RAG index and print top hits (file + line ranges). | When you want an exact, line-referenced retrieval result. In bundled installs, `adm.exe` forwards this to `adm-rag.exe`. |
| `--mcp` / `--mcp-http [host] [port]` | Run adm as an MCP server (stdio or HTTP). Startup validates local embedder readiness; `initialize` reports the resolved RAG DB path and configured embedding backend/model/device. | When you want to expose RAG tools to an MCP-capable client or run it as a local service. In bundled installs, `adm.exe` forwards this to `adm-rag.exe`. |
| `--log-insight <message>` | Appends a timestamped entry to `insights.md`. | Optional logging of a decision or finding. |
| `--check-tools` | Checks that rg, sed, astgrep, tree-sitter are available. | To diagnose missing tools. |
| `--sync-astgrepignore` | Mirrors `.gitignore` into `.astgrepignore`. | When .gitignore changed and you want astgrep to match. |
| `--log-progress [path]` | Appends progress entries (default `_progress_log.md`). | To record that a command ran and its outcome. |
| `--dry-run` | Simulates actions without writing to disk. | With `--apply` or other mutating commands to preview. |

---

## Recommended workflow for a new update

Use `tools/adm` (or `tools/adm.exe` on Windows) when the project has it; otherwise `uv run adm`.

1. Run `tools/adm --help`.
2. Run `tools/adm --template all` (or a specific template like `replace`, `overwrite`, `create`, `insert`, `delete`, `pattern-rule`, `refactor-replace-function`) → creates a timestamped descriptor under `updates/`.
3. Edit that file: set `<file>`, `<mode>`, payload in `<content_md5_*>`, etc. **Do not manually compute MD5**; leave placeholders and let adm normalize metadata.
4. Optionally run `tools/adm --fix-xml updates/<that_file>.xml` to normalize md5/size + tag names (useful for debugging/validation).
5. Run `tools/adm --apply updates/<that_file>.xml` (use `--dry-run` first if you want a preview).
6. Run `tools/adm --verify-all src tests adid_tests` (or chosen roots) to confirm no regressions.

To **replay history** (inspect descriptors in order): run `tools/adm --replay-updates` (or `tools/adm --replay-updates updates --until 20260202T120000 --limit 5`). Add `--unified-diff` for exact hunks; use `--execute --workdir DIR --confirm-execute APPLY_IN_WORKDIR_ONLY` only when you intentionally want to apply inside an isolated copy.

All mutations create backups and ledger entries; use `--rollback <file>` to undo a single file. Do not use git restore for one bad edit—use rollback.

**From now on, use this toolset:** template → edit the descriptor file → apply. **Use `tools/adm`** (or `tools/adm.exe` on Windows) when the project has it—same as AGENTS.md; stable executable, toolchain stays intact if you edit the tool with adm and hit an error. Multiple backups per file are intentional and beneficial (rollback, traceability); the project stays manageable while saving time and giving a clear, testable way to use the tool across different areas.

Bundled binary split note:
- `adm.exe` is now the lightweight base CLI.
- `adm-rag.exe` carries the local RAG/MCP runtime.
- Operators can still use `tools/adm.exe --rag ...` / `--query` / `--mcp*`; the base binary forwards those commands to `tools/adm-rag.exe`.

## Building Executables

To build standalone executables from the repo:

~~~bash
# Build all executables with Nuitka (recommended)
uv run scripts/_build.py

# Build specific components
uv run scripts/_build.py --adm                    # Only adm.exe
uv run scripts/_build.py --adm --adm-rag          # adm.exe + adm-rag.exe

# Build modes
uv run scripts/_build.py --fast                   # Quick build (faster)
uv run scripts/_build.py --release                # Release build (optimized)

# Legacy PyInstaller (bundles torch, ~200MB+)
uv run scripts/_build.py --backend=pyinstaller

# Get help
uv run scripts/_build.py --help
~~~

**Build backends:**
- **Nuitka** (default): Smaller executables (~10-20MB), PyTorch detected at runtime
- **PyInstaller** (legacy): Larger executables (~200MB+), torch bundled

**RAG Prerequisites (Nuitka builds):**
Nuitka executables do not bundle PyTorch. Install it separately:
~~~bash
# CPU version
uv pip install torch sentence-transformers

# CUDA version (for GPU acceleration)
uv pip install torch sentence-transformers --index-url https://download.pytorch.org/whl/cu121
~~~


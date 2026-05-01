---
name: rag
description: Index/query local repositories using adm RAG (adm.json + sqlite) and a local BAAI/bge-m3 embedder.
---

# rag (adm RAG)

This skill covers the `adm --rag ...` and `adm --query ...` commands.

## Requirements

- `adm.json` must exist in the launch folder (adm auto-creates it with defaults if missing).
- The local embedder must be loadable from `rag.embed.*` settings.
- Default embedder: `sentence_transformers` + `BAAI/bge-m3`.
- **External Python Runtime:** `adm-rag.exe` is a small Nuitka helper that attaches `site-packages` from a compatible Python 3.13 installation found on PATH, or from `ADID_RAG_PYTHON`.
- **PyTorch Runtime Detection:** the helper does not bundle the heavy ML stack. You must install the RAG packages into that Python separately:
  - CPU: `uv pip install torch sentence-transformers`
  - CUDA: `uv pip install torch sentence-transformers --index-url https://download.pytorch.org/whl/cu121`
- At indexing start, the system displays external runtime source, detected packages, and CPU/GPU execution choice:
  ~~~
  [RAG] Embeddings backend=sentence_transformers model=BAAI/bge-m3
  [RAG] Requested device: auto
  [RAG] External Python: C:\Python313\python.exe (version=3.13.2, source=python)
  [RAG] Package torch: 2.7.0+cu128 [required]
  [RAG] Package sentence-transformers: 5.3.0 [required]
  [RAG] PyTorch 2.7.0+cu128 detected
  [RAG] CUDA available: Yes
  [RAG]   Device: NVIDIA GeForce RTX 4090
  [RAG] Execution device: GPU (cuda:0)
  ~~~

## Commands

- Show settings: `tools/adm.exe --rag settings`
- List indexes: `tools/adm.exe --rag list`
- Index: `tools/adm.exe --rag index <index_name> <root1> [root2 ...]`
- Show indexed docs: `tools/adm.exe --rag docs <index_name> [limit]`
- Status: `tools/adm.exe --rag status <index_name>`
- Delete: `tools/adm.exe --rag delete <index_name>`
- Query: `tools/adm.exe --query <index_name> <request...>`
- Direct helper equivalents: replace `tools/adm.exe` with `tools/adm-rag.exe` when you want the explicit RAG/MCP binary

Defaults:
- `tools/adm.exe --rag index` => `index_name=<current folder name>`, `roots=['.']`
- `tools/adm.exe --rag delete` => `index_name=<current folder name>`

Bundled binary split note:
- `adm.exe` is the small front-end binary.
- `adm-rag.exe` contains the local embedding and MCP runtime.
- In installed/bundled layouts, `tools/adm.exe --rag ...` and `tools/adm.exe --query ...` forward to `tools/adm-rag.exe`.

Index runs now print live progress:
- initial `[RAG] indexing ...` banner,
- `[RAG][PROGRESS] stage=... processed=... indexed_docs=... skipped_docs=... error_docs=... indexed_chunks=... queued_chunks=...` (plus `file_chunks=...` on per-record indexing events),
- final `[RAG] index='...' docs=... chunks=...` summary.

## What gets indexed

- Native text/code/web files from `rag.include_globs`
- Force-included paths from `rag.add_patterns`
- Hard excludes from `rag.exclude_patterns`
- Structured ADID history docs (`adid://update/...`, `adid://trace/...`, `adid://ledger/...`)
- Structured `cmd_runner` run docs (`log://cmd_runner/<run_id>#summary|stdout|stderr|inbox`) when `rag.cmd_runner_logs_enabled=true`

Code chunks also carry:
- Tree-sitter structural tags: `symbol_kind:*`, `symbol_name:*`
- (Astgrep tags were removed from indexing for performance; `astgrep_tags_enabled` config flag is ignored)

Use `tools/adm.exe --rag settings` to inspect the effective `include_globs`, `add_patterns`, `exclude_globs`, `exclude_patterns`, and `cmd_runner_logs_enabled` values.

## Embedding + retrieval model

- Default embedder:
  - `rag.embed.backend = sentence_transformers`
  - `rag.embed.model = BAAI/bge-m3`
  - `rag.embed.device = auto`
  - `rag.embed.batch_size = 16`
- Hybrid retrieval:
  - full-vector similarity
  - SQLite FTS5
  - dual-quaternion signature shortlist/rerank using the same low-frequency packing style as `docs/dual-quaternion-truth-2.py`
- Suggested mental model:
  - embeddings + DQ = fast semantic/structural shortlist
  - ADID = time-resolved and semantically grounded reasoning over the shortlist
  - use DQ agreement as a structure-consistency signal before deep analysis
- Key tuning knobs:
  - `rag.vector_top_k`, `rag.fts_top_k`, `rag.dq_top_k`
  - `rag.weight_vector`, `rag.weight_fts`, `rag.weight_dq`
  - `rag.dq_projection_dim`, `rag.dq_keep_rows`, `rag.dq_keep_cols`, `rag.dq_include_energy`

## Common queries (smoke)

After indexing docs, these are good “first queries” to validate wiring:

- `tools/adm.exe --query <index_name> "ADIDInstaller.json"`
- `tools/adm.exe --query <index_name> "codex mcp add adid_rag"`
- `tools/adm.exe --query <index_name> "log://cmd_runner"`
- `tools/adm.exe --query <index_name> "symbol_name:Greeter"`

## Windows-specific

- Set repo-local cache/temp at start of work:
  - cmd.exe: `call scripts\\dev_env_windows.cmd`
  - PowerShell: `. .\\scripts\\dev_env_windows.ps1`

## Linux-specific

- For service deployments, prefer MCP HTTP mode (`adm --mcp-http`) with a systemd unit whose `WorkingDirectory` is the folder that contains `adm.json`.

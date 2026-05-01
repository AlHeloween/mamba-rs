---
name: adm-mcp-service
description: Run adm as an MCP server (stdio or HTTP) and install it as a service on Windows or Linux.
---

# adm-mcp-service

`adm` can run an MCP server that exposes RAG tools.

## Modes

- **Stdio (spawned by a client):** `tools/adm.exe --mcp` or direct helper `tools/adm-rag.exe --mcp`
- **HTTP (service-friendly):** `tools/adm.exe --mcp-http [host] [port]` or direct helper `tools/adm-rag.exe --mcp-http [host] [port]` (default: `127.0.0.1 7990`, endpoint: `POST /mcp`)

Both require `adm.json` in the launch folder.
Startup fails fast unless the configured local embedder can be loaded.
After a successful MCP `initialize`, the server reports the resolved RAG DB path and configured embedding backend/model/device.

Bundled binary split note:
- `adm.exe` is the lightweight front-end and forwards MCP/RAG commands to `adm-rag.exe`.
- For service definitions and client wiring, using `adm-rag.exe` directly is preferred because it avoids the extra forwarding hop.

## Wire into Codex (MCP client)

Codex can launch `adm` as a stdio MCP server and call the RAG tools through it.

- Add server (writes to `~/.codex/config.toml`):
  - `codex mcp add adid_rag --cwd <repo_root> -- <abs_path_to_adm_rag.exe> --mcp`
- Example (Windows):
  - `codex mcp add adid_rag --cwd D:\\zPython\\ADID_Python -- D:\\zPython\\ADID_Python\\tools\\adm-rag.exe --mcp`
- Verify:
  - `codex mcp list`
  - `codex mcp get adid_rag`

## Windows (service)

Install (Admin PowerShell):

- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\install_adm_mcp_service_windows.ps1 -RepoRoot <repo> -Port 7990`

Service target:
- point the service at `tools\\adm-rag.exe --mcp-http ...` when you want the direct helper entrypoint
- `tools\\adm.exe --mcp-http ...` still works because it forwards to the helper

Check:

- `sc.exe query ADID_ADM_MCP`

## Linux (systemd service)

Install:

- `sudo ./scripts/install_adm_mcp_service_linux.sh /abs/repo_root 7990`

Service target:
- prefer `/abs/repo_root/tools/adm-rag.exe --mcp-http ...` when using the packaged helper directly

Check:

- `systemctl status adid-adm-mcp.service --no-pager`

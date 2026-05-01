---
name: cmd-runner
description: Run interactive Windows commands safely via cmd_runner (ConPTY-only) with per-run logs and an inbox bridge.
---

# cmd-runner

Use this skill when a command may be:
- long/noisy,
- interactive (prompts, TUIs),
- crash-prone or likely to destabilize the agent when run directly.

## When to use cmd_runner

**Use cmd_runner for:**
- Long-running builds: `cargo build --release`, `msbuild`, `make`, `gradle build`
- Package installs: `npm install`, `pip install -r requirements.txt`, `uv sync`
- Test suites: `pytest`, `cargo test`, `npm test`, `mvn test`
- Interactive TUIs: `htop`, `ncurses` apps, installers with prompts
- Crash-prone or unstable commands
- Commands that produce thousands of lines of output

**Do NOT use cmd_runner for:**
- Quick checks: `ls`, `dir`, `git status`, `echo`, `cat`
- Simple file operations: `cp`, `mv`, `rm` (for small operations)
- Commands that complete in <1 second
- Commands you need to see output from immediately

## Why use cmd_runner?

1. **Prevents context flooding** — Long output doesn't consume your agent context window
2. **Isolates crashes** — If the subprocess crashes, your agent session stays stable
3. **Visible window control** — Subprocess opens in a minimized window (SW_SHOWMINNOACTIVE); you can always find it in the taskbar and close it if needed
4. **Cross-shell safety** — Paths are normalized to survive bash → cmd → PowerShell → Python transformations
5. **Programmatic input** — Send keystrokes/text via the inbox bridge without terminal hacks

## Process window behavior

All subprocesses created by cmd_runner open with `SW_SHOWMINNOACTIVE`:
- Window is minimized (not visible on screen initially)
- Window appears in the taskbar
- Window is NOT activated (doesn't steal focus)
- You can always find and close the window if the process hangs
- This is safer than hidden/headless processes (which can't be interacted with)

## What cmd_runner is (current)

- Windows-only, ConPTY-only, serverless (no background server, no TCP control plane).
- Root-only policy:
  - Repo checkout: `cmd_runner.exe` at repo root (Delphi binary).
  - Release bundle: run from the bundle root (cwd contains `cmd_runner.exe`).
- Logs are written to: `logs/cmd_runner/<run_id>/`
- Programmatic input bridge: append JSONL messages to `logs/cmd_runner/<run_id>/inbox.jsonl`.

## How to run it (recommended)

- Repo/dev:
  - `.\cmd_runner.exe ...` (Delphi binary at repo root)
- Release bundle:
  - `cmd_runner.exe ...` (preferred; no `uv` required)
- Via adm (integration; keeps a single progress-log cycle):
  - `tools/adm.exe --cmd-runner <cmd_runner args...>`
  - Example: `tools/adm.exe --cmd-runner start -- <command ...>`

## Core workflow

1) Start an interactive run (spawns a new window; interactive session is hosted there):
- `.\cmd_runner.exe start -- <command ...>`
  - Prints `run_id` and `inbox=` path in the *current* terminal.
- `.\cmd_runner.exe start --shell cmd -- <command ...>`
  - Explicitly wrap all args in a cmd.exe batch file.
- `.\cmd_runner.exe start --shell pwsh -- <command ...>`
  - Explicitly wrap all args in a PowerShell script.
- `.\cmd_runner.exe start --shell bash -- <command ...>`
  - Explicitly wrap all args in a bash/sh script (bash must be on PATH or use full path).
  Without `--shell`, the tool auto-detects `cmd /c`, `powershell -c`, `python -c`, and `bash -c` from the command argv.

2) Check status first (from the current terminal):
- `.\cmd_runner.exe list`
- `.\cmd_runner.exe status <run_id>`
  - Use `status` first to confirm the program is still alive and did not crash before reading output.

3) Tail output (from the current terminal):
- Repo checkout: `.\cmd_runner.exe tail <run_id>` (repo root)
- Release bundle: `cmd_runner.exe tail <run_id>` (bundle root)
  - Start with non-follow `tail` for a compact snapshot.
  - Add `--follow` only when live streaming is needed.
  - Prefer repeated `status`/non-follow `tail` checks over ad hoc shell sleeps; keep delay/wait handling inside the cmd_runner workflow.

4) Inject input programmatically (bridge):
- Append JSONL to: `logs/cmd_runner/<run_id>/inbox.jsonl`
- Built-in (preferred):
  - `.\cmd_runner.exe send <run_id> --keys "ctrl+c,ctrl+d,ENTER"`
  - `.\cmd_runner.exe send <run_id> --hex 03` (raw byte: Ctrl+C)
  - `.\cmd_runner.exe send <run_id> --text "whoami" --crlf` (text + CRLF)
  - `--keys` tokens: `LEFT`, `RIGHT`, `UP`, `DOWN`, `HOME`, `END`, `INSERT`, `DELETE`, `TAB`, `ESC`, `ENTER`, `BACKSPACE`, `ctrl+a`..`ctrl+z`, `ctrl+[`, `ctrl+\`, `TEXT:<text>`, `CHAR:<char>`, `HEX:<hex>`.

5) Stop (serverless terminate):
- `.\cmd_runner.exe stop <run_id> --reason "done"`
  - Writes `logs/cmd_runner/<run_id>/stop_request.json`; the hosting cmd_runner watches for it and terminates the Job Object.

### SSH session example

- Start: `.\cmd_runner.exe start -- ssh root@host` (interactive, enter password manually)
- Send command: `.\cmd_runner.exe send <run_id> --text "uptime; df -h" --crlf`
- Interrupt (Ctrl+C): `.\cmd_runner.exe send <run_id> --keys ctrl+c`
- End session (Ctrl+D): `.\cmd_runner.exe send <run_id> --keys ctrl+d`

Notes:
- `add_crlf` defaults to `false` (no implicit Enter). Use `ENTER` in `keys` or `--crlf` in the helper.
- Supported terminal hosts are `conhost` and `wt`.
- Omit `--terminal` unless you explicitly need to force a host.

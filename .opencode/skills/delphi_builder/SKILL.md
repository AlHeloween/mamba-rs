---
name: delphi_builder
description: Build Delphi (VCL/FMX) projects from the command line with MSBuild, including environment initialization (MSVC + rsvars) and packaging notes for ADIDInstaller.
---

# delphi_builder

## Goal

Provide a repeatable Windows CLI workflow for building Delphi `.dproj` projects using MSBuild without touching IDE options.

## Commands added to `adm`

- `tools/adm.exe --init-msvc [out.cmd]`
  - Generates `tools/init_msvc.cmd` (default) that first respects an already-initialized `cl` + `msbuild` environment, otherwise calls Visual Studio `VsDevCmd.bat`.
- `tools/adm.exe --init-delphi [out.cmd]`
  - Generates `tools/init_delphi.cmd` (default) that resolves Delphi from `adm.json` (`delphi.bds`) or from `where dcc64` / `where dcc32`, then calls `rsvars.bat` / `rsvars64.bat`.

## Typical build flow (Windows)

1. Init MSVC:
   - `cmd.exe`: `call tools\\init_msvc.cmd`
   - PowerShell (env persists only when dot-sourced): `. .\\tools\\init_msvc.ps1`
2. Init Delphi:
   - `cmd.exe`: `call tools\\init_delphi.cmd` (or `call tools\\init_delphi.cmd Win64`)
   - PowerShell (dot-source): `. .\\tools\\init_delphi.ps1 -Platform Win64`
3. Build:
   - Prefer wrapper (auto-generates `.dproj` from `.dpr` when missing):
     - `tools\\build_delphi_msbuild.cmd delphi\\ADIDInstallerFMX.dpr Win32 Release`
     - `tools\\build_delphi_msbuild.cmd delphi\\ADIDInstallerFMX.dpr Win64 Release`
   - PowerShell wrapper:
     - `.\\tools\\build_delphi_msbuild.ps1 -Dpr delphi\\ADIDInstallerFMX.dpr -Platform Win64 -Config Release`

Repo shortcut (FMX installer, Win64 Release):
- `delphi\\build_installer_msbuild.cmd Win64 Release`

Notes:
- Delphi 2009+ commonly uses `/p:config=<ConfigName>` (case-insensitive).
- Delphi 2007 commonly uses `/p:Configuration=<ConfigName>`.

## Library paths / SDK paths (without IDE option edits)

Prefer build-time configuration (project/group build props) instead of editing global IDE options. Strategy examples:
- Use per-repo build scripts that call `rsvars.bat` and set additional environment variables before MSBuild.
- Keep build-time path decisions in versioned files (project-local) so CI and team machines stay aligned.

## Linux64 (WSL2) overview (FMX only)

- VCL cannot target Linux; you need an FMX project.
- Linux64 builds typically require a configured Delphi **Remote Profile** + imported SDK (SDK Manager).
- Use `ADID_DELPHI_PROFILE` (or wrapper arg) to pass the Remote Profile name to MSBuild:
  - `set ADID_DELPHI_PROFILE=my_wsl_profile`
  - `tools\\build_delphi_msbuild.cmd path\\to\\YourFMXApp.dpr Linux64 Release`

Details: `docs\\delphi_linux_wsl2.md`

## FMX cross-platform builds (concept)

FMX targets like Android/iOS/macOS require additional platform SDKs and platform services (vendor tooling). The CLI MSBuild entrypoint is similar (`/p:Platform=<target>`), but environment prerequisites differ by target.

Examples (targets vary by project/Delphi version):
- `Platform=Android`
- `Platform=iOSDevice64` / `Platform=iOSSimulator`
- `Platform=OSX64`
- `Platform=Linux64`

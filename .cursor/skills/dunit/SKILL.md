---
name: dunit
description: Run and maintain Delphi DUnit tests for the ADID installer and related Delphi units.
---

# dunit

## Purpose

Provide a reliable workflow for building and running Delphi DUnit tests (console runner) and interpreting failures.

## Prerequisites

- Delphi toolchain on PATH (at minimum `dcc32`).
- Initialize the environment via `rsvars.bat` for your Delphi version, or use the repo helpers:
  - `call tools\init_msvc.cmd` (optional)
  - `call tools\init_delphi.cmd Win32`

## Commands (repo)

- Build + run all DUnit tests:
  - `delphi\tests\run_tests.cmd`

- Build tests only:
  - `delphi\tests\build_tests.cmd`

## Notes (Win64)

- The repo test scripts build the test runner via MSBuild for `Win64` (see `delphi\tests\build_tests.cmd`).

## Project layout (this repo)

- Test project: `delphi\tests\ADIDInstallerTests.dpr`
- Test units: `delphi\tests\TestInstallerCore.pas`, `delphi\tests\TestEmbeddedAssets.pas`

## How to add a new test

1. Add a new unit `delphi\tests\TestSomething.pas`.
2. Register it in `delphi\tests\ADIDInstallerTests.dpr` (typical DUnit pattern).
3. Re-run `delphi\tests\run_tests.cmd`.

## Notes

- DUnit assertions typically come from `TestFramework` (e.g. `CheckEquals`, `CheckTrue`, `CheckNotNull`).
- Prefer testing pure units (no VCL dependencies) so tests run headless and deterministic.

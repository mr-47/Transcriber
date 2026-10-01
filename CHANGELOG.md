# Changelog

All notable changes to this project are documented here. Earlier releases
predate this file and are not tracked.

## [1.0.0] - 2026-09-30

First stable release. The API, CLI and folder-watcher behaviour that has been
in production use since 0.3.0 is now locked in, with the executability gaps
that a 1.0 actually depends on closed.

### Added

- `install.cmd` / `run.cmd` for Windows, mirroring `install.sh` / `run.sh`:
  venv + dependency install, optional CUDA extra offered after an `nvidia-smi`
  GPU probe, `calls-*` folder creation, and a watcher launcher that forwards
  every argument to `transcriber watch`.
- `AGENTS.md` capturing the project's non-derivable decisions (engine naming,
  lock policy, CUDA fallbacks, folder-workflow invariants, batch/CRLF rules).
- `LICENSE` (MIT), matching the licence declared in `pyproject.toml`.

### Changed

- Version bumped to 1.0.0 across `pyproject.toml` and `__version__` (the CLI
  `--version` and the API `version` read the latter).
- Empty configuration environment variables now behave exactly as unset, so a
  stray `TRANSCRIBER_WHISPER_MODEL=""` no longer resolves to `.` instead of the
  `small` default.
- The one-shot CLI computes per-word timestamps only when `--json` and
  `--words` are both given instead of always, skipping the expensive
  word-level alignment on plain transcript requests.

### Fixed

- CUDA inference could fail with `RuntimeError` on the first `next()` of
  faster-whisper's lazy segments generator without triggering the CPU reload;
  the generator is now consumed inside the guarded region so the fallback
  actually runs (`test_transcribe_reloads_whisper_on_lazy_cuda_error`).
- A write or render failure while finalising a call left the audio stranded in
  `calls-process/` with no report; it now routes to `calls-failed/` with a
  `<name>.txt` error file (`test_result_write_failure_moves_to_failed_with_report`).
- `to_text` raised `IndexError` on an empty-text utterance; `to_text` and
  `to_srt` now skip empty texts.
- `anyio` is declared as a direct dependency (it was imported by the API
  without being listed).
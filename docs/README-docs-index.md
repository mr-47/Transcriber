# Transcriber Documentation

The project README should remain a concise landing page.

Detailed information is preserved in these documents rather than deleted.

## User documentation

- [Installation](installation.md) — requirements, dependency footprint, CUDA GPU setup, first-run behavior
- [Windows support](windows.md) — `install.cmd` / `run.cmd`, CUDA extra, verification status
- [Configuration](configuration.md) — complete environment-variable reference with defaults
- [REST API](api.md) — endpoints, one-process design, readiness, shutdown and security
- [Folder workflow](folder-workflow.md) — `calls-*` state machine, write-in-progress detection, collisions, crash recovery
- [Output formats](output-formats.md) — JSON schema, Markdown/TXT/HTML/SRT behavior
- [Troubleshooting](troubleshooting.md) — common runtime failures and what to check
- [Known limitations](known-limitations.md) — explicit current limitations

## Technical documentation

- [Architecture](architecture.md) — shared pipeline, one-lock serialization, CUDA fallbacks, module layout
- [Speaker diarization](diarization.md) — token gate, tuning knobs, tiny-turn absorption, attribution rules
- [Performance](performance.md) — measurements from a real GPU smoke run
- [Development](development.md) — test suite, coverage, lint, repository layout, release checklist
- [Licensing and attribution](licensing.md) — MIT license, model and dependency terms
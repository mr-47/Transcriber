# Development

## Setup and commands

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"        # GPU machines:  ".[dev,cuda]"

.venv/bin/ruff check src tests           # pinned rules: E4,E7,E9,F,I; line length 120
.venv/bin/python -m pytest --cov=transcriber --cov-report=term-missing -q
```

Note: `python` is often not on `PATH` in this dev environment; use
`.venv/bin/*` (or `python3` for the venv).

## Test suite

- needs no network and no GPU: every test injects a stub engine;
- **185 tests, 99% coverage**, ~5-6 s, offline;
- two `DeprecationWarning`s from `starlette.testclient` (anyio `BlockingPortal`
  alias) are expected third-party noise, not failures.

The only uncovered lines are intentional: the degenerate 999-collision
`RuntimeError` in `_unique_stem` and `_first_free` returning `None` in
`pipeline.py`.

Approach lessons encoded in the suite:

- **Assert at the layer where the value is handed over, not one below.**
  `--language` was once accepted by argparse and dropped before it reached the
  processor; `tests/test_cli.py` now stubs `Transcriber`, `CallFolderProcessor`
  and `watch_folders` and asserts the handover itself.
- **The stub transcriber is the workhorse; keep it honest.** When a call
  signature changes, every stub that records args must learn the new parameter.
- **Coverage hides gaps; mutating the code is the real check.** Break a fix on
  purpose and watch the suite go red before trusting a green run.
- `test_transcribe_reloads_whisper_on_lazy_cuda_error` pins the lazy-generator
  CPU reload; `test_result_write_failure_moves_to_failed_with_report` pins the
  "no stranded audio" property.

## Repository layout

```text
Transcriber/
├── README.md
├── AGENTS.md
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── install.sh / run.sh      (LF)
├── install.cmd / run.cmd    (CRLF — must stay CRLF)
├── calls-inbox/             (gitignored except .gitkeep)
├── calls-process/           (…)
├── calls-failed/            (…)
├── calls-results/           (…)
├── docs/
├── src/transcriber/
│   ├── __init__.py          version; `python -m transcriber` entry
│   ├── __main__.py
│   ├── cli.py               argparse; one-shot / watch dispatch
│   ├── api.py               FastAPI app
│   ├── config.py            environment variables, model resolution
│   ├── core.py              Transcriber engine (ASR + diarization + merge)
│   ├── pipeline.py          CallFolderProcessor, watch_folders
│   ├── merge.py             assign_speakers, build_utterances, absorb_tiny_turns
│   ├── format.py            md / txt / html / srt renderers
│   └── models.py            JSON contract (single source of truth)
└── tests/
    ├── test_api.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_core.py
    ├── test_format.py
    ├── test_merge.py
    └── test_pipeline.py
```

## Verifying a Windows/batch change

Wine 9.0 runs `cmd.exe` well enough to catch real batch bugs, but has no Windows
Python and cannot run 32-bit EXEs, so the installers can only be **parse-tested**:

```bash
wine cmd /c install.cmd --help      # help, exit 0
wine cmd /c install.cmd --bogus     # unknown option, exit 1
wine cmd /c run.cmd                 # ".venv is missing" (no Scripts\ here), exit 1
wine cmd /c install.cmd             # bail at the Python check, exit 1
```

Batch traps that have produced false results:

- `for /f` in Wine runs the `do` body even when the command produced no output
  and does not substitute the loop variable inside a multi-line block — GPU
  detection is a single-line `if not defined GPUNAME set "GPUNAME=%%G"` line.
- `findstr /c` treats its pattern as a literal, so `install.cmd` slices
  `!ARG:~0,1!` with delayed expansion to reject unknown flags.
- Multiple `.cmd` files must stay 100% CRLF (real `cmd.exe` misparses LF-only
  batch, most visibly at `goto`/labels). Verify with Python
  (`data.count("\n")` vs `data.count("\r\n")`), not `awk`. `.gitattributes`
  pins `*.cmd text eol=crlf` and `*.sh text eol=lf`.

## Updating the version

Version lives in exactly two editable places — `pyproject.toml` and
`__version__` in `src/transcriber/__init__.py` — and the CLI `--version`, the
API `version`, and `/docs` all read the `__version__` one. After bumping,
re-run the CLI and `/health` to confirm the string surfaces everywhere, and add
a `CHANGELOG.md` entry.

## Release checklist

The unit suite injects stub engines, so it proves nothing about the real
models. Before tagging a release, run the real pipeline once (network + GPU or
CPU needed):

1. Version consistency — `transcriber --version` and
   `python -c "import transcriber; print(transcriber.__version__)"` both show
   the new version.
2. Real ASR on a short recording with default `small`: the md renders, the JSON
   has `segments`/`utterances`, and `--words` adds word arrays.
3. Folder workflow end to end: drop a file into `calls-inbox/`, run
   `transcriber watch --once --format md,txt,html,srt`, confirm audio + JSON +
   the four formats land in `calls-results/` and inbox/process are empty.
4. API smoke: POST a file to `/transcribe`, hit `/health`, confirm the temp
   upload is cleaned up.
5. Diarization, only with an accepted gated model + token: a two-party
   recording should yield at least two distinct `SPEAKER_N` labels.
6. On a GPU machine, confirm the CUDA path with the `cuda` extra installed logs
   `Preloaded ... libcublas.so.12` and no CPU-reload warning; a Pascal/Maxwell
   card additionally logs the *expected* float16->float32 compute downgrade.
   Without the extra, expect the CPU-reload warning and a working `int8` run.

Real numbers from one such smoke run are in [performance.md](performance.md).
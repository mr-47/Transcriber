# AGENTS.md

Handoff notes for OpenCode working on Transcriber. This file intentionally
documents only what is **not** derivable from reading the code. Layout, module
responsibilities, CLI flags, API shape and the env-var reference all live in
`README.md` — read that first, then come back here for the parts that took real
work to learn (or real work to keep working).

## What this project is

A faster-whisper + pyannote.audio transcription service with speaker
diarization: a REST API (FastAPI), a one-shot CLI, and a `calls-*` folder
watcher. It is the **original** app that the sibling checkout
(`/home/user/OpenCode/VoxPipe`) was created to replace with a much smaller
whisper.cpp + sherpa-onnx install. The two repos share lineage: VoxPipe adapted
this repo's `format.py` and `merge.py` and follows its JSON contract. Do not
drag VoxPipe's architecture back here (no `media-*` folders, no `vendor/` of
binaries, no `requirements.lock`, no whisper-server) — and do not rename this
repo's internals to match VoxPipe's either.

## Naming — some names are deliberate

- Product, distribution, import package and CLI command are all `transcriber`
  (`transcriber --version`, `import transcriber`). Version `1.0.0` lives in
  exactly two editable places that must never drift — `pyproject.toml` and
  `src/transcriber/__init__.py` (`__version__`) — and both the CLI
  `--version` and the API `version` read the `__version__` one. Bumping means
  editing those two files, then re-running the CLI/`/health`/`/docs` to confirm
  the string surfaced everywhere.
- **The `Transcriber` class is not a leftover.** It is the internal engine
  (`core.py`) and the thing `api.py` and `pipeline.py` both hold. Do not rename
  it to match the project, and do not remove its `__enter__`/`__exit__`/`close`
  — the CLI wraps it in a `with`, the API lifespan closes it on shutdown.
  Likewise a local variable called `transcriber` names an object, not the
  project.
- Operating folders are `calls-{inbox,process,failed,results}`. `CallFolderProcessor`
  (`pipeline.py`) is the matching internal class. Renaming these is a breaking
  change: files left in the old folders would be silently ignored.

## Hard constraints

- **One process, one model, one lock.** `Transcriber.transcribe` serialises
  everything behind `self._lock` because the whisper/pyannote model binaries are
  not safe for concurrent use within one process. The API runs transcription in
  a worker thread (`anyio.to_thread.run_sync`) precisely so requests *queue up
  behind that lock* rather than fighting over the GPU. Never remove the lock or
  "parallelise" inside the process; if multi-GPU work ever happens, the lock is
  the first assumption to revisit.
- **CUDA fails soft, twice, and both paths are load-bearing.**
  - `_preload_cublas()`: CTranslate2 `dlopen()`s `libcublas.so.12` by soname and
    no pip package declares it. The durable fix is preloading the copy shipped
    by the `cuda` extra (`ctypes.CDLL` by absolute path), not
    `LD_LIBRARY_PATH` (lost on reboot) and not `os.environ` at runtime (glibc
    fixes its search path at process start).
  - `_torch_cuda_usable()`: `torch.cuda.is_available()` only reports driver
    presence and stays `True` on a card whose kernels are missing from the build
    (e.g. Pascal with a CUDA 13 torch). `.to("cuda")` is a plain copy that also
    succeeds regardless. The probe is a real matmul forced to sync, cached with
    `lru_cache` (the answer cannot change in-process).
  - faster-whisper `transcribe` returns a **lazy generator**: feature extraction
    and the first GPU encode happen on the first `next()`. `core.transcribe`
    therefore consumes the whole generator *inside* the guarded region
    (`run_whisper`), and a `RuntimeError` whose message mentions
    cuda/cublas/cudnn/cudart/driver — and only then, and only when
    `whisper_device != "cpu"` — triggers `_reload_whisper_cpu()`
    (device `cpu`, compute `int8`). Do not "simplify" this back to guarding only
    generator creation: that is exactly how the CUDA fallback became dead code.
    `test_transcribe_reloads_whisper_on_lazy_cuda_error` pins it.
  - A `ValueError` *load* failure retries with compute type `int8` **only when
    the configured compute type is not already `int8`**.
- **Empty HF token means disabled, and empty env vars mean "unset".**
  `config.py` reads every `TRANSCRIBER_*` var with `or <default>`, so an empty
  string behaves as if unset (documented in the README config table). In
  particular `TRANSCRIBER_HF_TOKEN=""` disables diarization — that is a feature
  (`transcriber --no-diarization` and the API's default flow rely on the same
  `bool(hf_token)` gate), not a bug to "fix".
- **Diarization tuning must never fail a transcription.** `_tune_pipeline`
  edits `pipeline.hyperparameters` (a dict), so a version change that drops a
  known key is caught and skipped with a warning. Ditto for the GPU move: a card
  with unusable kernels keeps diarization on CPU and warns.
- **Attribution is label-only; it cannot repair a bad partition.** The overlap
  rule in `assign_speakers` is *latest-start wins*
  (`max(covering, key=(start, end))`) — the most specific claim on that instant;
  first-wins would let one long turn swallow a file. But a pinned
  `TRANSCRIBER_DIAR_SPEAKERS` splits a two-party call ~98%/2% before attribution
  ever runs (pyannote `num_speakers`), and no overlap rule recovers a
  speaker missing from the rest of the file. Do not chase that symptom in
  `merge.py`; retain `absorb_tiny_turns` instead (below).
- **Absorb tiny turns; never drop them.** Sub-second turns cannot carry a
  reliable embedding and become singleton labels, but on a dense two-party call
  they are exactly the interjections that mark a speaker change. Dropping them
  merged whole minutes of dialogue into single-speaker blocks. `absorb_tiny_turns`
  keeps every turn and reassigns only the label of any speaker whose total speech
  is under `TRANSCRIBER_DIAR_MIN_REGION` to the temporally nearest survivor.
  `0` disables the pass.
- **The folder workflow moves files and never loses a transcript or a report.**
  - Files are *moved* between the four folders, never copied.
  - `_unique_stem`/`_unique_failed_stem` exist because `call.mp3` and `call.m4a`
    both want `call.json`: ownership is read off the audio files sitting next to
    the transcripts, never guessed, and failure reports additionally claim their
    name even after the audio is gone (`report_claims`) so a reason is never
    overwritten. A `.txt` input failing gets `<stem>.error.txt` — otherwise the
    report would destroy the file it describes. Numbered fallback goes 2..999;
    the `RuntimeError` past 999 is intentional and is exactly the code the test
    suite leaves uncovered.
  - Post-transcription finalise (stem → JSON → rendered formats → move to
    results) is wrapped so **any** write or render failure routes the audio to
    `calls-failed` with a `<name>.txt` report and nothing is stranded in
    `calls-process`. `test_result_write_failure_moves_to_failed_with_report`
    pins the "no stranded audio" property.
  - Don't let PyAV look like corrupt audio: if `av` is missing, `probe_audio`
    raises `RuntimeError` naming the fix; it never returns `False` (which would
    methodically file every inbox item as failed).
- **Readiness state must not leak.** `CallFolderProcessor._seen` is pruned
  against the inbox listing at the end of every `process_inbox` scan, or a
  long-running `watch` (a daemon that may run for months) leaks one entry per
  file ever seen. `recover_stale()` runs at watch start and moves leftovers from
  `calls-process` back to `calls-inbox`; it **returns** the recovered list and
  the tests use that return — do not drop it.
- **`install.cmd` and `run.cmd` are 100% CRLF and must stay that way.** Real
  `cmd.exe` misparses LF-only batch — most visibly at `goto`/labels, which is
  exactly what the option parsing and `:help` depend on. `install.sh`/`run.sh`
  are deliberately LF. Count endings with Python, not `awk`: a whole-file LF
  conversion has nearly been mistaken for a clean edit before.
- **`run.cmd` must not enable delayed expansion.** It forwards `%*` to
  `transcriber watch`, and with `EnableDelayedExpansion` on, an `!` in a path or
  argument would be consumed instead of passed through. That is why its probes
  are flat `if ... set` lines rather than nested blocks needing `!VAR!`.
  `install.cmd` *does* enable it (it needs `!ARG:~0,1!` to reject unknown flags).

## Decisions and their reasons

- **The `cuda` extra exists and is not in base deps.** The cuBLAS 12 runtime is
  a ~600 MB wheel that only matters on a GPU machine. CTranslate2 resolves it by
  soname, so the requirement is spelled out explicitly
  (`nvidia-cublas-cu12>=12.9`) rather than left to a dependency that does not
  declare it. Installers offer it only after detecting a GPU (`nvidia-smi`).
- **`av` is a hard dependency of the folder workflow, not an extra.**
  `faster-whisper`'s optional `av` extra is not enough: readiness probing
  (`probe_audio`) needs a real decode to tell "still being written" from
  "corrupt". The pyproject comment says so; the `(frame) -> ImportError` guard
  in `probe_audio` is written to raise, never to guess.
- **`anyio` is a direct dependency** (used directly in `api.py` for
  `to_thread.run_sync`) and is declared in `pyproject.toml`; it was briefly an
  undeclared import.
- **Compute type is never guessed from torch visibility.** `resolve_compute_type`
  returns the configured value or `"default"`, and CTranslate2 picks what the
  resolved backend actually supports (float16 on capable GPUs, int8 on CPU). A
  GPU can be visible to torch while CTranslate2 still refuses float16, so
  "look at torch.cuda to pick float16" was deliberately rejected.
- **Model resolution treats paths and hub IDs differently.** A value that
  *exists* on disk wins; a value that is unambiguously a path (absolute,
  `~`, `./`, `../`) is checked eagerly so a typo fails loudly; a bare name or
  `org/model` id passes through to the hub (they cannot be told apart by
  syntax). `/health` reports `model_path` only when the resolved value is an
  absolute path — `org/model` contains `/` but is a hub reference, not a path.
- **word timestamps are expensive, so they are computed exactly when serialised.**
  - one-shot CLI: only when `--json` **and** `--words` are both present;
  - watcher: always (the machine-readable `<stem>.json` it writes always
    carries words via `include_words=True`);
  - API: default `words=True` query param.
  Do not "simplify" by always computing them in the CLI.
- **Formats choose themselves by extension.** `-o file.ext` picks the format
  from the suffix; an unknown suffix **warns** and writes the default format's
  content into that file anyway — it used to silently write markdown into
  `report.mp3`, which looked like a corrupt recording. `--format` on watch is an
  `append` flag that also accepts comma-separated lists; empty parts are skipped;
  an unknown value is a `parser.error` (exit code 2), not a silent `md`.
- **Empty-text utterances must not crash renderers.** `to_text` previously
  `IndexError`ed on an empty utterance (paragraph wrapping indexing). `to_text`
  and `to_srt` now skip empty texts; `to_html` escapes *everything* (speaker,
  text, title) so transcripts can contain markup safely and a speaker named
  `<script>` cannot run.
- **The JSON contract is defined once, in `models.py`.** `Transcript.to_dict`
  is the single source for `--json`, the watcher's `<stem>.json`, and the API
  response body (`_to_out`). Never hand-build a parallel dict in any of the
  three callers.
- **The API owns a transcriber per process and closes it.** Created in the
  lifespan (so models load on first request and stay warm), closed on shutdown
  only if the app created it — an injected stub (tests) is never closed. Temp
  uploads are cleaned up in a `finally` covering the **entire** request,
  including a client that disconnects mid-upload.
- **No auth in the API, by design.** The module docstring says so; the contract
  is "bind loopback or sit behind a reverse proxy". Do not add auth inside the
  app without a product decision.
- **Installers do not download models.** VoxPipe vendors `whisper.cpp` weights;
  Transcriber does not — faster-whisper and pyannote pull into the Hugging Face
  cache on first run (`~/.cache/huggingface`, on Windows
  `%USERPROFILE%\.cache\huggingface`), same as `pip install -e .` on Linux.
  The installers create the venv, install deps, create the `calls-*` folders,
  and prompt (via `nvidia-smi`) for the CUDA extra. `run.cmd` mirrors
  `run.sh`: default formats `md,txt,html` from `TRANSCRIBER_FORMATS`, everything
  else forwarded to `transcriber watch`.

## Commands

```bash
# setup (after cloning)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # add optional GPU support with:  ".[dev,cuda]"

# lint (pinned rule set: E4,E7,E9,F,I, line-length 120)
.venv/bin/ruff check src tests

# full suite + coverage: 185 passed, 99% coverage, ~6 s, offline
.venv/bin/python -m pytest --cov=transcriber --cov-report=term-missing -q

# one-shot transcription
.venv/bin/transcriber some.mp3 -o out.md --json out.json --words

# folder watcher
.venv/bin/transcriber watch --once
.venv/bin/transcriber watch --format md,txt,html --language ru

# REST API
.venv/bin/uvicorn transcriber.api:app --host 0.0.0.0 --port 8000

# Windows
install.cmd     # venv + deps + optional CUDA extra + calls-* folders
run.cmd         # watcher
```

Notes:

- `python` is **not on PATH** in this dev environment; use `.venv/bin/*` (or
  `python3` for creating the venv).
- The only uncovered lines are the two intentional functions in
  `pipeline.py`: the degenerate 999-collision `RuntimeError` in `_unique_stem`
  and `_first_free` returning `None`. Try to mutate those and the suite should
  stay green.
- The suite needs no network and no GPU: every test injects a stub engine. A
  real end-to-end run (models downloaded, real CUDA/CPU) is a manual
  `transcriber watch --once` with a real file. **Do not** forward the repo's
  test count as proof a real-GPU change works.
- Two `DeprecationWarning`s from `starlette.testclient` (anyio `BlockingPortal`
  alias) are expected noise from a third-party dep, not failures.

## Testing approach — hard-won lessons

- **Assert at the layer where the value is handed over, not one below.**
  `--language` was once accepted by argparse and dropped before it reached the
  processor; the old test constructed the processor directly, *below* the bug,
  and passed for months. `tests/test_cli.py` now stubs `Transcriber`,
  `CallFolderProcessor` and `watch_folders` and asserts the handover itself
  (including that `--language` reaches `CallFolderProcessor(language=...)`).
- **The stub transcriber is the workhorse; keep it honest.** Tests inject fake
  transcribers/pipelines that return tiny `Transcript` objects. When you change
  a call signature (e.g. `word_timestamps`), every stub that records call args
  must learn the new parameter — the CLI's one-shot test records whether
  word timestamps were requested and asserts the exact boolean for the flag
  combinations.
- **Coverage hides gaps; mutating the code is the real check.** The suite is at
  99% partly because of deliberately added regression tests (CUDA lazy reload,
  empty-utterance rendering, empty-env config, result-write failure). When a
  change has no plausible failure, break the code on purpose and watch the
  suite go red before believing the fix.
- **The format and merge layers share lineage with VoxPipe.** If you change
  `format.py`'s paragraph wrapping, sentence splitting, caption timing, or
  `merge.py`'s overlap/absorb logic, treat VoxPipe's tests and docs as an
  upstream reference and its `AGENTS.md` notes about diarization traps as
  directly applicable here.

## Verifying a Windows/batch change

Wine 9.0 runs `cmd.exe` well enough to catch real batch bugs, but it cannot run
any 32-bit EXE and has no Windows Python, so the installers can only be
**parse-tested**, not run to completion:

- `wine cmd /c install.cmd --help`      → help, exit 0
- `wine cmd /c install.cmd --bogus`     → unknown option, exit 1
- `wine cmd /c run.cmd`                 → ".venv is missing" (no `Scripts\` here), exit 1
- `wine cmd /c install.cmd`             → immediately bail at the Python check, exit 1

Three batch traps that have all produced false results:

- **`for /f` in Wine runs the `do` body even when the command produced no
  output, and does not substitute `%%G` inside a multi-line parenthesised
  block.** GPU detection is therefore written as a single-line
  `do if not defined GPUNAME set "GPUNAME=%%G"` — and even then the *positive*
  path is not verifiable under Wine because `nvidia-smi` is absent from the
  prefix. Verify the consumer of the variable, not the detection itself.
- **`findstr /c` treats its pattern as a literal**, so "starts with a dash"
  cannot be expressed with it; `install.cmd` slices `!ARG:~0,1!` with delayed
  expansion instead.
- **Assert on output, and confirm a probe actually contains the text it means
  to exercise** (a substring can occur in a comment before the code being
  tested). Check that a mutation actually changed the file (CRLF files can
  defeat `$`-anchored edits silently).

Do not install a new `.cmd`/`.sh` pair or touch the existing ones without
re-running the four parse tests above; if you adjust line endings, verify with
Python (`data.count('\n')` vs `data.count('\r\n')`) that the `.cmd` files are
still 100% CRLF.

## Known non-bugs

Do not "fix" these; they are expected and were looked into.

- **The two uncovered lines in `pipeline.py`** (the 999-exhaustion `RuntimeError`
  and `_first_free` returning `None`) are intentional: they can only trigger in
  degenerate collision runs that recording workflows should never reach.
- **`watch --once` never fails a held file.** The stalled-scan limit
  (`MAX_STALLED_SCANS = 3`) needs three settled scans to trip; a `--once` run
  usually exits after one. Hold-backs in `--once` are deferred, not failures —
  use a continuous `transcriber watch` to exercise the fail path.
- **The first run downloads models (~460 MB `small` + the diarization
  pipeline).** That long pause is expected, not a hang — `cli.main()` configures
  logging *before* running precisely so the download is not silent.
- **A `.venv` that will not start after the checkout moves** is expected; its
  scripts bake in absolute paths. Recreate it; never move or patch it.
- **Docs drift silently.** The README's env table, feature claims and folder
  names are maintained by hand. When behaviour changes, grep the `.md` files
  for the changed flag/path/env var rather than trusting the prose. The prior
  `TRANSCRIBER_WHISPER_MODEL=""` fix (empty→default) was a real behaviour
  change to keep README truth aligned with.

## Working agreement

- Do not commit unless asked. The 1.0 release changes are not committed yet;
  make that commit deliberately.
- Keep `Transcriber` as the engine name and the lock holder; if parallelising
  ever happens, those two assumptions break first.
- `.venv/`, `calls-*/*` (except their `.gitkeep` files), `.pytest_cache/`,
  `.ruff_cache/` and `.coverage` stay untracked. Anything large or unexpected
  in `git status` is a mistake.
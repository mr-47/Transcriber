# Architecture

## Shared pipeline

```text
Audio file
    ↓
faster-whisper (CTranslate2)          ← word & segment timestamps
    ↓
Segments
    ↓
pyannote diarization (token-gated)    ← speaker turns (optional)
    ↓
assign_speakers + build_utterances
    ↓
Transcript.to_dict → JSON  /  renderers → md / txt / html / srt
```

CLI, watch mode and the REST API all use the same `Transcriber` engine
(`core.py`). `Transcriber` is the internal engine the whole project is named
after; it is not a leftover and holds the lock (below).

## One model per process, serialized by a lock

Models load lazily on first use and are cached for the life of the process (a
CLI run, or one API app). `Transcriber.transcribe` is guarded by a single
`threading.Lock` because faster-whisper and pyannote binaries are not safe for
concurrent use within a process. The API runs transcription in a worker thread
(`anyio.to_thread.run_sync`) precisely so concurrent requests *queue up behind
the lock* instead of fighting over the GPU. If multi-GPU parallelism is ever
wanted, the lock is the first assumption to revisit.

## CUDA fails soft, twice — both paths are load-bearing

### cuBLAS preload

CTranslate2 `dlopen()`s `libcublas.so.12` by soname and no pip package declares
it. `_preload_cublas()` preloads the copy shipped by the `cuda` extra by
absolute path (`ctypes.CDLL`), which is durable — unlike `LD_LIBRARY_PATH`
(lost on reboot) or setting `os.environ` at runtime (glibc fixes its search
path at process start). Without it, CUDA whisper silently falls back to CPU and
logs the reload warning.

### torch CUDA probe

`torch.cuda.is_available()` only reports driver presence and stays `True` on a
card whose kernels are missing from the build (e.g. Pascal with a CUDA 13
torch). `.to("cuda")` is a plain copy that also succeeds regardless.
`_torch_cuda_usable()` therefore runs a real matmul forced to sync, cached with
`lru_cache` (the answer cannot change in-process). A card with unusable kernels
keeps **diarization** on CPU and warns instead of failing the transcription.

### The lazy-generator trap

faster-whisper `transcribe` returns a **lazy generator**: feature extraction
and the first GPU encode happen on the first `next()`. `core.transcribe`
consumes the whole generator *inside* the guarded region, and a `RuntimeError`
whose message mentions cuda/cublas/cudnn/cudart/driver — and only then, and only
when `whisper_device != "cpu"` — triggers `_reload_whisper_cpu()` (device `cpu`,
compute `int8`). Guarding only generator creation is exactly how the CUDA
fallback became dead code once; `test_transcribe_reloads_whisper_on_lazy_cuda_error`
pins the fix.

A `ValueError` *load* failure retries with compute type `int8` only when the
configured compute type is not already `int8`.

## Speaker attribution

`assign_speakers` aligns each whisper segment (by midpoint) with diarization
turns:

- when several turns cover a point, **latest-start wins** (`max(covering,
  key=(start, end))`) — the most specific claim on that instant; first-wins
  would let one long turn swallow a file;
- falling back to the temporally nearest turn, then to the previously assigned
  speaker, then `SPEAKER_00`.

`absorb_tiny_turns` (the `TRANSCRIBER_DIAR_MIN_REGION` pass) keeps every turn
and reassigns only the label of any speaker whose total speech is under the
floor to the temporally nearest survivor; it never drops turns. See
[diarization.md](diarization.md).

## Warm model lifecycle

A single `Transcriber` is created once per process:

- CLI: wrapped in `with Transcriber(settings)`, `close()`d on the way out;
- API: created in the app lifespan (so models load on the first request and
  stay warm), closed on shutdown only if the app created it — an injected stub
  is never closed;
- watcher: one `Transcriber` drives every file in a `watch` run.

## Folder state machine

Watch mode moves files through `calls-inbox/` -> `calls-process/` ->
`calls-results/` (or `calls-failed/`), with write-in-progress detection,
filename-collision disambiguation and crash recovery. The full semantics are in
[folder-workflow.md](folder-workflow.md).

## Module layout

```text
src/transcriber/
├── __init__.py    version (+ `python -m transcriber` entry point)
├── __main__.py    `python -m transcriber` → cli.main
├── cli.py         argparse; one-shot / watch dispatch
├── api.py         FastAPI app
├── config.py      environment variables, model resolution, defaults
├── core.py        Transcriber engine: ASR + diarization + attribution
├── pipeline.py    CallFolderProcessor + watch_folders
├── merge.py       assign_speakers, build_utterances, absorb_tiny_turns
├── format.py      md / txt / html / srt renderers
└── models.py      Transcript dict contract (single source of truth)
```

Version lives in exactly two editable places that must never drift —
`pyproject.toml` and `__version__` in `__init__.py` — and the CLI `--version`
and API `version` both read the latter.
# Installation

## Requirements

- Linux or Windows
- Python >= 3.10 (the floor `pyproject.toml` declares)
- A GPU is optional: transcription works on CPU (`int8`), and CUDA is probed at
  runtime and gracefully skipped when unavailable
- Windows: CUDA inference needs an NVIDIA GPU and driver; the dependency wheels
  are installed by `install.cmd`

The standard repository layout requires no environment variables.

## Quick installation

### Linux

The two repositority launchers mirror each other across platforms:

```bash
./install.sh
./run.sh
```

`install.sh` creates `.venv` and installs the package with dev extras
(`.[dev]`). It does not download models — those pull into the Hugging Face cache
on the first transcription, so the install itself is quick and network-light. It
reminds you if `TRANSCRIBER_HF_TOKEN` is unset (diarization defaults to
disabled).

`run.sh` starts the folder watcher and forwards everything after it to
`transcriber watch`:

```bash
./run.sh --once
./run.sh --format srt
```

The default formats are `md,txt,html`; override with `TRANSCRIBER_FORMATS=md,srt`.

### Manual Linux install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

`pyproject.toml` declares the full runtime set (`faster-whisper`, `pyannote.audio`,
`fastapi`, `uvicorn[standard]`, `python-multipart`, `anyio`, `av`). `.[dev]` adds
pytest, hatchling and ruff.

### Windows

```bat
install.cmd
run.cmd
```

`install.cmd` checks for Python 3.10+ on PATH, creates `.venv`, installs
dependencies, probes for an NVIDIA GPU with `nvidia-smi` and offers the `cuda`
extra when one is found, and creates the `calls-*` working folders. Pass
`--cuda` / `--no-cuda` to force the choice. `run.cmd` mirrors `run.sh` and
forwards to `transcriber watch`.

See [windows.md](windows.md) for the details and the verification status.

## GPU setup

faster-whisper drives whisper through CTranslate2, which needs the CUDA 12
cuBLAS runtime (`libcublas.so.12`) but does not declare it. Install:

```bash
.venv/bin/pip install -e ".[cuda]"        # nvidia-cublas-cu12 >= 12.9, ~600 MB
```

The service preloads that copy at startup, so no `LD_LIBRARY_PATH` is needed.
See [architecture.md](architecture.md).

Diarization is a *separate* CUDA runtime: pyannote moves its pipeline with
PyTorch. Install a `torch` build whose kernels cover your card. CUDA 13 builds
of `torch` dropped Pascal, Maxwell and Volta: the install succeeds and
`torch.cuda.is_available()` returns `True`, but the first real kernel launch
fails with `no kernel image is available for execution on the device`. Check
your card's compute capability:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

For a Pascal/Maxwell/Volta card use a CUDA 12.6 build and pin the exact local
tag, otherwise pip reports "already satisfied" and keeps the incompatible
build:

```bash
pip install "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

## Diarization setup

Speaker labels come from a gated pyannote pipeline:

1. Create a free account at https://huggingface.co and accept the terms of these
   model cards:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
2. Create a read token at https://huggingface.co/settings/tokens
3. Export it: `export TRANSCRIBER_HF_TOKEN=hf_...`

Without a token the service still transcribes, and everything is attributed to a
single speaker (`SPEAKER_00`). That is a feature of the same `bool(hf_token)`
gate the CLI's `--no-diarization` flag uses.

## Models and first-run behavior

Models are **not** installed by the installers. They download into the Hugging
Face cache on the first transcription (`~/.cache/huggingface` on Linux,
`%USERPROFILE%\.cache\huggingface` on Windows):

- the Whisper model, roughly 460 MB for the default `small`
  (`Systran/faster-whisper-small`), plus whatever the selected size needs
- the diarization pipeline, only when `TRANSCRIBER_HF_TOKEN` is set

The first run therefore pauses for a download. Logging is configured before
model loading precisely so that pause is visible rather than look like a hang.
Later runs reuse the cache.

Models cache alongside dependencies: the engine loads a lazy
faster-whisper model on first use and keeps it for the life of the process
(one `Transcriber` per CLI run / API app).
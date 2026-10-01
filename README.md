# Transcriber

Audio transcription service with speaker diarization.

Transcriber combines [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
for speech-to-text with [pyannote.audio](https://github.com/pyannote/pyannote-audio)
for speaker diarization.

It can transcribe individual files, automatically process files dropped into a
folder, or run as a REST API.

```bash
transcriber meeting.mp3
transcriber watch
uvicorn transcriber.api:app --port 8000
```

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)](#platform-support)

## Features

- Word- and segment-level timestamps from Whisper
- Speaker attribution per utterance from pyannote diarization
- Graceful single-speaker fallback when diarization is disabled/unavailable
- Output as Markdown, plain text, HTML, SRT and JSON
- Automatic language detection (`--language` to hint)
- Input audio: `.mp3`, `.wav`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.wma`, `.mp4`, `.webm`, `.mka`, `.mkv`
- CUDA GPU acceleration with automatic CPU fallback (`int8`)
- Folder workflow with safe crash recovery and filename-collision handling
- Linux and Windows support

## Quick start

### Linux

Requirements: Linux, Python 3.10+.

```bash
git clone https://github.com/mr-47/Transcriber.git
cd Transcriber
./install.sh        # venv + dependencies (models download on first run)
./run.sh            # watcher; forwards args to `transcriber watch`
```

Or install manually and use the CLI directly:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/transcriber meeting.mp3 -o meeting.md
```

GPU machines: `pip install -e ".[dev,cuda]"` for the cuBLAS runtime faster-whisper
needs (see [GPU setup](docs/installation.md)). The venv must be activated in
every new terminal (`source .venv/bin/activate`) or called by full path.

### Windows

```bat
install.cmd         :: venv + deps + optional CUDA extra + calls-* folders
run.cmd             :: watcher (forwards args to `transcriber watch`)
```

`install.cmd --cuda` / `--no-cuda` force the GPU choice. See
[Windows support](docs/windows.md).

### Diarization

Speaker labels need a Hugging Face token plus acceptance of the gated model
terms:

```bash
export TRANSCRIBER_HF_TOKEN=hf_...
```

Without a token everything is attributed to a single `SPEAKER_00`.
See [Installation -> Diarization setup](docs/installation.md).

## Usage

### Transcribe a file

```bash
transcriber meeting.mp3                      # markdown to stdout
transcriber meeting.mp3 -o meeting.md        # extension picks the format
transcriber meeting.mp3 -o meeting.txt --format txt --layout sentence
transcriber meeting.mp3 -o meeting.srt --format srt
transcriber meeting.mp3 -o meeting.md --json meeting.json --words
transcriber meeting.mp3 --language en --timestamps
transcriber meeting.mp3 --no-diarization
transcriber meeting.mp3 --model tiny
```

Supported input formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.opus`, `.m4a`,
`.aac`, `.wma`, `.mp4`, `.webm`, `.mka`, `.mkv` (the same list the REST API and
the folder watcher accept).

### Folder watcher

```bash
transcriber watch                # poll calls-inbox every 5s, forever
transcriber watch --once         # drain the inbox, then exit
transcriber watch --dir /data    # folders under a different base directory
transcriber watch --once --format txt,md,html,srt
transcriber watch --language ru --model tiny
```

Drop audio into `calls-inbox/` and it moves through
`calls-inbox/` → `calls-process/` → `calls-results/` (or `calls-failed/` with a
`<name>.txt` report). Files still being written are held, not failed. See
[Folder workflow](docs/folder-workflow.md).

### REST API

```bash
uvicorn transcriber.api:app --host 0.0.0.0 --port 8000
# interactive docs: http://localhost:8000/docs
```

| Endpoint | Description |
|---|---|
| `GET /health` | Service status, diarization state, model/device/compute-type |
| `POST /transcribe` | Multipart upload -> transcript JSON |

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@meeting.mp3" -F "language=en"
curl -X POST "http://localhost:8000/transcribe?words=false" -F "file=@meeting.mp3"
curl http://localhost:8000/health
```

The API has no authentication — bind loopback or sit behind a reverse proxy.
See [REST API](docs/api.md).

## Output formats

| Format | Use |
|---|---|
| Markdown | Human-readable transcripts (default) |
| TXT | Plain text |
| HTML | Self-contained, color-coded page |
| SRT | Subtitles (short 5-7 word captions, proportionally timed) |
| JSON | Programmatic processing |

Example:

```markdown
**SPEAKER_00** · [00:00 – 00:03]

Good morning everyone. Let's start the review.
```

The JSON transcript carries `language`, `language_probability`, `duration`,
`segments` (with optional `words`), speaker-labelled `utterances`, and `text`.
Speaker labels are local to each recording. See
[Output formats](docs/output-formats.md).

## CLI options

| Option | Description |
|---|---|
| `-o FILE` | Write the transcript to a file (default: stdout) |
| `--format {txt,md,html,srt}` | Output format (default: `md`) |
| `--layout {paragraph,sentence}` | Transcript structure (default: `paragraph`) |
| `--json FILE` | Also write the machine-readable JSON transcript |
| `--timestamps` | Include timestamps in `txt` output |
| `--caption-words N` | Max words per SRT caption (default: 6) |
| `--language LANG` | Language hint (`en`, `de`, `ru`, ...) |
| `--model SIZE` | Model override (`tiny`/`base`/`small`/`medium`/`large-v3`) |
| `--no-diarization` | Disable speaker diarization |
| `--words` | Include per-word timestamps in `--json` output |

## Configuration

The standard layout needs no environment variables. Common ones:

| Variable | Default | Description |
|---|---:|---|
| `TRANSCRIBER_WHISPER_MODEL` | `small` | Whisper model or hub id / local path |
| `TRANSCRIBER_WHISPER_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `TRANSCRIBER_HF_TOKEN` | *(empty)* | Token; **empty disables diarization** |
| `TRANSCRIBER_DIAR_MIN_REGION` | `1.0` | Fold tiny speakers into their nearest neighbour; `0` disables |
| `TRANSCRIBER_FORMATS` | `md,txt,html` | Default formats for `run.sh` / `run.cmd` |

Empty variables behave as unset. See the full reference in
[Configuration](docs/configuration.md).

## Platform support

| | Linux | Windows |
|---|---|---|
| CLI | Yes | Yes |
| Folder watcher | Yes | Yes |
| REST API | Yes | Yes |
| Speaker diarization | Yes | Yes |
| CPU inference | Yes | Yes |
| CUDA GPU inference | Yes (verified) | Yes (installer) |
| Python | 3.10+ | 3.10+ |

## Documentation

This README is the landing page. Full reference material lives in
[`docs/`](docs/README-docs-index.md).

### User documentation

| Document | Contents |
|---|---|
| [Installation](docs/installation.md) | Requirements, CUDA GPU setup, first-run behavior |
| [Windows support](docs/windows.md) | `install.cmd` / `run.cmd`, CRLF, verification status |
| [Configuration](docs/configuration.md) | Complete environment-variable reference |
| [REST API](docs/api.md) | Endpoints, one-process design, readiness, security |
| [Folder workflow](docs/folder-workflow.md) | State machine, write-in-progress, collisions, recovery |
| [Output formats](docs/output-formats.md) | JSON schema and all renderers |
| [Troubleshooting](docs/troubleshooting.md) | Common failures and what to check |
| [Known limitations](docs/known-limitations.md) | Explicit current limitations |

### Technical documentation

| Document | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Shared pipeline, one-lock serialization, CUDA fallbacks |
| [Speaker diarization](docs/diarization.md) | Token gate, tuning knobs, attribution rules |
| [Performance](docs/performance.md) | Measurements from a real GPU smoke run |
| [Development](docs/development.md) | Test suite, layout, Windows checks, release checklist |
| [Licensing and attribution](docs/licensing.md) | MIT license, model and dependency terms |

## License

Transcriber source is licensed under the [MIT License](LICENSE). Model weights
and pip dependencies are separate works keeping their own terms. See
[Licensing and attribution](docs/licensing.md).

---

Built with:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [FastAPI](https://github.com/fastapi/fastapi)
- [PyAV](https://github.com/PyAV-Org/PyAV)
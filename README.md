# Transcriber

Audio transcription service with speaker diarization, built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text and
[pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization.

Exposes two interfaces:

- a **REST API** (FastAPI) that accepts an audio file upload and returns a structured transcript
- a **CLI** that writes JSON / plain-text / SRT output to disk

## Features

- Word-level and segment-level timestamps from Whisper
- Speaker attribution per utterance from pyannote speaker diarization
- Graceful fallback to a single speaker when diarization is disabled/unavailable
- Output in Markdown, plain text, HTML, SRT and JSON
- CPU-friendly defaults (`int8` quantization, `small` model) and automatic
  fallback to CPU when CUDA is unavailable/broken

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU inference, add the `cuda` extra, which pulls in the cuBLAS 12 runtime that
CTranslate2 needs but no dependency declares:

```bash
pip install -e ".[cuda]"
```

> **GPU setup.** Install a `torch` build that actually supports your card.
> CTranslate2 (Whisper) and PyTorch (diarization) are separate CUDA runtimes and
> need separate wheels.
>
> CUDA 13 builds of `torch` dropped support for Pascal, Maxwell and Volta. On one
> of those cards the install *succeeds* and `torch.cuda.is_available()` returns
> `True`, but the first real GPU operation fails with `no kernel image is
> available for execution on the device`. Check with
> `python -c "import torch; print(torch.cuda.get_arch_list())"` — your card's
> compute capability (e.g. `6.1` for a GTX 1070) must be covered. Use a CUDA 12.6
> build for those cards:
>
> ```bash
> pip install "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
> ```
>
> Pin the full version including the `+cu126` local tag. A bare `torch==2.14.0`
> already matches an installed CUDA 13 build, so pip reports "already satisfied"
> and silently leaves the incompatible build in place.

### Windows

From a checkout on Windows, run `install.cmd` (from `cmd`, or double-click it).
It checks that Python 3.10+ is on PATH, creates `.venv`, offers the `cuda`
extra when it detects an NVIDIA GPU via `nvidia-smi`, installs the package, and
creates the `calls-*` folders:

```batch
install.cmd
```

Start the folder watcher with:

```batch
run.cmd
```

`run.cmd` passes anything you give it through to `transcriber watch`, so
`run.cmd --once` drains the inbox and exits (handy for Task Scheduler),
`run.cmd --format srt` changes the written formats, and so on. The default
formats are `md,txt,html`; override with an environment
`TRANSCRIBER_FORMATS=md,txt`.

Use `install.cmd --cuda` / `--no-cuda` to force the CUDA choice instead of the
probe. Diarization is enabled by setting the token persistently, which in cmd
is `setx TRANSCRIBER_HF_TOKEN hf_...` (see below for the gated-model terms).
Models still download into the Hugging Face cache on first run, exactly as on
Linux.

### Diarization setup (required for speaker labels)

pyannote's diarization models are gated on the Hugging Face Hub:

1. Create a free account at https://huggingface.co and accept the conditions of these models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
2. Create a read token at https://huggingface.co/settings/tokens
3. Export it: `export TRANSCRIBER_HF_TOKEN=hf_...`

Without a token the service still transcribes, but all speech is attributed to a
single speaker.

## How to run

The `transcriber` command lives inside the virtualenv, so activate it in **every
new terminal session** before running anything:

```bash
cd ~/OpenCode/Transcriber   # your checkout location
source .venv/bin/activate
```

Then pick a mode:

```bash
# 1) Watch the calls-inbox folder workflow (recommended)
transcriber watch                # poll calls-inbox every 5s, forever
transcriber watch --once         # process whatever is in calls-inbox, then exit
transcriber watch --dir /data    # use folders under a different base directory

# 2) Transcribe a single file
transcriber meeting.mp3 -o meeting.md

# 3) Serve the REST API (separate terminal)
uvicorn transcriber.api:app --host 0.0.0.0 --port 8000
```

If `transcriber: command not found`, you forgot to activate the venv. Either run
`source .venv/bin/activate` first, call it by its full path
(`.venv/bin/transcriber ...`), or add the venv to your `PATH` once so it is
always available without activation:

```bash
echo 'export PATH="$HOME/OpenCode/Transcriber/.venv/bin:$PATH"' >> ~/.bashrc
exec bash
```

The first run downloads the Whisper model (~460 MB for `small`) and, with an HF
token, the diarization pipeline. Later runs use the local cache
(`~/.cache/huggingface`).

Devices are handled automatically. If CUDA is unusable the service falls back to
CPU with `int8` — slower, but it still works. Set `TRANSCRIBER_WHISPER_DEVICE=cpu`
upfront to skip the attempt entirely.

You do **not** need to set `LD_LIBRARY_PATH` for cuBLAS. CTranslate2 loads
`libcublas.so.12` by soname, and the service preloads the copy installed by the
`cuda` extra at startup, so the loader can resolve it without any environment
setup. This matters because a `LD_LIBRARY_PATH` export in your shell is lost on
reboot, and setting `os.environ` at runtime does not work — the loader fixes its
search path when the process starts. If you see

```
WARNING transcriber.core: CUDA inference unavailable; reloading whisper model on CPU
```

then the `cuda` extra is missing from that environment; reinstall with
`pip install -e ".[cuda]"`.

PyTorch is probed separately with a real GPU matmul rather than
`torch.cuda.is_available()`, because that call only reports driver presence and
stays `True` on a card whose kernels are missing from the build. If the probe
fails, diarization stays on CPU and logs a warning instead of failing the request.

## CLI usage

```bash
# Markdown transcript to stdout (default format)
transcriber meeting.mp3

# Markdown transcript to a file (extension picks the format)
transcriber meeting.mp3 -o meeting.md

# Different formats & layouts
transcriber meeting.mp3 -o meeting.txt --format txt --layout sentence
transcriber meeting.mp3 -o meeting.html --format html
transcriber meeting.mp3 -o meeting.srt --format srt

# Also write the machine-readable JSON
transcriber meeting.mp3 -o meeting.md --json meeting.json --words

# Language hint + timestamps in text output
transcriber meeting.mp3 --language en --timestamps
```

Options:

| Flag | Description |
| --- | --- |
| `-o FILE` | Write the transcript to a file (default: stdout). A known extension in the filename picks the format automatically |
| `--format {txt,md,html,srt}` | Transcript format (default: `md`) |
| `--layout {paragraph,sentence}` | Text structure: wrapped paragraphs or one line per sentence (default: `paragraph`) |
| `--json FILE` | Also write the machine-readable JSON transcript |
| `--timestamps` | Include timestamps in `txt` output (markdown/html/srt always show them) |
| `--caption-words N` | Max words per SRT caption; long turns are split into 5-7 word lines (default: `6`) |
| `--language LANG` | Hint the audio language to Whisper (e.g. `en`, `de`, `ja`) |
| `--model SIZE` | Override the Whisper model size (default `small`) |
| `--no-diarization` | Disable speaker diarization |
| `--words` | Include per-word timestamps in `--json` output |

## Folder workflow (watch mode)

`transcriber watch` processes calls by moving audio files through four folders
under the base directory (default: current directory, override with `--dir`):

| Folder | Purpose |
| --- | --- |
| `calls-inbox/` | Drop new calls here to be processed |
| `calls-process/` | Call currently being transcribed (in transit) |
| `calls-failed/` | Calls that could not be processed, plus a `<name>.txt` error file |
| `calls-results/` | Successfully transcribed calls with their transcript |

Files are **moved** between folders, never copied. On success the audio keeps
its name and lands next to `<name>.json` (full transcript with word timestamps)
and the transcript file(s). The transcript format mirrors the CLI:
`--format {txt,md,html,srt}` and `--layout {paragraph,sentence}`
(extension `.md`/`.txt`/`.html`/`.srt`). On failure the audio is moved to
`calls-failed/` and the error (including traceback) is written to
`calls-failed/<name>.txt` — both the audio and the error keep the original file
name, only the extension differs.

A second recording that shares a name but not an extension (say `call.mp3` and
`call.m4a`) would both claim `call.json`. The later one instead writes its
transcripts as `call_m4a.*` (its extension folded into the stem), so neither
result is lost; the same disambiguation applies to failure reports. A re-run
over an existing result keeps the original name and overwrites it in place.

Files that are still being written are left in `calls-inbox/` and retried on the
next poll instead of failing immediately. The pipeline first checks that a
file's size and mtime have stopped changing — i.e. the recorder has finished
writing it (an m4a only becomes readable once its index is flushed at the end) —
and only then does it decode and process it. The mtime part also catches a
source that *replaces* the file at the same byte size, which a size-only check
would miss. If a file stays unreadable and unchanged for 3 consecutive scans,
it is moved to `calls-failed/`.

```bash
transcriber watch                      # poll calls-inbox every 5s forever
transcriber watch --dir /data/calls    # use a different base directory
transcriber watch --once               # drain the inbox once, then exit
transcriber watch --interval 1         # poll faster
transcriber watch --language ru        # hint the language on every file
transcriber watch --format txt --layout sentence   # plain dialogue-style results

# multiple formats at once (comma-separated or repeat the flag)
transcriber watch --once --format txt,md,html,srt
transcriber watch --once --format txt --format html
```

Folders are created automatically on start. Any leftover files sitting in
`calls-process/` (e.g. after a crash) are moved back to `calls-inbox/` for
retry. Unsupported file types in the inbox are moved to `calls-failed/`.

## REST API

```bash
uvicorn transcriber.api:app --host 0.0.0.0 --port 8000
```

Interactive docs at http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.mp3" \
  -F "language=en"
# omit word timestamps to shrink the response:
curl -X POST "http://localhost:8000/transcribe?words=false" \
  -F "file=@meeting.mp3"
```

`GET /health` reports service status, whether diarization is enabled, and the
model/device/compute-type the running instance would use. Uploads with an
unknown suffix and empty uploads are rejected with HTTP 400 before any model
runs.

## Configuration (environment variables)

| Variable | Default | Description |
| --- | --- | --- |
| `TRANSCRIBER_WHISPER_MODEL` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) or a local path. A path that does not exist fails with an actionable message instead of a retried download |
| `TRANSCRIBER_WHISPER_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `TRANSCRIBER_WHISPER_COMPUTE_TYPE` | `default` | `int8`, `float16`, `float32`, ...; CTranslate2's `default` picks a type the device supports and downgrades instead of failing. `float16` needs a GPU with fast FP16, so Pascal-era cards (e.g. GTX 1070) land on `float32` |
| `TRANSCRIBER_DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization pipeline identifier |
| `TRANSCRIBER_HF_TOKEN` | *(empty)* | Hugging Face read token for gated diarization models |
| `TRANSCRIBER_DIAR_THRESHOLD` | *(unset)* | pyannote clustering threshold; lower splits more speakers, higher merges them. Leave unset for the pretrained default |
| `TRANSCRIBER_DIAR_SPEAKERS` | *(unset)* | Fix the number of speakers (e.g. `2`); forces exactly that many clusters. Defaults to pyannote's automatic estimate |
| `TRANSCRIBER_DIAR_MIN_ON` / `TRANSCRIBER_DIAR_MIN_OFF` | *(unset)* | Minimum speaker-region on/off durations in seconds (pyannote segmentation). Leave unset for defaults |
| `TRANSCRIBER_DIAR_MIN_REGION` | `1.0` | Speakers whose *total* speech is below this many seconds are folded into their temporally nearest surviving speaker. Fixes sub-second singleton labels that fragment long meetings; `0` disables |

A string variable set to an empty value behaves exactly as if it were unset,
so the documented default applies.

## Output format

The JSON transcript (API response, `--json` file, or `calls-results/<name>.json`)
looks like:

```json
{
  "language": "en",
  "language_probability": 0.97,
  "duration": 42.1,
  "segments": [
    { "start": 0.0, "end": 2.1, "text": "Good morning everyone." }
  ],
  "utterances": [
    { "speaker": "SPEAKER_00", "start": 0.0, "end": 3.4, "text": "Good morning everyone. Let's start." },
    { "speaker": "SPEAKER_01", "start": 3.5, "end": 6.2, "text": "Thanks for joining." }
  ],
  "text": "SPEAKER_00: Good morning everyone. Let's start.\nSPEAKER_01: Thanks for joining."
}
```

The human-readable transcripts (`md` by default) are structured to avoid the
"wall of text": each speaker turn is separated with a label and time span, and
long turns are either **wrapped** into paragraphs that break at logical sentence
boundaries, or split into **short sentence-per-line** entries, depending on
`--layout`.

Markdown (`--format md`, default):

```markdown
**SPEAKER_00** · [00:00 – 00:03]

Good morning everyone. Let's start the review.
```

Sentence-per-line markdown (`--layout sentence`):

```markdown
- [00:00 – 00:03] **SPEAKER_00**: Good morning everyone.
- [00:00 – 00:03] **SPEAKER_00**: Let's start the review.
```

`--format html` produces a self-contained, color-coded page (each speaker has
its own accent color) that renders in any browser.

`--format srt` produces subtitles. Every caption is kept short — the utterance
text is split into **5-7 word** chunks (`--caption-words` to change the target)
and the utterance's time span is divided between the captions proportionally, so
timing stays smooth and each subtitle is easy to read:

```srt
1
00:00:00,000 --> 00:00:02,747
SPEAKER_00: Да. Добрый день. Меня зовут Виктория

2
00:00:02,747 --> 00:00:05,494
SPEAKER_00: Кампаньяк. Вам удобно сейчас разговаривать есть
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Release checklist

The unit suite injects stub engines, so it proves nothing about the real
models. Before tagging a release, run the real pipeline once (network + a
GPU or CPU are needed):

```bash
# 1. The version string surfaces everywhere.
transcriber --version                     # transcriber 1.0.0
python -c "import transcriber; print(transcriber.__version__)"

# 2. Real ASR on a short real recording (tiny model; ~75 MB, no token).
#    Generate some speech, e.g. `ffmpeg ...` or say a sentence into a mic.
TRANSCRIBER_WHISPER_MODEL=tiny transcriber test.mp3 -o out.md --json out.json --words
#    Accept: out.md renders, out.json has utterances + word arrays.

# 3. Folder workflow end to end.
mkdir -p calls-inbox && cp test.mp3 calls-inbox/
TRANSCRIBER_WHISPER_MODEL=tiny transcriber watch --once --format md,txt,html,srt
#    Accept: audio + call.json + the four formats land in calls-results/.

# 4. API.
uvicorn transcriber.api:app --port 8000 &
curl -F "file=@test.mp3" http://localhost:8000/transcribe > /dev/null
curl http://localhost:8000/health
kill %1

# 5. Diarization, only with an accepted gated model + token:
#    TRANSCRIBER_HF_TOKEN=hf_... TRANSCRIBER_WHISPER_MODEL=tiny \
#      transcriber test.mp3 -o out.md
#    A two-party recording should yield at least two distinct SPEAKER_N labels.
```

On a GPU machine also confirm the CUDA path: with the `cuda` extra installed,
`TRANSCRIBER_WHISPER_MODEL=tiny` should log `device=cuda` and no reload
warning; without it, the run should fall back to `cpu,int8` with
`WARNING ... reloading whisper model on CPU` and still produce a transcript.
See AGENTS.md for the Windows installer checks.
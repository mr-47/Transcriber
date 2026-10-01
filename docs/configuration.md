# Configuration

The standard repository layout requires no environment variables.

## Environment variables

| Variable | Default | Description |
|---|---:|---|
| `TRANSCRIBER_WHISPER_MODEL` | `small` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`) or a Hugging Face model id such as `Systran/faster-whisper-large-v3`, or a local path (see Model resolution) |
| `TRANSCRIBER_WHISPER_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `TRANSCRIBER_WHISPER_COMPUTE_TYPE` | *(empty)* | `int8`, `float16`, `float32`, ...; empty means CTranslate2's `default`, which picks a type the resolved backend supports and downgrades instead of failing |
| `TRANSCRIBER_DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization pipeline identifier |
| `TRANSCRIBER_HF_TOKEN` | *(empty)* | Hugging Face read token for the gated diarization models; empty **disables diarization** |
| `TRANSCRIBER_DIAR_THRESHOLD` | *(unset)* | pyannote clustering threshold; lower splits more speakers, higher merges them |
| `TRANSCRIBER_DIAR_SPEAKERS` | *(unset)* | Fix the number of speakers (e.g. `2`), passed as pyannote `num_speakers`; unset leaves the automatic estimate |
| `TRANSCRIBER_DIAR_MIN_ON` / `TRANSCRIBER_DIAR_MIN_OFF` | *(unset)* | Minimum speaker-region on/off durations in seconds (pyannote segmentation) |
| `TRANSCRIBER_DIAR_MIN_REGION` | `1.0` | Speakers whose *total* speech is below this many seconds are folded into their temporally nearest surviving speaker; `0` disables the pass |
| `TRANSCRIBER_FORMATS` | `md,txt,html` | Default output formats for `run.sh` / `run.cmd` only (the watcher itself defaults to `md` unless `--format` is given) |

## Empty variables behave as unset

Every `TRANSCRIBER_*` variable is read with `or <default>`, so an empty string
behaves exactly as if it were unset and the documented default applies. In
particular, `TRANSCRIBER_HF_TOKEN=""` disables diarization — that is a feature,
not a bug; `--no-diarization` relies on the same `bool(hf_token)` gate.

The `TRANSCRIBER_DIAR_*` tuning knobs default to `-1`, which means "leave the
pipeline's pretrained hyperparameters alone". Only values `>= 0` are applied.

## Model resolution

`resolve_whisper_model` treats paths and hub ids differently:

- a value that **exists on disk** wins (file or directory);
- a value that is unambiguously a path (absolute, `~`, `./`, `../`) is checked
  eagerly, so a typo fails with a clear `FileNotFoundError` instead of quietly
  downloading against the wrong name;
- a bare name (`small`) or `org/model` id passes through to the hub — the two
  cannot be told apart by syntax, so the hub reports the error if it does not
  exist.

`/health` reports `model_path` only when the resolved value is an absolute path:
`org/model` contains `/` but is a hub reference, not a path.

## Compute type is never guessed from torch

`resolve_compute_type` returns the configured value or `"default"`, letting
CTranslate2 pick what the resolved backend supports (float16 on capable GPUs,
int8 on CPU). The code deliberately does **not** look at `torch.cuda.is_available()`
to pick float16, because a GPU can be visible to torch while CTranslate2 still
refuses float16 — the same mismatch that the CUDA reload logic in
[architecture.md](architecture.md) exists to catch.

## Example

```bash
TRANSCRIBER_WHISPER_MODEL=large-v3 \
TRANSCRIBER_WHISPER_COMPUTE_TYPE=int8 \
TRANSCRIBER_HF_TOKEN=hf_... \
transcriber meeting.mp3 -o meeting.md
```
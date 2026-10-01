# Performance

These are measurements from a real smoke run on the project's own development
machine (one sample each), useful as engineering baselines rather than
universal guarantees.

## Environment

- NVIDIA GTX 1070 (8 GB), CUDA 13.0 driver
- 4 CPU cores
- Python 3.12 venv, `cuda` extra installed, fresh processes
- `small` model, compute type `default` — on this Pascal-era card CTranslate2
  downgraded to float32, so these are *float32* numbers; an FP16-capable GPU
  should be faster

## Model load

Loading `small` at process start (before the first transcription): **~11 s**.

## Recordings

Log-timed end to end (whisper only; no diarization — no token was set):

| Recording | Duration | ASR time | Realtime |
|---|---|---:|---:|
| 8 kHz mono phone call | 156.4 s | ~12 s | ~13x |
| 48 kHz stereo scene | 389.8 s | ~16 s | ~25x |

The second figure is with a warm model. Combined, two files (546 s of audio)
processed in about 39 s including the model load.

Language detection worked on both: `ru` at 0.99, `en` at 0.80 (rhymed,
music-overlaid narration).

## What is not measured here

- Diarization (pyannote) timings require the token-gated models, which this run
  did not use. pyannote can move to the GPU when the torch build is usable, but
  the segmentation/embedding stages are still long for large recordings.
- CPU-only whisper (`int8`) timings.
- The `large-v3` model and per-word timestamp extraction, both of which are
  materially slower than the baseline above.

## Context from the sibling project

VoxPipe's archived measurements put the old faster-whisper ASR at roughly
14.3 s for the same 156 s call in its benchmark environment, which is in the
same range as the ~12 s measured here — the rewrite's selling point was a
smaller environment and no token, not raw ASR speed.

A single process serializes everything behind one lock, so throughput scales
with recording batch queuing, not with the number of concurrent requests.
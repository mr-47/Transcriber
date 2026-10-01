# Licensing and Attribution

Transcriber source code is licensed under the [MIT License](../LICENSE),
declared as `license = "MIT"` in `pyproject.toml`. The license names no
individual copyright holder — `Copyright (c) 2026 The Transcriber authors`.

The license covers Transcriber's own code only. Models and pip dependencies are
separate works that retain their own terms; Transcriber does not redeclare them.

## Python dependencies

Dependencies are not vendored. Each keeps its own license, read from its
installed package metadata, and none is distributed inside this repository.
The notable runtimes:

- faster-whisper (CTranslate2-backed) and its model weights on the Hugging Face
  Hub (`Systran/faster-whisper-*`) — permissive terms (MIT-family), ungated;
- pyannote.audio and the diarization models
  (`pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`) — **gated**:
  acceptance of the model-card terms plus a read token is required before the
  models download;
- FastAPI / Starlette / uvicorn / pydantic — permissive (BSD/MIT-family).

Nothing is copied: this repo ships no vendored binaries and no third-party
notices file, because there is nothing vendored to attribute. Keep that
property — do not copy a `THIRD-PARTY-NOTICES.md` in without actually
distributing third-party artifacts.

## Model download terms

Models are fetched into the Hugging Face cache on first transcription, not
installed by the installers. Their governing documents live on the Hugging Face
model cards (including the gated-model acceptance step for diarization); see
[installation.md](installation.md). Pre-trained weights are separate works from
Transcriber's code and keep their own terms.

## Relationship to VoxPipe

The sibling project VoxPipe adapted this repository's `format.py` and `merge.py`
and follows the same JSON contract. Its licensing documentation records that
provenance on its side; nothing in this repository's history copies from
VoxPipe — Transcriber is the original.
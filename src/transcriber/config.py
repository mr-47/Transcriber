from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_AUDIO_SUFFIXES = frozenset(
    {
        ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac",
        ".wma", ".mp4", ".webm", ".mka", ".mkv",
    }
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


class TranscriberSettings:
    """Runtime configuration, overridable via environment variables."""

    def __init__(self) -> None:
        self.whisper_model = os.environ.get("TRANSCRIBER_WHISPER_MODEL") or "small"
        self.whisper_device = os.environ.get("TRANSCRIBER_WHISPER_DEVICE") or "auto"
        self.whisper_compute_type = os.environ.get("TRANSCRIBER_WHISPER_COMPUTE_TYPE") or ""
        self.diarization_model = os.environ.get(
            "TRANSCRIBER_DIARIZATION_MODEL"
        ) or "pyannote/speaker-diarization-3.1"
        self.hf_token = os.environ.get("TRANSCRIBER_HF_TOKEN") or ""

        # pyannote clustering/segmentation tuning. All default to "leave the
        # pipeline's pretrained hyperparameters alone"; set them to fix a
        # recording that over-splits into too many speakers.
        self.diarization_threshold = _env_float("TRANSCRIBER_DIAR_THRESHOLD", -1.0)
        self.diarization_num_speakers = _env_int("TRANSCRIBER_DIAR_SPEAKERS", -1)
        self.diarization_min_duration_on = _env_float("TRANSCRIBER_DIAR_MIN_ON", -1.0)
        self.diarization_min_duration_off = _env_float("TRANSCRIBER_DIAR_MIN_OFF", -1.0)

        # Post-processing pass on the diarization output. Speakers whose *total*
        # speech is below this floor get folded into whichever surviving speaker
        # sits closest in time; 0 disables the pass. Fixes the sub-second
        # singleton labels that fragment long meetings.
        self.diarization_min_region_seconds = _env_float(
            "TRANSCRIBER_DIAR_MIN_REGION", 1.0
        )

    @property
    def diarization_enabled(self) -> bool:
        return bool(self.hf_token)

    def resolve_compute_type(self) -> str:
        """Compute type to request from faster-whisper.

        Defaults to CTranslate2's "default", which picks a type the resolved
        backend actually supports (float16 on capable GPUs, int8 on CPU) and
        gracefully downgrades instead of hard-failing. We deliberately do not
        guess from PyTorch's CUDA visibility, because a GPU can be visible to
        torch while CTranslate2 still refuses float16.
        """
        return self.whisper_compute_type or "default"

    def resolve_whisper_model(self) -> str:
        """Resolve the whisper model name to something faster-whisper can load.

        A value pointing at an existing local file or directory wins; otherwise
        it is treated as a Hugging Face model id and passed through. Values that
        are unambiguously *paths* -- absolute, home-relative (``~``), or
        dotted-relative (``./``/``../``) -- are checked eagerly, so a path typo
        fails loudly instead of making faster-whisper retry a download against a
        wrong name. A bare name and an ``org/model`` id (e.g. ``Systran/faster-
        whisper-large-v3``) cannot be told apart by syntax alone, so they pass
        through and the hub reports the error if they do not exist.
        """
        value = self.whisper_model
        candidate = Path(value).expanduser()

        if candidate.exists():
            return str(candidate)

        clearly_a_path = candidate.is_absolute() or value.startswith(("~", ".", os.sep))
        if clearly_a_path:
            raise FileNotFoundError(
                f"whisper model {value!r} not found: {candidate} does not exist. "
                "Set TRANSCRIBER_WHISPER_MODEL to a model file/directory that "
                "exists, or to a Hugging Face model id that faster-whisper can "
                "download (e.g. 'small', 'large-v3', 'Systran/faster-whisper-large-v3')."
            )

        return value
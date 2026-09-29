from __future__ import annotations

import os

SUPPORTED_AUDIO_SUFFIXES = frozenset(
    {
        ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac",
        ".wma", ".mp4", ".webm", ".mka", ".mkv",
    }
)


class TranscriberSettings:
    """Runtime configuration, overridable via environment variables."""

    def __init__(self) -> None:
        self.whisper_model = os.environ.get("TRANSCRIBER_WHISPER_MODEL", "small")
        self.whisper_device = os.environ.get("TRANSCRIBER_WHISPER_DEVICE", "auto")
        self.whisper_compute_type = os.environ.get("TRANSCRIBER_WHISPER_COMPUTE_TYPE", "")
        self.diarization_model = os.environ.get(
            "TRANSCRIBER_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
        )
        self.hf_token = os.environ.get("TRANSCRIBER_HF_TOKEN", "")

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
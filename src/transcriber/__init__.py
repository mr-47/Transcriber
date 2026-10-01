from __future__ import annotations

from .config import SUPPORTED_AUDIO_SUFFIXES, TranscriberSettings
from .core import Transcriber
from .format import (
    FORMATS,
    render,
    split_sentences,
    to_html,
    to_markdown,
    to_srt,
    to_text,
)
from .merge import (
    absorb_tiny_turns,
    assign_speakers,
    build_utterances,
)
from .models import (
    Segment,
    SpeakerTurn,
    Transcript,
    Utterance,
    Word,
)
from .pipeline import CallFolderProcessor, watch_folders

__version__ = "1.0.0"

__all__ = [
    "FORMATS",
    "SUPPORTED_AUDIO_SUFFIXES",
    "CallFolderProcessor",
    "Segment",
    "SpeakerTurn",
    "Transcriber",
    "TranscriberSettings",
    "Transcript",
    "Utterance",
    "Word",
    "absorb_tiny_turns",
    "assign_speakers",
    "build_utterances",
    "render",
    "split_sentences",
    "to_html",
    "to_markdown",
    "to_srt",
    "to_text",
    "watch_folders",
]
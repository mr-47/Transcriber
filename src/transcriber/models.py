from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str | None
    language_probability: float | None
    duration: float
    segments: list[Segment]
    utterances: list[Utterance]

    def to_dict(self, include_words: bool = False) -> dict:
        from .format import to_text

        segments = []
        for s in self.segments:
            item: dict = {"start": s.start, "end": s.end, "text": s.text}
            if include_words:
                item["words"] = [
                    {"start": w.start, "end": w.end, "word": w.word} for w in s.words
                ]
            segments.append(item)
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "segments": segments,
            "utterances": [
                {"speaker": u.speaker, "start": u.start, "end": u.end, "text": u.text}
                for u in self.utterances
            ],
            "text": to_text(self.utterances),
        }
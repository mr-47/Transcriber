from __future__ import annotations

from .models import Segment, SpeakerTurn, Utterance


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[str]:
    """Align each whisper segment with a speaker using diarization turn timestamps.

    For each segment we use its midpoint; the speaker of the turn covering that
    point wins, otherwise we fall back to the nearest turn and finally to the
    previously assigned speaker.
    """
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    speakers: list[str] = []
    idx = 0
    previous: str | None = None

    for seg in segments:
        mid = (seg.start + seg.end) / 2.0
        while idx < len(ordered) and ordered[idx].end <= mid:
            idx += 1

        best: str | None = None
        best_gap = float("inf")
        for j in range(max(0, idx - 2), min(len(ordered), idx + 3)):
            turn = ordered[j]
            if turn.start <= mid <= turn.end:
                best = turn.speaker
                break
            gap = min(abs(turn.start - mid), abs(turn.end - mid))
            if gap < best_gap:
                best_gap = gap
                best = turn.speaker

        if best is None and previous is not None:
            best = previous
        if best is None:
            best = "SPEAKER_00"

        speakers.append(best)
        previous = best

    return speakers


def build_utterances(segments: list[Segment], speakers: list[str]) -> list[Utterance]:
    """Merge consecutive segments spoken by the same speaker into utterances."""
    utterances: list[Utterance] = []
    for seg, spk in zip(segments, speakers):
        if utterances and utterances[-1].speaker == spk:
            current = utterances[-1]
            current.text = f"{current.text} {seg.text}".strip()
            current.end = seg.end
        else:
            utterances.append(
                Utterance(speaker=spk, start=seg.start, end=seg.end, text=seg.text)
            )
    return utterances
from __future__ import annotations

import logging
from bisect import bisect_right

from .models import Segment, SpeakerTurn, Utterance

logger = logging.getLogger(__name__)


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[str]:
    """Align each whisper segment with a speaker using diarization turn timestamps.

    For each segment we use its midpoint. Normally exactly one turn covers that
    point, but diarization output can overlap -- measured on a two-party call,
    3-4 overlapping pairs -- so when several cover it the *latest starting* one
    wins: it is the most specific claim on that instant, whereas taking the
    first would let one long turn swallow the whole file. Failing that we fall
    back to the nearest turn and finally to the previously assigned speaker.

    Note this only makes the *attribution* sane. It cannot repair a degenerate
    partition: a pinned ``TRANSCRIBER_DIAR_SPEAKERS`` splits a two-party call
    98%/2% by speech, and no rule for picking among overlapping turns recovers
    the speaker who is missing from the rest of the recording.
    """
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    starts = [t.start for t in ordered]
    speakers: list[str] = []
    idx = 0
    previous: str | None = None
    last_mid: float | None = None

    for seg in segments:
        mid = (seg.start + seg.end) / 2.0
        # ``idx`` is the first turn that ends after ``mid``. Waved through
        # segments it advances monotonically and amortises to one pass; a
        # reordered segment (mid going backwards) resets it so the scan stays
        # correct instead of silently skipping earlier turns.
        if last_mid is not None and mid < last_mid:
            idx = 0
        last_mid = mid
        while idx < len(ordered) and ordered[idx].end <= mid:
            idx += 1

        # Only turns after the idx pointer can still cover ``mid`` (the earlier
        # ones ended before it), and none past the first turn that *starts*
        # after ``mid`` can either. Bounding the scan this way keeps the whole
        # pass linear-ish in practice instead of quadratic on long calls.
        pos = bisect_right(starts, mid)
        covering = ordered[idx:pos]

        if covering:
            best = max(covering, key=lambda t: (t.start, t.end)).speaker
            speakers.append(best)
            previous = best
            continue

        best: str | None = None
        best_gap = float("inf")
        for j in range(max(0, idx - 2), min(len(ordered), idx + 3)):
            turn = ordered[j]
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


def absorb_tiny_turns(
    turns: list[SpeakerTurn], min_region_seconds: float
) -> list[SpeakerTurn]:
    """Fold speakers with too little speech into their temporal neighbour.

    A sub-second turn cannot carry a reliable embedding, so it tends to become
    a singleton cluster and a label of its own. Dropping those turns is tempting
    and wrong: on a dense two-party call the short turns are exactly the
    interjections that mark a speaker change, and removing them merged whole
    minutes of dialogue into single-speaker blocks. The turns are kept, so
    attribution keeps its evidence, and only the label is reassigned -- to
    whichever surviving speaker sits closest in time.

    ``min_region_seconds`` is the total speech a speaker must carry to keep a
    label of its own; 0 disables the pass.
    """
    if min_region_seconds <= 0 or len(turns) < 2:
        return turns

    totals: dict[str, float] = {}
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for turn in turns:
        span = turn.end - turn.start
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + span
        first.setdefault(turn.speaker, turn.start)
        last[turn.speaker] = max(last.get(turn.speaker, 0.0), turn.end)

    keepers = [s for s in totals if totals[s] >= min_region_seconds]
    if not keepers:
        return turns
    dropped = [s for s in totals if totals[s] < min_region_seconds]
    if not dropped:
        return turns

    def nearest(speaker: str) -> str:
        return min(
            keepers,
            key=lambda k: min(abs(first[k] - last[speaker]), abs(first[speaker] - last[k])),
        )

    remap = {s: nearest(s) for s in dropped}
    for speaker, target in remap.items():
        logger.info("Absorbing tiny speaker %s (%.2fs) into %s", speaker, totals[speaker], target)

    order: dict[str, str] = {}
    for turn in turns:
        root = remap.get(turn.speaker, turn.speaker)
        order.setdefault(root, f"SPEAKER_{len(order):02d}")

    return [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=order[remap.get(turn.speaker, turn.speaker)])
        for turn in turns
    ]
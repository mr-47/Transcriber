from transcriber.merge import (
    absorb_tiny_turns,
    assign_speakers,
    build_utterances,
)
from transcriber.models import Segment, SpeakerTurn


def _seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


def test_assign_speakers_uses_midpoint():
    turns = [
        SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=5.0, end=10.0, speaker="SPEAKER_01"),
    ]
    segments = [
        _seg(0.0, 4.0, "hello"),
        _seg(2.0, 5.0, "world"),
        _seg(5.5, 9.0, "foo"),
    ]
    assert assign_speakers(segments, turns) == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]


def test_assign_speakers_nearest_turn_fallback():
    turns = [SpeakerTurn(start=5.0, end=10.0, speaker="SPEAKER_01")]
    segments = [_seg(0.0, 1.0, "before")]
    assert assign_speakers(segments, turns) == ["SPEAKER_01"]


def test_assign_speakers_empty_turns_defaults():
    segments = [_seg(0.0, 1.0, "a"), _seg(1.0, 2.0, "b")]
    assert assign_speakers(segments, []) == ["SPEAKER_00", "SPEAKER_00"]


def test_assign_speakers_unsorted_turns():
    turns = [
        SpeakerTurn(start=6.0, end=10.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
    ]
    segments = [_seg(0.0, 2.0, "a"), _seg(7.0, 9.0, "b")]
    assert assign_speakers(segments, turns) == ["SPEAKER_00", "SPEAKER_01"]


def test_assign_speakers_latest_starting_turn_wins_on_overlap():
    # Two overlapping turns cover the same midpoint; the latest starting one is
    # the most specific claim on that instant and should win, not the first.
    turns = [
        SpeakerTurn(start=0.0, end=100.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=40.0, end=60.0, speaker="SPEAKER_01"),
    ]
    segments = [_seg(45.0, 55.0, "mid")]

    assert assign_speakers(segments, turns) == ["SPEAKER_01"]


def test_assign_speakers_handles_an_out_of_order_segment():
    """A later-listed segment with an earlier midpoint must not be misattributed.

    The scan cursor advances assuming midpoints never regress; a reordered
    segment must reset it rather than silently skipping the earlier turn.
    """
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=20.0, end=30.0, speaker="SPEAKER_01"),
    ]
    segments = [
        _seg(25.0, 26.0, "later-listed-late"),
        _seg(5.0, 6.0, "earlier-midpoint"),
    ]

    assert assign_speakers(segments, turns) == ["SPEAKER_01", "SPEAKER_00"]


def test_assign_speakers_earliest_covering_when_overlap_shared_start():
    # Equal start times: the longer turn (latest end) is the tie-break.
    turns = [
        SpeakerTurn(start=10.0, end=20.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=10.0, end=15.0, speaker="SPEAKER_01"),
    ]
    segments = [_seg(12.0, 14.0, "both")]

    assert assign_speakers(segments, turns) == ["SPEAKER_00"]


def test_build_utterances_groups_consecutive_speakers():
    segments = [_seg(0, 1, "hello"), _seg(1, 2, "world"), _seg(2, 3, "bye")]
    speakers = ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    out = build_utterances(segments, speakers)
    assert len(out) == 2
    assert out[0].speaker == "SPEAKER_00"
    assert out[0].text == "hello world"
    assert out[0].end == 2.0
    assert out[1].speaker == "SPEAKER_01"
    assert out[1].text == "bye"


def test_build_utterances_alternating_speakers():
    segments = [_seg(0, 1, "a"), _seg(1, 2, "b"), _seg(2, 3, "c")]
    speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    out = build_utterances(segments, speakers)
    assert len(out) == 3
    assert [u.text for u in out] == ["a", "b", "c"]


def test_absorb_tiny_turns_folds_small_speaker():
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=0.5, end=0.6, speaker="SPEAKER_01"),
        SpeakerTurn(start=20.0, end=30.0, speaker="SPEAKER_02"),
    ]
    out = absorb_tiny_turns(turns, min_region_seconds=1.0)

    # SPEAKER_01 (0.1s) is absorbed: SPEAKER_00's window is closer (gap 0.6)
    # than SPEAKER_02's (gap 20), and SPEAKER_02 keeps its label as SPEAKER_01.
    assert [(t.speaker, t.start, t.end) for t in out] == [
        ("SPEAKER_00", 0.0, 10.0),
        ("SPEAKER_00", 0.5, 0.6),
        ("SPEAKER_01", 20.0, 30.0),
    ]
    assert {t.speaker for t in out} == {"SPEAKER_00", "SPEAKER_01"}


def test_absorb_tiny_turns_leaves_all_tiny_recordings_alone():
    """No speaker passes the floor: relabeling anything would be arbitrary."""
    turns = [
        SpeakerTurn(start=0.0, end=0.2, speaker="SPEAKER_00"),
        SpeakerTurn(start=0.3, end=0.4, speaker="SPEAKER_01"),
    ]
    assert absorb_tiny_turns(turns, min_region_seconds=1.0) == turns


def test_absorb_tiny_turns_leaves_all_keepers_alone():
    """Every speaker carries enough speech: nothing to absorb."""
    turns = [
        SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=6.0, end=7.0, speaker="SPEAKER_01"),
    ]
    assert absorb_tiny_turns(turns, min_region_seconds=1.0) == turns


def test_absorb_tiny_turns_disabled_at_zero():
    turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=0.5, end=0.6, speaker="SPEAKER_01"),
    ]
    assert absorb_tiny_turns(turns, min_region_seconds=0) == turns


def test_absorb_tiny_turns_keeps_turn_count():
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.0, end=2.1, speaker="SPEAKER_01"),
        SpeakerTurn(start=10.0, end=11.0, speaker="SPEAKER_00"),
    ]
    out = absorb_tiny_turns(turns, min_region_seconds=1.0)

    assert len(out) == 3
    assert {t.speaker for t in out} == {"SPEAKER_00"}
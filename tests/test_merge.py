from transcriber.merge import assign_speakers, build_utterances
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
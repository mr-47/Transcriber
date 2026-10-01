import pytest

from transcriber.format import (
    render,
    split_sentences,
    to_html,
    to_markdown,
    to_srt,
    to_text,
)
from transcriber.models import Transcript, Utterance


def _utt(speaker, start, end, text):
    return Utterance(speaker=speaker, start=start, end=end, text=text)


def _transcript(*utterances):
    return Transcript(
        language="en",
        language_probability=0.99,
        duration=10.0,
        segments=[],
        utterances=list(utterances),
    )


def test_split_sentences_basic():
    text = "Good morning everyone! Let's start. Any questions?"
    assert split_sentences(text) == [
        "Good morning everyone!",
        "Let's start.",
        "Any questions?",
    ]


def test_split_sentences_avoids_mr_smith():
    assert split_sentences("Mr. Smith called at noon.") == ["Mr. Smith called at noon."]


def test_split_sentences_strips_blanks():
    assert split_sentences("  Hello.  World.  ") == ["Hello.", "World."]


def test_to_text_paragraph_alignment():
    text = "word " * 40
    out = to_text([_utt("SPEAKER_00", 5, 40, text)])
    first, *rest = out.splitlines()
    assert first.startswith("SPEAKER_00: ")
    assert all(line.startswith(" " * len("SPEAKER_00: ")) for line in rest)


def test_to_text_skips_empty_utterances():
    assert to_text([_utt("SPEAKER_00", 5, 40, "")]) == ""
    mixed = [
        _utt("SPEAKER_00", 5, 40, ""),
        _utt("SPEAKER_01", 41, 50, "hi"),
    ]
    assert to_text(mixed) == "SPEAKER_01: hi"


def test_to_text_paragraph_breaks_at_sentence_boundaries():
    text = "Alpha one. Beta two. Gamma three four."
    out = to_text([_utt("SPEAKER_00", 0, 1, text)], width=28)
    first, second = out.splitlines()
    assert first == "SPEAKER_00: Alpha one. Beta two."
    assert second == " " * 12 + "Gamma three four."


def test_to_text_paragraph_no_partial_sentence_when_it_fits():
    text = "Alpha one. Beta two. Gamma three."
    lines = to_text([_utt("SPEAKER_00", 0, 2, text)], width=28).splitlines()
    assert len(lines) == 2
    for line in lines:
        assert line.rstrip().endswith((".", "!", "?", "\u2026"))


def test_long_sentence_still_hard_wraps():
    text = "This is a single very long sentence that far exceeds the available width of the line."
    lines = to_text([_utt("SPEAKER_00", 0, 2, text)], width=40).splitlines()
    assert len(lines) > 1


def test_to_text_sentence_layout():
    esc = [_utt("SPEAKER_00", 0, 2, "Hello there. Call me Alice."), _utt("SPEAKER_01", 2, 5, "Hi. How are you?")]
    out = to_text(esc, layout="sentence")
    assert out == (
        "SPEAKER_00: Hello there.\n"
        "SPEAKER_00: Call me Alice.\n"
        "SPEAKER_01: Hi.\n"
        "SPEAKER_01: How are you?"
    )


def test_to_text_sentence_layout_timestamps():
    out = to_text([_utt("SPEAKER_00", 65, 70, "One minute in.")], layout="sentence", timestamps=True)
    assert out == "[01:05] SPEAKER_00: One minute in."


def test_to_markdown_paragraph():
    md = to_markdown([_utt("SPEAKER_00", 0, 2, "Hello world.")])
    assert "**SPEAKER_00**" in md
    assert "[00:00 \u2013 00:02]" in md
    assert "Hello world." in md


def test_to_markdown_paragraph_breaks_at_sentences():
    text = "Alpha one. Beta two. Gamma three four."
    md = to_markdown([_utt("SPEAKER_00", 0, 1, text)], width=28)
    assert "\n\nAlpha one. Beta two.\nGamma three four." in md


def test_to_markdown_sentence():
    md = to_markdown(
        [_utt("SPEAKER_00", 0, 1, "One. Two.")], layout="sentence"
    )
    assert md.splitlines() == [
        "- [00:00 \u2013 00:01] **SPEAKER_00**: One.",
        "- [00:00 \u2013 00:01] **SPEAKER_00**: Two.",
    ]


def test_to_html_escapes_and_labels():
    tab = [_utt("SPEAKER_00", 0, 1, "<script>alert(1)</script>")]
    out = to_html(tab)
    assert "&lt;script&gt;" in out
    assert "SPEAKER_00" in out
    assert "<!DOCTYPE html>" in out


def test_to_srt_blocks():
    out = to_srt([_utt("SPEAKER_00", 0, 1, "hey")])
    assert out == "1\n00:00:00,000 --> 00:00:01,000\nSPEAKER_00: hey"


def test_srt_splits_long_turns_into_word_chunks():
    text = "one two three four five six seven eight nine ten"
    utt = _utt("SPEAKER_00", 0, 10, text)
    out = to_srt([utt])
    blocks = [b for b in out.split("\n\n") if b]
    assert len(blocks) == 2
    words = [b.split("\n")[2].replace("SPEAKER_00: ", "").split() for b in blocks]
    assert [len(w) for w in words] == [6, 4]
    assert words[0] == ["one", "two", "three", "four", "five", "six"]
    assert words[1] == ["seven", "eight", "nine", "ten"]


def test_srt_chunk_timestamps_are_proportional():
    text = "one two three four five six seven eight"
    utt = _utt("SPEAKER_00", 2, 6, text)
    out = to_srt([utt])
    blocks = [b for b in out.split("\n\n") if b]
    assert len(blocks) == 2
    first_start, first_end = _parse_srt_range(blocks[0])
    second_start, second_end = _parse_srt_range(blocks[1])
    assert first_start == 2.0
    assert second_end == 6.0
    assert abs(first_end - 2.0 - 3.0) < 1e-6
    assert abs(second_start - first_end) < 1e-6


def test_to_srt_skips_empty_utterance_text():
    """A whitespace-only utterance must not emit a (broken) subtitle block."""
    mixed = [
        _utt("SPEAKER_00", 0, 1, "   "),
        _utt("SPEAKER_01", 1, 2, "hi"),
    ]
    out = to_srt(mixed)
    blocks = [b for b in out.split("\n\n") if b]
    assert len(blocks) == 1
    assert "SPEAKER_01" in blocks[0]


def test_to_srt_renders_nothing_for_all_empty():
    out = to_srt([_utt("SPEAKER_00", 0, 1, "  ")])
    assert out == ""


@pytest.mark.parametrize("words_per_caption", [0, -3])
def test_to_srt_rejects_a_non_positive_caption_size(words_per_caption):
    with pytest.raises(ValueError, match="words_per_caption"):
        to_srt([_utt("SPEAKER_00", 0, 1, "hello")], words_per_caption=words_per_caption)


def test_srt_custom_words_per_caption():
    text = "one two three four five six seven"
    out = to_srt([_utt("SPEAKER_00", 0, 7, text)], words_per_caption=5)
    blocks = [b for b in out.split("\n\n") if b]
    lengths = [len(b.split("\n")[2].split()[1:]) for b in blocks]
    assert lengths == [5, 2]


def _parse_srt_range(block: str) -> tuple[float, float]:
    start_raw, end_raw = block.split("\n")[1].split(" --> ")
    return _srt_to_seconds(start_raw), _srt_to_seconds(end_raw)


def _srt_to_seconds(raw: str) -> float:
    hours, minutes, rest = raw.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def test_render_defaults_to_markdown():
    out = render(_transcript(_utt("SPEAKER_00", 0, 1, "hi")), "md")
    assert out.startswith("**SPEAKER_00**")
    assert "hi" in out


def test_render_txt_and_srt_and_html():
    tr = _transcript(_utt("SPEAKER_00", 0, 1, "hi"))
    assert render(tr, "txt") == "SPEAKER_00: hi"
    assert render(tr, "srt").startswith("1\n00:00:00,000")
    assert render(tr, "html").startswith("<!DOCTYPE html>")


def test_render_rejects_unknown_format():
    try:
        render(_transcript(), "pdf")
    except ValueError as exc:
        assert "pdf" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
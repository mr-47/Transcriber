from __future__ import annotations

import html
import re
import textwrap

from .models import Transcript, Utterance

DEFAULT_WIDTH = 100

FORMATS = {"txt", "md", "html", "srt"}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[^\sa-z])")

_KNOWN_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "rep.", "sen.",
    "jr.", "sr.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "no.",
)

_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _fmt_srt_timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _fmt_clock(seconds: float) -> str:
    secs = int(seconds)
    return f"{secs // 60:02d}:{secs % 60:02d}"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on `. ! ? …` boundaries.

    A simple heuristic splitter: it avoids splitting before lowercase text and
    after known abbreviations ("Mr.", "Dr.", "e.g.", ...), so common sentences
    survive but exotic abbreviations may still mis-split.
    """
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        previous = text[start:match.start()].strip()
        if previous.lower().endswith(_KNOWN_ABBREVIATIONS):
            continue
        parts.append(previous)
        start = match.end()
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def wrap(text: str, width: int = DEFAULT_WIDTH) -> str:
    normalized = " ".join(text.split())
    return textwrap.fill(
        normalized,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wrap_sentences(text: str, width: int) -> list[str]:
    """Wrap text preferring breaks at logical sentence boundaries.

    Sentences are packed onto lines up to `width`; a line breaks before a
    sentence rather than cutting one in half. A sentence longer than the width
    is hard-wrapped in place as a last resort.
    """
    sentences = split_sentences(text)
    lines: list[str] = []
    buffer = ""
    index = 0
    while index < len(sentences):
        sentence = sentences[index]
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= width:
            buffer = candidate
            index += 1
            continue
        if buffer:
            lines.append(buffer)
            buffer = ""
            continue
        pieces = textwrap.wrap(
            sentence,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(pieces[:-1])
        buffer = pieces[-1]
        index += 1
    if buffer:
        lines.append(buffer)
    return lines


def _paragraph_block(text: str, prefix: str, width: int) -> str:
    indent = " " * len(prefix)
    text_width = max(20, width - len(prefix))
    wrapped = _wrap_sentences(text, text_width)
    return prefix + wrapped[0] if len(wrapped) == 1 else "\n".join([prefix + wrapped[0], *[indent + line for line in wrapped[1:]]])


def to_text(
    utterances: list[Utterance],
    layout: str = "paragraph",
    timestamps: bool = False,
    width: int = DEFAULT_WIDTH,
) -> str:
    lines = []
    for u in utterances:
        prefix = f"{u.speaker}: "
        if timestamps:
            prefix = f"[{_fmt_clock(u.start)}] {prefix}"
        if layout == "sentence":
            for sentence in split_sentences(u.text):
                lines.append(prefix + sentence)
        else:
            lines.append(_paragraph_block(u.text, prefix, width))
    return "\n".join(lines)


def to_markdown(
    utterances: list[Utterance],
    layout: str = "paragraph",
    width: int = DEFAULT_WIDTH,
) -> str:
    if layout == "sentence":
        lines: list[str] = []
        for u in utterances:
            span = f"[{_fmt_clock(u.start)} \u2013 {_fmt_clock(u.end)}]"
            for sentence in split_sentences(u.text):
                lines.append(f"- {span} **{u.speaker}**: {sentence}")
        return "\n".join(lines)

    blocks = []
    for u in utterances:
        span = f"[{_fmt_clock(u.start)} \u2013 {_fmt_clock(u.end)}]"
        body = "\n".join(_wrap_sentences(u.text, width))
        blocks.append(f"**{u.speaker}** \u00b7 {span}\n\n{body}")
    return "\n\n".join(blocks)


def to_html(utterances: list[Utterance], title: str = "Transcript") -> str:
    colors: dict[str, str] = {}
    rows = []
    for u in utterances:
        if u.speaker not in colors:
            colors[u.speaker] = _PALETTE[len(colors) % len(_PALETTE)]
        color = colors[u.speaker]
        rows.append(
            '<div class="turn">'
            f'<div class="meta"><span class="speaker" style="color:{color}">{html.escape(u.speaker)}</span>'
            f'<span class="time">{_fmt_clock(u.start)}\u2013{_fmt_clock(u.end)}</span></div>'
            f'<div class="body">{html.escape(u.text)}</div>'
            "</div>"
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        "<style>\n"
        "body { max-width: 800px; margin: 2rem auto; padding: 0 1rem; "
        'font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #222; }\n'
        ".turn { margin: 1.2rem 0; padding-left: 0.8rem; border-left: 4px solid #eee; }\n"
        ".meta { margin-bottom: 0.3rem; }\n"
        ".speaker { font-weight: 600; }\n"
        ".time { font-size: 0.8em; color: #777; margin-left: 0.6em; }\n"
        ".body { white-space: pre-wrap; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n" + "\n".join(rows) + "\n</body>\n</html>\n"
    )


def to_srt(utterances: list[Utterance], words_per_caption: int = 6) -> str:
    """Render subtitles.

    Each caption holds up to `words_per_caption` words (default 6, i.e. short
    5-7 word lines), and the utterance time span is divided between captions in
    proportion to how many words each one contains so the timing stays smooth.
    """
    blocks = []
    index = 1
    for u in utterances:
        words = u.text.split()
        if not words:
            continue
        captions = [
            words[i : i + words_per_caption]
            for i in range(0, len(words), words_per_caption)
        ]
        total = len(words)
        span = max(0.0, u.end - u.start)
        cursor = u.start
        for caption in captions:
            fraction = len(caption) / total
            end = cursor + span * fraction
            blocks.append(
                f"{index}\n"
                f"{_fmt_srt_timestamp(cursor)} --> {_fmt_srt_timestamp(end)}\n"
                f"{u.speaker}: {' '.join(caption)}"
            )
            cursor = end
            index += 1
    return "\n\n".join(blocks)


def render(
    transcript: Transcript,
    fmt: str,
    layout: str = "paragraph",
    timestamps: bool = False,
    words_per_caption: int = 6,
) -> str:
    """Render a transcript in the requested output format."""
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose one of: {', '.join(sorted(FORMATS))}")
    if fmt == "srt":
        return to_srt(transcript.utterances, words_per_caption=words_per_caption)
    if fmt == "html":
        return to_html(transcript.utterances)
    if fmt == "md":
        return to_markdown(transcript.utterances, layout=layout)
    return to_text(transcript.utterances, layout=layout, timestamps=timestamps)
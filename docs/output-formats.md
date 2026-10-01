# Output Formats

Transcriber renders human-readable transcripts and one machine-readable JSON
transcript. Every call site (CLI `--json`, the watcher's `<name>.json`, the API
response body) shares the same `Transcript.to_dict` — the contract is defined
once, in `models.py`; the three callers never hand-build a parallel dict.

## JSON schema

```json
{
  "language": "ru",
  "language_probability": 0.9888,
  "duration": 156.4,
  "segments": [
    { "start": 0.0, "end": 2.1, "text": "Да. Добрый день." }
  ],
  "utterances": [
    { "speaker": "SPEAKER_00", "start": 0.0, "end": 3.4, "text": "Да. Добрый день." }
  ],
  "text": "SPEAKER_00: Да. Добрый день."
}
```

- `language` is faster-whisper's shorthand (`ru`, `en`, ...), with
  `language_probability` from the same detection pass
- `segments[].words` appears only when word timestamps are included
- `text` is the plain-text rendering of the utterances (the API historically
  omitted it; Transcriber includes it)

## Word timestamps are computed exactly when serialized

Word-level alignment is expensive, so it is never computed eagerly. The exact
triggers differ per interface — do not "simplify" the CLI's gating:

| Interface | Words computed |
|---|---|
| One-shot CLI | only when `--json` is given with `--words` |
| Watcher `calls-results/<name>.json` | always (`include_words=True`) |
| API | default `words=true` query parameter |

## Speaker labels

Labels are assigned by clustering order (`SPEAKER_00`, `SPEAKER_01`, ...) and
are stable only within a single file. Transcriber does not do persistent
speaker identity recognition. Without a diarization token, everything is
`SPEAKER_00`.

## Markdown

Default paragraph layout — a label and time span per turn, long turns wrapped at
logical sentence boundaries:

```markdown
**SPEAKER_00** · [00:00 – 00:03]

Good morning everyone. Let's start the review.
```

Sentence layout:

```markdown
- [00:00 – 00:03] **SPEAKER_00**: Good morning everyone.
- [00:00 – 00:03] **SPEAKER_00**: Let's start the review.
```

## Plain text

Identical structure to markdown without the markup. Timestamps appear only with
`--format txt --timestamps`.

## HTML

A self-contained, color-coded page (each speaker gets an accent color from a
fixed palette) that opens in any browser. Transcription content is **escaped** —
speaker, text and title — so a transcript containing markup, or a speaker named
`<script>`, cannot run in the page.

## SRT

Each utterance is split into captions of up to `--caption-words` words (default
6, i.e. short 5-7 word lines), and the utterance's time span is divided across
captions in proportion to their word counts so subtitle timing stays smooth:

```srt
1
00:00:00,000 --> 00:00:02,747
SPEAKER_00: Да. Добрый день. Меня зовут Виктория

2
00:00:02,747 --> 00:00:05,494
SPEAKER_00: Кампаньяк. Вам удобно сейчас разговаривать есть
```

## Empty texts never crash a renderer

`to_text` and `to_srt` skip empty-text utterances (the old `to_text`
`IndexError`ed on one), and empty captions are skipped.

## CLI examples

```bash
transcriber meeting.mp3                          # markdown to stdout
transcriber meeting.mp3 -o meeting.md            # extension picks the format
transcriber meeting.mp3 -o meeting.txt --format txt --layout sentence
transcriber meeting.mp3 -o meeting.html --format html
transcriber meeting.mp3 -o meeting.srt --format srt
transcriber meeting.mp3 -o meeting.md --json meeting.json --words
transcriber meeting.mp3 --language en --timestamps
```

An unknown suffix in `-o FILE` **warns** (on stderr) and writes the selected
format's content into that file anyway — it used to silently write markdown
into `report.mp3`, which looked like a corrupt recording.
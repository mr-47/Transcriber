from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import TranscriberSettings
from .core import Transcriber
from .format import FORMATS, render
from .pipeline import CallFolderProcessor, watch_folders

DEFAULT_FORMAT = "md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcribe audio files and attribute speech to speakers (diarization).",
    )
    parser.add_argument("audio", help="Path to the input audio file (mp3, wav, m4a, ...)")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Write the transcript to this file instead of stdout; the extension picks the format when present",
    )
    parser.add_argument(
        "--format",
        choices=sorted(FORMATS),
        default=DEFAULT_FORMAT,
        help=f"Transcript format: txt, md, html, srt (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--layout",
        choices=["paragraph", "sentence"],
        default="paragraph",
        help="How transcript text is structured: wrapped paragraph or one line per sentence (default: paragraph)",
    )
    parser.add_argument(
        "--json", type=Path, help="Also write the machine-readable JSON transcript to this file"
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in txt output (markdown/html/srt always show them)",
    )
    parser.add_argument(
        "--caption-words",
        type=int,
        default=6,
        help="Max words per SRT caption (subtitles are split into short 5-7 word lines; default: 6)",
    )
    parser.add_argument(
        "--language", help="Optional language hint for whisper (e.g. 'en', 'de', 'ja')"
    )
    parser.add_argument(
        "--model", help="Whisper model size to override config (tiny/base/small/medium/large-v3)"
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization (everything attributed to one speaker)",
    )
    parser.add_argument(
        "--words", action="store_true", help="Include per-word timestamps in the --json output"
    )
    return parser


def build_watch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber watch",
        description="Watch calls-inbox and process every new audio file through the calls-* folders.",
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Base directory containing the calls-inbox/process/failed/results folders (default: current directory)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the current calls-inbox contents once, then exit",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Poll interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--format",
        action="append",
        metavar="{txt,md,html,srt}",
        help="Transcript format(s) written to calls-results; comma-separated or repeat the flag "
        "(e.g. --format txt,md,html). Default: md",
    )
    parser.add_argument(
        "--layout",
        choices=["paragraph", "sentence"],
        default="paragraph",
        help="How transcript text is structured in the results (default: paragraph)",
    )
    parser.add_argument(
        "--caption-words",
        type=int,
        default=6,
        help="Max words per SRT caption (subtitle lines; default: 6)",
    )
    parser.add_argument(
        "--language", help="Optional language hint for whisper (e.g. 'en', 'de', 'ja')"
    )
    parser.add_argument(
        "--model", help="Whisper model size to override config (tiny/base/small/medium/large-v3)"
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )
    return parser


def _resolve_format(fmt: str, output: Path | None) -> str:
    if output is not None and output.suffix:
        extension = output.suffix.lower().lstrip(".")
        if extension in FORMATS:
            return extension
    return fmt


def _handle_transcript(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    settings = TranscriberSettings()
    if args.model:
        settings.whisper_model = args.model
    if args.no_diarization:
        settings.hf_token = ""

    transcriber = Transcriber(settings)
    result = transcriber.transcribe(args.audio, language=args.language)

    if args.json:
        payload = json.dumps(
            result.to_dict(include_words=args.words), ensure_ascii=False, indent=2
        )
        args.json.write_text(payload + "\n", encoding="utf-8")

    fmt = _resolve_format(args.format, args.output)
    content = render(
        result,
        fmt,
        layout=args.layout,
        timestamps=args.timestamps,
        words_per_caption=args.caption_words,
    )

    if args.output:
        output = args.output
        if not output.suffix:
            output = Path(f"{output}.{fmt}")
        output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)

    return 0


def _parse_watch_formats(raw: list[str] | None) -> list[str]:
    if not raw:
        return [DEFAULT_FORMAT]
    formats: list[str] = []
    for value in raw:
        for part in value.split(","):
            fmt = part.strip().lower()
            if not fmt:
                continue
            if fmt not in FORMATS:
                raise ValueError(
                    f"Unsupported format {fmt!r}. Choose from: {', '.join(sorted(FORMATS))}"
                )
            if fmt not in formats:
                formats.append(fmt)
    return formats or [DEFAULT_FORMAT]


def _handle_watch(argv: list[str]) -> int:
    args = build_watch_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    settings = TranscriberSettings()
    if args.model:
        settings.whisper_model = args.model
    if args.no_diarization:
        settings.hf_token = ""

    transcriber = Transcriber(settings)
    try:
        formats = _parse_watch_formats(args.format)
    except ValueError as exc:
        parser = build_watch_parser()
        parser.error(str(exc))
    processor = CallFolderProcessor(
        transcriber,
        args.dir,
        output_formats=formats,
        layout=args.layout,
        words_per_caption=args.caption_words,
    )
    processed = watch_folders(processor, interval=args.interval, once=args.once)
    logging.getLogger(__name__).info("Finished, processed %d file(s)", processed)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "watch":
        return _handle_watch(args[1:])
    return _handle_transcript(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
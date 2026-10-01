from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import TranscriberSettings
from .core import Transcriber
from .format import FORMATS, render
from .pipeline import CallFolderProcessor, watch_folders

DEFAULT_FORMAT = "md"

VERSION_STRING = f"transcriber {__version__}"


def _add_version_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", action="version", version=VERSION_STRING)


def _positive_int(value: str) -> int:
    """argparse type for counts that must be >= 1.

    ``--caption-words 0`` reaches ``range(0, n, 0)`` and dies with a bare
    ValueError, and a negative value writes an empty subtitle file that looks
    like a successful run. Rejecting here names the actual problem.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {number}")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcribe audio files and attribute speech to speakers (diarization).",
    )
    _add_version_flag(parser)
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
        type=_positive_int,
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
    _add_version_flag(parser)
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
        type=_positive_float,
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
        type=_positive_int,
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

    with Transcriber(settings) as transcriber:
        result = transcriber.transcribe(
            args.audio,
            language=args.language,
            # Word timestamps are only ever serialised by `--json --words`;
            # skip the expensive word-level alignment otherwise.
            word_timestamps=bool(args.json and args.words),
        )

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
        if output.suffix:
            extension = output.suffix.lower().lstrip(".")
            if extension not in FORMATS:
                # The flag is honoured (the user pointed at this file), but an
                # unknown extension no longer silently picks a format: writing
                # markdown into `report.mp3` looked like a corrupt recording.
                print(
                    f"transcriber: warning: {extension!r} is not a transcript format; "
                    f"writing {fmt} content to {output}",
                    file=sys.stderr,
                )
        else:
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

    settings = TranscriberSettings()
    if args.model:
        settings.whisper_model = args.model
    if args.no_diarization:
        settings.hf_token = ""

    with Transcriber(settings) as transcriber:
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
            language=args.language,
        )
        processed = watch_folders(processor, interval=args.interval, once=args.once)
    logging.getLogger(__name__).info("Finished, processed %d file(s)", processed)
    return 0


def main(argv: list[str] | None = None) -> int:
    # One-shot transcription loads a multi-hundred-MB model and can spend
    # minutes downloading it on the first run; silence would look like a hang.
    # Configure once for both commands.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "watch":
        return _handle_watch(args[1:])
    return _handle_transcript(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
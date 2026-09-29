from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import SUPPORTED_AUDIO_SUFFIXES
from .core import Transcriber
from .format import FORMATS, render

logger = logging.getLogger(__name__)

FOLDER_INBOX = "calls-inbox"
FOLDER_PROCESS = "calls-process"
FOLDER_FAILED = "calls-failed"
FOLDER_RESULTS = "calls-results"

#: Consecutive scans with an unchanged size before a non-readable audio file is
#: declared failed. Protects against files that are still being written.
MAX_STALLED_SCANS = 3


def probe_audio(path) -> bool:
    """Return True if the file is decodable audio right now.

    A cheap way to catch files that are still being written (e.g. an m4a stream
    capture that has not written its moov index yet) or are corrupt, *before*
    moving them through the pipeline.
    """
    try:
        import av

        with av.open(str(path), mode="r", metadata_errors="ignore") as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return False
            next(container.decode(stream), None)
        return True
    except Exception:  # noqa: BLE001 - any probe failure means "not usable now"
        return False


def _file_size(path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


class CallFolderProcessor:
    """Moves audio files through the calls-* folder workflow.

    calls-inbox    -> new calls to process
    calls-process  -> call currently being transcribed
    calls-failed   -> failures, with a <name>.txt error file
    calls-results  -> successes with the transcript (same name, .json + transcript file)

    Files are *moved* between folders, never copied.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        base_dir: str | os.PathLike = ".",
        output_formats: str | list[str] = "md",
        layout: str = "paragraph",
        words_per_caption: int = 6,
    ) -> None:
        if isinstance(output_formats, str):
            requested = [part for part in output_formats.split(",") if part]
        else:
            requested = output_formats
        formats = [fmt.strip().lower() for fmt in requested]
        invalid = [fmt for fmt in formats if fmt not in FORMATS]
        if invalid:
            raise ValueError(
                f"Unsupported output format(s) {invalid!r}. Choose one of: {', '.join(sorted(FORMATS))}"
            )
        if not formats:
            formats = ["md"]
        self._transcriber = transcriber
        self.output_formats = formats
        self.layout = layout
        self.words_per_caption = words_per_caption
        # File name -> (last_size, settled_unreadable_count). Tracks files that
        # are not processable yet (still being written, or corrupt) across scans.
        self._seen: dict[str, tuple[int, int]] = {}
        self.base = Path(base_dir).resolve()
        self.inbox = self.base / FOLDER_INBOX
        self.processing = self.base / FOLDER_PROCESS
        self.failed = self.base / FOLDER_FAILED
        self.results = self.base / FOLDER_RESULTS
        for folder in (self.inbox, self.processing, self.failed, self.results):
            folder.mkdir(parents=True, exist_ok=True)

    def recover_stale(self) -> list[Path]:
        """Move leftover files in calls-process back to calls-inbox for retry."""
        recovered: list[Path] = []
        for item in self.processing.iterdir():
            if item.is_file():
                os.replace(item, self.inbox / item.name)
                recovered.append(item)
        if recovered:
            logger.warning("Recovered %d stale file(s) from %s", len(recovered), FOLDER_PROCESS)
        return recovered

    def process_inbox(self) -> list[Path]:
        """Process every audio file currently in calls-inbox. Returns final paths."""
        finished: list[Path] = []
        for item in sorted(self.inbox.iterdir()):
            if not item.is_file() or item.name.startswith("."):
                continue
            try:
                path = self.process_one(item)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not process %s: %s", item, exc)
                continue
            if path is not None:
                finished.append(path)
        return finished

    def process_one(self, audio_path: Path) -> Path | None:
        """Run a single call through the pipeline.

        Returns where the audio ended up, or None if the file was deferred
        (left in calls-inbox because it is not decodable yet).
        """
        audio = audio_path.resolve()
        if audio.parent != self.inbox:
            raise ValueError(f"Audio file must be inside {self.inbox}: {audio}")

        name = audio.name
        stem = audio.stem
        suffix = audio.suffix.lower()

        if not suffix or suffix not in SUPPORTED_AUDIO_SUFFIXES:
            reason = f"Unsupported file type: {suffix or 'no extension'}"
            return self._fail(audio, reason, cause=None)

        status = self._readiness(audio)
        if status == "hold":
            return None
        if status == "fail":
            return self._fail(
                audio,
                "Audio file is unreadable (invalid or never finished writing): "
                "av.open/decode failed",
                cause=None,
            )
        working = self.processing / name
        try:
            os.replace(audio, working)
        except FileNotFoundError:
            logger.warning("%s disappeared before processing", audio)
            return audio

        try:
            logger.info("Transcribing %s", working)
            transcript = self._transcriber.transcribe(str(working))
        except Exception as exc:
            logger.exception("Transcription failed for %s", working)
            return self._fail(working, f"Transcription failed: {exc}", cause=exc)

        json_path = self.results / f"{stem}.json"
        json_path.write_text(
            json.dumps(transcript.to_dict(include_words=True), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for fmt in self.output_formats:
            path = self.results / f"{stem}.{fmt}"
            path.write_text(
                render(
                    transcript,
                    fmt,
                    layout=self.layout,
                    words_per_caption=self.words_per_caption,
                )
                + "\n",
                encoding="utf-8",
            )
        os.replace(working, self.results / name)
        logger.info("Processed %s -> %s", name, FOLDER_RESULTS)
        return self.results / name

    def _readiness(self, audio: Path) -> str:
        """Decide whether to process `audio` now, hold it, or fail it.

        Returns "process", "hold", or "fail" (the caller acts on it).

        1. First sighting: probe. A readable file is processed immediately
           (finished files dropped into the inbox work on the first scan). An
           unreadable one is recorded as a baseline and held in calls-inbox.
        2. Later sightings: the size check comes first. If the size changed the
           source is still writing, so we hold without even probing (this also
           catches header-first formats the probe cannot spot). Once the size
           has settled we probe: readable -> process; unreadable for
           MAX_STALLED_SCANS settled scans in a row -> fail.
        """
        name = audio.name
        current = _file_size(audio)

        previous = self._seen.get(name)
        if previous is None:
            if probe_audio(audio):
                return "process"
            self._seen[name] = (current, 0)
            logger.info(
                "Deferring %s: audio not decodable yet - is the source still writing?",
                name,
            )
            return "hold"

        last_size, settled_count = previous
        if last_size != current:
            self._seen[name] = (current, 0)
            logger.info(
                "Deferring %s: size changed (%d -> %d bytes) - still being written",
                name,
                last_size,
                current,
            )
            return "hold"

        settled_count += 1
        self._seen[name] = (current, settled_count)
        if probe_audio(audio):
            self._seen.pop(name, None)
            logger.info("%s became readable (settled at %d bytes)", name, current)
            return "process"

        if settled_count >= MAX_STALLED_SCANS:
            logger.warning(
                "%s stayed unreadable through %d settled scan(s); returning fail",
                name,
                settled_count,
            )
            self._seen.pop(name, None)
            return "fail"
        logger.info(
            "Deferring %s: readable check failed %d/%d times at unchanged size",
            name,
            settled_count,
            MAX_STALLED_SCANS,
        )
        return "hold"

    def _fail(self, audio: Path, message: str, cause: BaseException | None) -> Path:
        target = self.failed / audio.name
        try:
            os.replace(audio, target)
        except FileNotFoundError:
            target = audio
        self._write_error(target, message, cause)
        logger.error("Failed %s -> %s", target.name, FOLDER_FAILED)
        return target

    def _write_error(
        self, target: Path, message: str, cause: BaseException | None
    ) -> None:
        error_path = self.failed / f"{target.stem}.txt"
        content = (
            f"Audio file: {target.name}\n"
            f"Failed at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"Error: {message}\n"
        )
        if cause is not None:
            content += f"\nTraceback:\n{traceback.format_exc()}\n"
        try:
            error_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.error("Could not write error file %s: %s", error_path, exc)


def watch_folders(
    processor: CallFolderProcessor,
    interval: float = 5.0,
    once: bool = False,
) -> int:
    """Poll calls-inbox and process new calls until interrupted."""
    processor.recover_stale()
    processed = 0
    try:
        while True:
            handled = processor.process_inbox()
            processed += len(handled)
            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Stopped by user after processing %d file(s)", processed)
    return processed
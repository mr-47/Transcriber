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
    except ImportError as exc:  # pragma: no cover - av is a hard dependency
        # Do not let a missing dependency look like corrupt audio: that would
        # quietly move every file in the inbox to calls-failed.
        raise RuntimeError(
            "Audio readiness probing needs PyAV, which is not installed. "
            "Reinstall with `pip install -e .` (av is a dependency) or "
            "`pip install av`."
        ) from exc

    try:
        with av.open(str(path), mode="r", metadata_errors="ignore") as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return False
            next(container.decode(stream), None)
        return True
    except Exception:  # noqa: BLE001 - any probe failure means "not usable now"
        return False


def _file_stamp(path) -> tuple[int, int]:
    """(size, mtime_ns) identity of a file, or (0, 0) when unreadable.

    mtime_ns is included because size alone cannot tell a file that was
    *replaced* at the same size apart from one still being written: a source
    that rewrites a fixed-size stream (e.g. a circular capture) would otherwise
    keep the old stamp and skip the readiness gate.
    """
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns
    except OSError:
        return 0, 0


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
        language: str | None = None,
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
        # Language hint applied to every file. ``transcriber watch --language ru``
        # used to be accepted and then dropped on the floor, because nothing
        # between the CLI and this call carried it; whisper then fell back to
        # auto-detection per file.
        self.language = language
        # File name -> ((size, mtime_ns), settled_unreadable_count). Tracks
        # files that are not processable yet (still being written, or corrupt)
        # across scans. The mtime_ns part makes a same-size replacement bump
        # the stamp instead of passing as "unchanged".
        self._seen: dict[str, tuple[tuple[int, int], int]] = {}
        self.base = Path(base_dir).resolve()
        self.inbox = self.base / FOLDER_INBOX
        self.processing = self.base / FOLDER_PROCESS
        self.failed = self.base / FOLDER_FAILED
        self.results = self.base / FOLDER_RESULTS
        for folder in (self.inbox, self.processing, self.failed, self.results):
            folder.mkdir(parents=True, exist_ok=True)

    def recover_stale(self) -> list[Path]:
        """Move leftover audio files in calls-process back to calls-inbox for retry."""
        recovered: list[Path] = []
        for item in self.processing.iterdir():
            if item.is_file() and not item.name.startswith("."):
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
        self._prune_seen()
        return finished

    def _prune_seen(self) -> None:
        """Drop readiness state for files that are no longer in the inbox.

        A held file is usually deleted or moved by whatever is feeding the inbox
        (a cleanup script, the source app clearing its outbox). Nothing re-reads
        those entries afterwards, so a long-running ``watch`` would otherwise
        accumulate one entry per file ever seen -- a slow leak in a daemon that
        is meant to run for months. The inbox listing is the whole truth here:
        every name still present is still being watched, and every other name is
        stale by definition.
        """
        present = {item.name for item in self.inbox.iterdir() if item.is_file()}
        stale = self._seen.keys() - present
        for name in stale:
            del self._seen[name]
        if stale:
            logger.debug("Pruned %d readiness entries no longer in %s", len(stale), FOLDER_INBOX)

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
            self._seen.pop(name, None)
            return audio

        try:
            logger.info("Transcribing %s", working)
            transcript = self._transcriber.transcribe(str(working), language=self.language)
        except Exception as exc:
            logger.exception("Transcription failed for %s", working)
            return self._fail(working, f"Transcription failed: {exc}", cause=exc)

        try:
            out_stem = self._unique_stem(stem, suffix)
            json_path = self.results / f"{out_stem}.json"
            json_path.write_text(
                json.dumps(transcript.to_dict(include_words=True), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            for fmt in self.output_formats:
                path = self.results / f"{out_stem}.{fmt}"
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
            if out_stem != stem:
                logger.warning(
                    "Wrote transcripts as %s.* because %s.* already belongs to a different recording",
                    out_stem,
                    stem,
                )
            os.replace(working, self.results / name)
        except Exception as exc:  # noqa: BLE001 - a write/render failure fails the call the same way
            logger.exception("Finalising results failed for %s", working)
            return self._fail(working, f"Could not write results: {exc}", cause=exc)
        logger.info("Processed %s -> %s", name, FOLDER_RESULTS)
        return self.results / name

    def _unique_stem(self, stem: str, suffix: str) -> str:
        """Pick a transcript stem that will not clobber an existing transcript.

        Two recordings that share a stem but not an extension (``call.mp3`` and
        ``call.m4a``) both want ``call.json`` / ``call.md``. The second one used
        to overwrite the first: the audio survived in calls-results, its
        transcript vanished, and nothing was logged. Fall back to the full name
        with the extension turned into part of the stem (``call.m4a`` ->
        ``call_m4a``) so both transcripts coexist.
        """
        if not self._stem_taken(self.results, stem, suffix):
            return stem
        candidate = f"{stem}{suffix.replace('.', '_')}".replace(os.sep, "_")
        if not self._stem_taken(self.results, candidate, suffix):
            return candidate
        # Two files with an identical stem *and* extension cannot both be in
        # the inbox, but a re-run over an existing result can collide. Never
        # overwrite: add a counter.
        numbered = self._first_free(self.results, candidate, suffix)
        if numbered is None:
            raise RuntimeError(f"Could not find a free transcript name for {stem}{suffix}")
        return numbered

    @staticmethod
    def _stem_taken(
        folder: Path, stem: str, suffix: str, *, report_claims: bool = False
    ) -> bool:
        """True when this stem is already owned by a *different* recording.

        Ownership is read off the audio sitting next to the transcript rather
        than guessed. ``call.*`` belongs to whichever ``call.<ext>`` audio is
        next to it, so a stem is claimed when a different extension owns it. An
        unclaimed stem (no audio there at all) is free even if a transcript
        file of that name exists, which is what makes re-running a recording --
        where the audio has been moved back to the inbox -- overwrite in place
        instead of accumulating numbered copies.

        Failure reports are stricter: ``report_claims`` also lets an existing
        ``.txt`` own its name even though its audio is gone, so the record of
        the original failure is never overwritten.
        """
        if report_claims and not (folder / f"{stem}.txt").exists():
            return False
        owners = {
            item.suffix.lower()
            for item in folder.iterdir()
            if item.is_file() and item.stem == stem and item.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        }
        return bool(owners) and owners != {suffix}

    def _first_free(
        self, folder: Path, candidate: str, suffix: str, *, report_claims: bool = False
    ) -> str | None:
        """First free ``<candidate>_<n>`` name in 2..999, else None."""
        for n in range(2, 1000):
            numbered = f"{candidate}_{n}"
            if not self._stem_taken(folder, numbered, suffix, report_claims=report_claims):
                return numbered
        return None

    def _readiness(self, audio: Path) -> str:
        """Decide whether to process `audio` now, hold it, or fail it.

        Returns "process", "hold", or "fail" (the caller acts on it).

        1. First sighting: probe. A readable file is processed immediately
           (finished files dropped into the inbox work on the first scan). An
           unreadable one is recorded as a baseline and held in calls-inbox.
        2. Later sightings: the stamp check comes first. If either the size or
           the mtime changed the source is still writing (or the file was
           replaced), so we hold without even probing (this also catches
           header-first formats the probe cannot spot). Once the stamp has
           settled we probe: readable -> process; unreadable for
           MAX_STALLED_SCANS settled scans in a row -> fail.
        """
        name = audio.name
        current = _file_stamp(audio)

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

        last_stamp, settled_count = previous
        if last_stamp != current:
            self._seen[name] = (current, 0)
            logger.info(
                "Deferring %s: file changed (size %d->%d, mtime_ns %d->%d) "
                "- still being written or replaced",
                name,
                last_stamp[0],
                current[0],
                last_stamp[1],
                current[1],
            )
            return "hold"

        settled_count += 1
        self._seen[name] = (current, settled_count)
        if probe_audio(audio):
            self._seen.pop(name, None)
            logger.info("%s became readable (settled at %d bytes)", name, current[0])
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
            "Deferring %s: readable check failed %d/%d times at unchanged stamp",
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
        # Same collision as the transcripts: `call.mp3` and `call.m4a` both fail
        # to `call.txt`, and the first reason would be lost. Disambiguate the
        # report name the same way the transcript name is disambiguated.
        error_path = self.failed / f"{self._unique_failed_stem(target)}.txt"
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

    def _unique_failed_stem(self, target: Path) -> str:
        """Stem for the ``.txt`` report, so no report is ever lost or destructive.

        Two things can go wrong with the obvious ``<stem>.txt``:

        * The input is itself a ``.txt`` (an unsupported file type, which is
          exactly what lands in calls-failed). The report path is then the audio
          path, and writing the report destroys the file it describes.
        * Two recordings share a stem but not an extension, so one report would
          overwrite the other and its reason would be lost.
        """
        stem, suffix = target.stem, target.suffix.lower()
        # `.txt` in, `.txt` report out: pick a name that cannot be the input.
        base = f"{stem}.error" if suffix == ".txt" else stem
        if not self._stem_taken(self.failed, base, suffix, report_claims=True):
            return base
        candidate = f"{base}{suffix.replace('.', '_')}"
        if not self._stem_taken(self.failed, candidate, suffix, report_claims=True):
            return candidate
        numbered = self._first_free(self.failed, candidate, suffix, report_claims=True)
        return candidate if numbered is None else numbered


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
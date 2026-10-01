import json
import sys
import types
import wave
from pathlib import Path

import pytest

from transcriber.models import Transcript, Utterance
from transcriber.pipeline import MAX_STALLED_SCANS, CallFolderProcessor


def _write_audio(path) -> None:
    """Write a tiny valid PCM WAV so the pipeline's pre-flight probe passes."""
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence


class _FakeTranscriber:
    def __init__(self, fail_names: set[str] | None = None) -> None:
        self.fail_names = fail_names or set()
        self.last_path = None
        self.languages: list[str | None] = []

    def transcribe(self, path: str, language=None) -> Transcript:
        self.last_path = path
        self.languages.append(language)
        if any(name in path for name in self.fail_names):
            raise RuntimeError("boom")
        return Transcript(
            language="en",
            language_probability=0.99,
            duration=1.5,
            segments=[],
            utterances=[Utterance(speaker="SPEAKER_00", start=0.0, end=1.0, text="hello")],
        )


def _proc(tmp_path, fail_names: set[str] | None = None) -> CallFolderProcessor:
    return CallFolderProcessor(_FakeTranscriber(fail_names=fail_names), tmp_path)


def test_success_moves_through_folders(tmp_path):
    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "call.mp3"
    _write_audio(inbox_file)

    result = proc.process_one(inbox_file.resolve())

    assert result == (tmp_path / "calls-results" / "call.mp3").resolve()
    assert (tmp_path / "calls-results" / "call.mp3").exists()
    assert (tmp_path / "calls-results" / "call.json").exists()
    assert (tmp_path / "calls-results" / "call.md").exists()
    assert not inbox_file.exists()
    assert not (tmp_path / "calls-process" / "call.mp3").exists()
    assert not list((tmp_path / "calls-failed").glob("*"))

    transcript = json.loads((tmp_path / "calls-results" / "call.json").read_text())
    assert transcript["utterances"][0]["text"] == "hello"
    markdown = (tmp_path / "calls-results" / "call.md").read_text()
    assert "SPEAKER_00" in markdown
    assert "hello" in markdown
    assert proc._transcriber.last_path == str(tmp_path / "calls-process" / "call.mp3")


def test_success_with_txt_format(tmp_path):
    proc = CallFolderProcessor(_FakeTranscriber(), tmp_path, output_formats="txt")
    inbox_file = tmp_path / "calls-inbox" / "call.mp3"
    _write_audio(inbox_file)

    proc.process_one(inbox_file.resolve())

    txt = (tmp_path / "calls-results" / "call.txt").read_text()
    assert txt == "SPEAKER_00: hello\n"
    assert not (tmp_path / "calls-results" / "call.md").exists()


def test_success_multiple_formats(tmp_path):
    proc = CallFolderProcessor(
        _FakeTranscriber(), tmp_path, output_formats=["txt", "html", "srt"]
    )
    inbox_file = tmp_path / "calls-inbox" / "call.mp3"
    _write_audio(inbox_file)

    proc.process_one(inbox_file.resolve())

    results = tmp_path / "calls-results"
    assert (results / "call.json").exists()
    assert (results / "call.txt").exists()
    assert (results / "call.html").exists()
    assert (results / "call.srt").exists()
    assert not (results / "call.md").exists()
    assert (results / "call.srt").read_text().startswith("1\n00:00:00,000")


def test_multi_format_comma_string(tmp_path):
    proc = CallFolderProcessor(_FakeTranscriber(), tmp_path, output_formats="md,txt")
    assert proc.output_formats == ["md", "txt"]


def test_failure_moves_to_failed_with_error_file(tmp_path):
    proc = _proc(tmp_path, fail_names={"call.wav"})
    inbox_file = tmp_path / "calls-inbox" / "call.wav"
    _write_audio(inbox_file)

    result = proc.process_one(inbox_file.resolve())

    assert result == (tmp_path / "calls-failed" / "call.wav").resolve()
    assert (tmp_path / "calls-failed" / "call.wav").exists()
    assert not list((tmp_path / "calls-results").glob("*"))
    assert not inbox_file.exists()
    error = (tmp_path / "calls-failed" / "call.txt").read_text()
    assert "boom" in error
    assert "RuntimeError" in error
    assert "Traceback" in error


def test_result_write_failure_moves_to_failed_with_report(tmp_path, monkeypatch):
    """A failure while writing results must fail the call, not strand its audio."""
    import transcriber.pipeline as pipeline

    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "call.wav"
    _write_audio(inbox_file)

    def boom_render(*args, **kwargs):
        raise ValueError("words_per_caption must be at least 1")

    monkeypatch.setattr(pipeline, "render", boom_render)

    result = proc.process_one(inbox_file.resolve())

    assert result == (tmp_path / "calls-failed" / "call.wav").resolve()
    assert (tmp_path / "calls-failed" / "call.wav").exists()
    assert not list((tmp_path / "calls-process").iterdir()), "audio must not be stranded"
    error = (tmp_path / "calls-failed" / "call.txt").read_text()
    assert "Could not write results" in error


def test_unsupported_type_fails(tmp_path):
    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "notes.pdf"
    inbox_file.write_bytes(b"nope")

    proc.process_one(inbox_file.resolve())

    assert (tmp_path / "calls-failed" / "notes.pdf").exists()
    error = (tmp_path / "calls-failed" / "notes.txt").read_text()
    assert "Unsupported file type" in error


def test_recover_stale_moves_back_to_inbox(tmp_path):
    proc = _proc(tmp_path)
    stale = tmp_path / "calls-process" / "call.mp3"
    stale.write_bytes(b"audio")

    recovered = proc.recover_stale()

    assert recovered == [stale.resolve()]
    assert not stale.exists()
    assert (tmp_path / "calls-inbox" / "call.mp3").exists()


def test_recover_stale_ignores_dotfiles(tmp_path):
    proc = _proc(tmp_path)
    keep = tmp_path / "calls-process" / ".gitkeep"
    keep.write_bytes(b"")

    recovered = proc.recover_stale()

    assert recovered == []
    assert keep.exists()
    assert not (tmp_path / "calls-inbox" / ".gitkeep").exists()


def test_process_inbox_handles_each_file(tmp_path):
    proc = _proc(tmp_path, fail_names={"bad.wav"})
    good = tmp_path / "calls-inbox" / "good.mp3"
    bad = tmp_path / "calls-inbox" / "bad.wav"
    _write_audio(good)
    _write_audio(bad)

    finished = proc.process_inbox()

    assert len(finished) == 2
    assert (tmp_path / "calls-results" / "good.mp3").exists()
    assert (tmp_path / "calls-results" / "good.md").exists()
    assert (tmp_path / "calls-failed" / "bad.wav").exists()
    assert (tmp_path / "calls-failed" / "bad.txt").exists()


def test_hidden_files_in_inbox_are_never_picked_up(tmp_path):
    proc = _proc(tmp_path)
    _write_audio(tmp_path / "calls-inbox" / ".hidden-call.mp3")

    assert proc.process_inbox() == []
    assert (tmp_path / "calls-inbox" / ".hidden-call.mp3").exists()
    assert not list((tmp_path / "calls-failed").glob("*"))


def test_language_is_forwarded_to_every_transcription(tmp_path):
    transcriber = _FakeTranscriber()
    proc = CallFolderProcessor(transcriber, tmp_path, language="ru")
    _write_audio(tmp_path / "calls-inbox" / "a.mp3")
    _write_audio(tmp_path / "calls-inbox" / "b.mp3")

    proc.process_inbox()

    assert transcriber.languages == ["ru", "ru"]


def test_language_defaults_to_none(tmp_path):
    transcriber = _FakeTranscriber()
    proc = CallFolderProcessor(transcriber, tmp_path)
    _write_audio(tmp_path / "calls-inbox" / "a.mp3")

    proc.process_inbox()

    assert transcriber.languages == [None]


def test_unreadable_file_is_deferred_not_failed(tmp_path):
    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "call.m4a"
    inbox_file.write_bytes(b"\x00\x01\x02\x03" * 4)

    result = proc.process_one(inbox_file.resolve())

    assert result is None
    assert inbox_file.exists()
    assert not list((tmp_path / "calls-failed").glob("call*"))
    assert not list((tmp_path / "calls-process").glob("call*"))
    assert not list((tmp_path / "calls-results").glob("call*"))


def test_deferred_file_is_processed_once_readable(tmp_path):
    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "call.m4a"
    inbox_file.write_bytes(b"\x00\x01\x02\x03" * 4)

    assert proc.process_one(inbox_file.resolve()) is None

    _write_audio(inbox_file)
    assert proc.process_one(inbox_file.resolve()) is None  # size changed -> hold

    result = proc.process_one(inbox_file.resolve())

    assert result == (tmp_path / "calls-results" / "call.m4a").resolve()
    assert "call.m4a" not in proc._seen


def test_growing_file_is_held_and_never_fails(tmp_path):
    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "call.m4a"
    data = b"\x00\x01\x02\x03" * 8
    inbox_file.write_bytes(data)

    for _ in range(MAX_STALLED_SCANS + 2):
        inbox_file.write_bytes(inbox_file.read_bytes() + b"\x00\x00")
        assert proc.process_one(inbox_file.resolve()) is None

    assert inbox_file.exists()
    assert not list((tmp_path / "calls-failed").glob("call*"))
    assert not list((tmp_path / "calls-process").glob("call*"))
    assert not list((tmp_path / "calls-results").glob("call*"))


def test_same_size_replacement_is_held_then_reprobed(tmp_path):
    """A file replaced at the same byte size must not be treated as unchanged."""
    import os

    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "call.m4a"
    inbox_file.write_bytes(b"\x00\x01\x02\x03" * 4)
    assert proc.process_one(inbox_file.resolve()) is None  # first sight, unreadable

    # Replace with a new, still-unreadable payload of the same size.
    replacement = b"\xff\xfe\xfd\xfc" * 4
    inbox_file.write_bytes(replacement)
    os.utime(inbox_file, ns=(1_700_000_000, 1_700_000_001_000))

    # Same size, new mtime -> treated as "changed", held without failing.
    assert proc.process_one(inbox_file.resolve()) is None
    assert not list((tmp_path / "calls-failed").glob("call*"))


def test_stalled_unreadable_file_goes_to_failed(tmp_path):
    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "call.m4a"
    inbox_file.write_bytes(b"\x00\x01\x02\x03" * 4)

    for _ in range(MAX_STALLED_SCANS):
        assert proc.process_one(inbox_file.resolve()) is None
    assert inbox_file.exists()

    result = proc.process_one(inbox_file.resolve())

    assert result == (tmp_path / "calls-failed" / "call.m4a").resolve()
    assert not inbox_file.exists()
    error = (tmp_path / "calls-failed" / "call.txt").read_text()
    assert "unreadable" in error


def test_invalid_output_format_raises(tmp_path):
    try:
        CallFolderProcessor(_FakeTranscriber(), tmp_path, output_formats="pdf")
    except ValueError as exc:
        assert "pdf" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_empty_output_formats_default_to_md(tmp_path):
    proc = CallFolderProcessor(_FakeTranscriber(), tmp_path, output_formats="")
    assert proc.output_formats == ["md"]


def test_process_rejects_a_file_outside_the_inbox(tmp_path):
    proc = _proc(tmp_path)
    outside = tmp_path / "elsewhere.wav"
    _write_audio(outside)

    with pytest.raises(ValueError, match="must be inside"):
        proc.process_one(outside.resolve())


def test_audio_vanishing_between_readiness_and_move_is_handled(tmp_path, monkeypatch):
    import transcriber.pipeline as pipeline

    proc = _proc(tmp_path)
    inbox_file = tmp_path / "calls-inbox" / "call.mp3"
    _write_audio(inbox_file)

    def gone(src, dst):
        raise FileNotFoundError("file vanished mid-flight")

    monkeypatch.setattr(pipeline.os, "replace", gone)

    result = proc.process_one(inbox_file.resolve())

    assert result == inbox_file.resolve(), "a vanished file must not be treated as processed"
    assert "call.mp3" not in proc._seen


def test_process_inbox_survives_unexpected_errors(tmp_path):
    proc = _proc(tmp_path)
    _write_audio(tmp_path / "calls-inbox" / "a.mp3")

    def boom(path):
        raise RuntimeError("unexpected")

    proc.process_one = boom  # type: ignore[method-assign]

    assert proc.process_inbox() == []


def test_file_stamp_is_zero_for_an_unreadable_path(tmp_path):
    import transcriber.pipeline as pipeline

    assert pipeline._file_stamp(tmp_path / "gone.wav") == (0, 0)


def test_unique_stem_falls_back_to_numbering(tmp_path):
    """If every plain variant is claimed, a numbered suffix must win."""
    proc = _proc(tmp_path)
    (tmp_path / "calls-results" / "call.m4a").write_bytes(b"a")      # claims "call"
    (tmp_path / "calls-results" / "call_mp3.m4a").write_bytes(b"b")  # claims "call_mp3"
    _write_audio(tmp_path / "calls-inbox" / "call.mp3")

    proc.process_inbox()

    results = tmp_path / "calls-results"
    assert (results / "call_mp3_2.json").exists(), "numbered fallback must kick in"
    assert (results / "call.mp3").exists()


def test_unique_failed_stem_falls_back_to_numbering(tmp_path):
    proc = _proc(tmp_path, fail_names={"call"})
    failed = tmp_path / "calls-failed"
    (failed / "call.m4a").write_bytes(b"a")
    (failed / "call.txt").write_text("first reason")
    (failed / "call_mp3.m4a").write_bytes(b"b")
    (failed / "call_mp3.txt").write_text("second reason")
    _write_audio(tmp_path / "calls-inbox" / "call.mp3")

    proc.process_inbox()

    assert (failed / "call_mp3_2.txt").exists(), "report numbering must never clobber"


def test_probe_accepts_a_real_wav(tmp_path):
    import transcriber.pipeline as pipeline

    path = tmp_path / "ok.wav"
    _write_audio(path)

    assert pipeline.probe_audio(path) is True


def test_probe_rejects_garbage(tmp_path):
    import transcriber.pipeline as pipeline

    path = tmp_path / "bad.m4a"
    path.write_bytes(b"\x00\x01\x02\x03" * 16)

    assert pipeline.probe_audio(path) is False


def test_probe_rejects_a_container_without_an_audio_stream(tmp_path, monkeypatch):
    import transcriber.pipeline as pipeline

    class _Container:
        streams: list = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_av = types.ModuleType("av")
    fake_av.open = lambda *args, **kwargs: _Container()
    monkeypatch.setitem(sys.modules, "av", fake_av)

    path = tmp_path / "video.mkv"
    path.write_bytes(b"x")

    assert pipeline.probe_audio(path) is False


def test_failure_vanishing_before_the_move_to_failed(tmp_path, monkeypatch):
    """A file that fails *and* vanishes mid-move must still get an error report."""
    import transcriber.pipeline as pipeline

    proc = _proc(tmp_path, fail_names={"call"})
    inbox_file = tmp_path / "calls-inbox" / "call.wav"
    _write_audio(inbox_file)
    working = tmp_path / "calls-process"
    real_replace = pipeline.os.replace

    def vanish_to_failed(src, dst):
        if "calls-failed" in str(dst):
            raise FileNotFoundError("gone between transcribe and fail-move")
        return real_replace(src, dst)

    monkeypatch.setattr(pipeline.os, "replace", vanish_to_failed)

    result = proc.process_one(inbox_file.resolve())

    assert result == (working / "call.wav").resolve()
    assert (tmp_path / "calls-failed" / "call.txt").exists(), "the error report must still be written"


def test_error_report_write_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    import logging

    import transcriber.pipeline as pipeline

    proc = _proc(tmp_path, fail_names={"call"})
    inbox_file = tmp_path / "calls-inbox" / "call.wav"
    _write_audio(inbox_file)
    real_write_text = pipeline.Path.write_text

    def broken_write(self, *args, **kwargs):
        if self.suffix == ".txt" and "calls-failed" in str(self):
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(pipeline.Path, "write_text", broken_write)

    with caplog.at_level(logging.ERROR, logger="transcriber.pipeline"):
        proc.process_one(inbox_file.resolve())

    assert "Could not write error file" in caplog.text


def test_readiness_state_does_not_outlive_the_file(tmp_path, monkeypatch):
    """A held file that is deleted must not keep an entry forever."""
    import transcriber.pipeline as pipeline

    monkeypatch.setattr(pipeline, "probe_audio", lambda _path: False)
    proc = _proc(tmp_path)

    for round_index in range(3):
        for i in range(5):
            _write_audio(tmp_path / "calls-inbox" / f"call-{round_index}-{i}.wav")
        proc.process_inbox()
        assert len(proc._seen) == 5, proc._seen
        for i in range(5):
            (tmp_path / "calls-inbox" / f"call-{round_index}-{i}.wav").unlink()
        proc.process_inbox()
        assert proc._seen == {}, f"entries outlived their files: {proc._seen}"


def test_pruning_keeps_a_file_that_is_still_being_written(tmp_path, monkeypatch):
    import transcriber.pipeline as pipeline

    monkeypatch.setattr(pipeline, "probe_audio", lambda _path: False)
    proc = _proc(tmp_path)
    _write_audio(tmp_path / "calls-inbox" / "half-written.wav")

    proc.process_inbox()
    assert list(proc._seen) == ["half-written.wav"], proc._seen
    first = proc._seen["half-written.wav"]

    proc.process_inbox()
    assert "half-written.wav" in proc._seen, "a live held file must stay tracked"
    assert proc._seen["half-written.wav"][1] == first[1] + 1, "the settled count must advance"


def test_a_recreated_file_starts_fresh_after_a_scan_pruned_it(tmp_path, monkeypatch):
    """A file unwatchable in one round and replaced later must not inherit the
    old settled count -- it should get the full fresh-file grace period."""
    import transcriber.pipeline as pipeline

    monkeypatch.setattr(pipeline, "probe_audio", lambda _path: False)
    proc = _proc(tmp_path)
    held = tmp_path / "calls-inbox" / "call.m4a"
    held.write_bytes(b"\x00\x01\x02\x03" * 4)

    proc.process_one(held.resolve())
    assert list(proc._seen) == ["call.m4a"]

    held.unlink()
    proc.process_inbox()  # prunes now-missing file
    assert proc._seen == {}

    held.write_bytes(b"\xff\xfe\xfd\xfc" * 4)
    assert proc.process_one(held.resolve()) is None
    assert proc._seen["call.m4a"][1] == 0, "a re-created file must restart the settle count"


def test_same_stem_different_extensions_keep_both_transcripts(tmp_path):
    """`call.mp3` and `call.m4a` must not overwrite each other's transcript."""
    proc = _proc(tmp_path)
    proc.output_formats = ["md", "txt"]
    _write_audio(tmp_path / "calls-inbox" / "call.m4a")
    _write_audio(tmp_path / "calls-inbox" / "call.mp3")

    proc.process_inbox()

    results = tmp_path / "calls-results"
    assert (results / "call.m4a").exists() and (results / "call.mp3").exists()
    assert (results / "call.json").exists(), "first recording keeps the plain name"
    assert (results / "call_mp3.json").exists(), "second is disambiguated, not lost"
    assert (results / "call.md").exists() and (results / "call_mp3.md").exists()
    assert (results / "call.txt").exists() and (results / "call_mp3.txt").exists()


def test_rerunning_the_same_recording_overwrites_in_place(tmp_path):
    """Re-processing a recording must not accumulate numbered copies."""
    proc = _proc(tmp_path)
    _write_audio(tmp_path / "calls-inbox" / "call.m4a")
    _write_audio(tmp_path / "calls-inbox" / "call.mp3")
    proc.process_inbox()

    _write_audio(tmp_path / "calls-inbox" / "call.mp3")
    proc.process_inbox()

    results = tmp_path / "calls-results"
    assert sorted(p.name for p in results.glob("call_mp3*")) == ["call_mp3.json", "call_mp3.md"]

    assert not list(results.glob("*_2*")), "a re-run must overwrite, not fork"


def test_same_stem_failures_keep_both_error_reports(tmp_path):
    proc = _proc(tmp_path, fail_names={"call"})
    _write_audio(tmp_path / "calls-inbox" / "call.m4a")
    _write_audio(tmp_path / "calls-inbox" / "call.mp3")

    proc.process_inbox()

    failed = tmp_path / "calls-failed"
    assert (failed / "call.m4a").exists() and (failed / "call.mp3").exists()
    reports = sorted(p.name for p in failed.glob("*.txt"))
    assert reports == ["call.txt", "call_mp3.txt"]


def test_txt_input_failure_does_not_destroy_the_file(tmp_path):
    """A `.txt` in the inbox fails as unsupported; the report must not destroy it."""
    proc = _proc(tmp_path)
    note = tmp_path / "calls-inbox" / "notes.txt"
    note.write_text("keep me")

    proc.process_one(note.resolve())

    failed = tmp_path / "calls-failed"
    assert (failed / "notes.txt").read_text() == "keep me"
    report = (failed / "notes.error.txt").read_text()
    assert "Unsupported file type" in report


def test_watch_recovers_leftovers_before_the_first_scan(monkeypatch):
    import transcriber.pipeline as pipeline_module

    events: list[str] = []
    _sleep_that_stops_after(monkeypatch, times=1, events=events)

    assert pipeline_module.watch_folders(_StubProcessor(events, ["a.mp3"]), interval=0.01) == 1
    assert events == ["recover", "scan", "sleep"]


def test_watch_exits_cleanly_on_ctrl_c(monkeypatch, caplog):
    import logging

    import transcriber.pipeline as pipeline_module

    events: list[str] = []
    _sleep_that_stops_after(monkeypatch, times=2, events=events)

    with caplog.at_level(logging.INFO, logger="transcriber.pipeline"):
        processed = pipeline_module.watch_folders(_StubProcessor(events, ["a.mp3", "b.mp3"]), interval=5.0)

    assert processed == 4, "both scans before the interrupt should be counted"
    assert events == ["recover", "scan", "sleep", "scan", "sleep"]
    assert "Stopped by user" in caplog.text


def test_watch_once_scans_a_single_time(monkeypatch):
    import transcriber.pipeline as pipeline_module

    events: list[str] = []
    _sleep_that_stops_after(monkeypatch, times=99, events=events)

    assert pipeline_module.watch_folders(_StubProcessor(events, ["a.mp3"]), interval=0.01, once=True) == 1
    assert events == ["recover", "scan"], "--once must not sleep at all"


class _StubProcessor:
    """Just enough of `CallFolderProcessor` to drive the `watch_folders` loop."""

    def __init__(self, events: list[str], per_scan: list[str]) -> None:
        self.events = events
        self.per_scan = per_scan

    def recover_stale(self) -> list:
        self.events.append("recover")
        return []

    def process_inbox(self) -> list:
        self.events.append("scan")
        return [Path(name) for name in self.per_scan]


def _sleep_that_stops_after(monkeypatch, times: int, events: list[str]) -> None:
    """Replace the loop's sleep so the test terminates, and cannot hang."""
    from transcriber import pipeline as pipeline_module

    calls = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        calls["n"] += 1
        events.append("sleep")
        if calls["n"] >= times:
            raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module.time, "sleep", fake_sleep)
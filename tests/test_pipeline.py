import json
import wave

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

    def transcribe(self, path: str, language=None) -> Transcript:
        self.last_path = path
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
"""Top-level CLI surface: the shared flags and the command dispatch.

Nothing here loads a model or binds a port; `--version` is handled by argparse
and exits before any real work, so these are safe on a base install.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from transcriber import __version__
from transcriber.cli import main


@pytest.mark.parametrize("argv", [["--version"], ["watch", "--version"]])
def test_version_flag_reports_the_package_version(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"transcriber {__version__}"


def test_version_is_exposed_on_the_package() -> None:
    assert isinstance(__version__, str) and __version__


def test_python_dash_m_transcriber_dispatches_to_main() -> None:
    """`python -m transcriber` is a second entry point and is wired separately."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "transcriber", "--version"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"transcriber {__version__}"


def test_missing_audio_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2


def test_non_positive_caption_words_is_rejected_at_parse_time(capsys: pytest.CaptureFixture[str]) -> None:
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit) as excinfo:
            main(["call.mp3", "--format", "srt", "--caption-words", bad])
        assert excinfo.value.code == 2
        assert "must be 1 or greater" in capsys.readouterr().err


def test_non_numeric_caption_words_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["call.mp3", "--caption-words", "abc"])

    assert excinfo.value.code == 2
    assert "expected a whole number" in capsys.readouterr().err


def test_non_numeric_interval_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["watch", "--interval", "abc"])

    assert excinfo.value.code == 2
    assert "expected a number" in capsys.readouterr().err


def test_watch_with_an_empty_format_part_defaults_to_md(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--format ','` is all empties and must fall back to the default, not crash."""
    recorded = _record_watch_run(monkeypatch)

    assert main(["watch", "--once", "--format", ","]) == 0

    assert recorded["processor"]["output_formats"] == ["md"]


def test_python_m_invokes_the_cli(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    monkeypatch.setattr(sys, "argv", ["transcriber", "--version"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("transcriber", run_name="__main__")

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith("transcriber ")


def test_watch_rejects_a_non_positive_interval(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["watch", "--interval", "-1"])

    assert excinfo.value.code == 2
    assert "must be greater than 0" in capsys.readouterr().err


def test_valid_positive_values_are_accepted() -> None:
    from transcriber.cli import build_parser, build_watch_parser

    assert build_parser().parse_args(["a.mp3", "--caption-words", "1"]).caption_words == 1
    assert build_watch_parser().parse_args(["--interval", "0.5"]).interval == 0.5


def _record_watch_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Drive `_handle_watch` with the engine and the loop replaced.

    Parsing a flag is not the same as forwarding it. `--language` was accepted
    by argparse and then dropped before it reached the processor, and no test
    caught it because the processor test constructed the processor directly --
    one layer below the bug. Every flag that crosses the CLI boundary needs an
    assertion at this level, not just at its destination.
    """
    from transcriber import cli

    recorded: dict[str, object] = {}

    class _StubTranscriber:
        def __init__(self, settings) -> None:  # noqa: ANN001
            recorded["settings"] = settings

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

    def stub_processor(*args, **kwargs):  # noqa: ANN002, ANN003
        recorded["transcriber"] = args[0]
        recorded["base_dir"] = args[1]
        recorded["processor"] = kwargs
        return object()

    monkeypatch.setattr(cli, "Transcriber", _StubTranscriber)
    monkeypatch.setattr(cli, "CallFolderProcessor", stub_processor)
    monkeypatch.setattr(cli, "watch_folders", lambda *_a, **_k: 0)
    return recorded


def test_watch_forwards_every_flag_to_the_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _record_watch_run(monkeypatch)

    assert main(["watch", "--language", "ru", "--format", "srt", "--layout", "sentence", "--caption-words", "3"]) == 0

    assert recorded["processor"]["language"] == "ru"
    assert recorded["processor"]["output_formats"] == ["srt"]
    assert recorded["processor"]["layout"] == "sentence"
    assert recorded["processor"]["words_per_caption"] == 3


def test_watch_passes_the_directory_through(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _record_watch_run(monkeypatch)

    assert main(["watch", "--dir", "/tmp/transcriber-cli-test"]) == 0

    assert recorded["base_dir"] == "/tmp/transcriber-cli-test"
    assert recorded["processor"]["language"] is None


def test_watch_honours_no_diarization_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBER_HF_TOKEN", "hf_test")
    recorded = _record_watch_run(monkeypatch)

    assert main(["watch", "--no-diarization", "--model", "small"]) == 0

    assert recorded["settings"].hf_token == ""
    assert recorded["settings"].whisper_model == "small"


def _record_one_shot(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Drive `_handle_transcript` with the engine replaced.

    The one-shot path is a second, independent copy of the flag plumbing.
    """
    from transcriber import cli
    from transcriber.models import Segment, Transcript, Utterance

    recorded: dict[str, object] = {}

    class _StubTranscriber:
        def __init__(self, settings) -> None:  # noqa: ANN001
            recorded["settings"] = settings

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

        def transcribe(
            self, audio_path: str, language: str | None = None, word_timestamps: bool = True
        ) -> Transcript:
            recorded["audio"] = audio_path
            recorded["language"] = language
            recorded["word_timestamps"] = word_timestamps
            return Transcript(
                language="ru",
                language_probability=0.9,
                duration=2.0,
                segments=[Segment(0.0, 2.0, "privet")],
                utterances=[Utterance("SPEAKER_00", 0.0, 2.0, "privet")],
            )

    monkeypatch.setattr(cli, "Transcriber", _StubTranscriber)
    return recorded


def test_one_shot_forwards_language_and_diarization(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("TRANSCRIBER_HF_TOKEN", "hf_test")
    recorded = _record_one_shot(monkeypatch)

    assert main(["some.mp3", "--language", "de", "--no-diarization", "--model", "small"]) == 0

    assert recorded["settings"].hf_token == ""
    assert recorded["language"] == "de"
    assert recorded["audio"] == "some.mp3"
    assert recorded["settings"].whisper_model == "small"
    assert "privet" in capsys.readouterr().out


def test_one_shot_diarizes_by_default(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("TRANSCRIBER_HF_TOKEN", "hf_test")
    recorded = _record_one_shot(monkeypatch)

    assert main(["some.mp3"]) == 0

    assert recorded["language"] is None
    assert recorded["settings"].hf_token == "hf_test"


def test_one_shot_writes_the_output_file_and_json(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _record_one_shot(monkeypatch)
    out = tmp_path / "call.md"
    js = tmp_path / "call.json"

    assert main(["some.mp3", "-o", str(out), "--json", str(js), "--words"]) == 0

    assert recorded["word_timestamps"] is True
    assert out.read_text().strip() != ""
    assert "privet" in out.read_text()
    payload = json.loads(js.read_text())
    assert payload["language"] == "ru"
    assert payload["segments"], "--words should include the word array"


def test_one_shot_skips_word_timestamps_unless_json_and_words(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = _record_one_shot(monkeypatch)

    assert main(["some.mp3"]) == 0
    assert recorded["word_timestamps"] is False

    js = tmp_path / "call.json"
    assert main(["some.mp3", "--json", str(js)]) == 0
    assert recorded["word_timestamps"] is False


def _record_render(monkeypatch: pytest.MonkeyPatch, recorded: dict) -> None:
    """Replace `render` with a spy so flag forwarding through `_handle_transcript`
    is observed at the render boundary rather than in the transcript text."""
    from transcriber import cli

    def spy_render(transcript, fmt, layout, timestamps, words_per_caption):
        recorded.update(
            fmt=fmt,
            layout=layout,
            timestamps=timestamps,
            words_per_caption=words_per_caption,
        )
        return "rendered"

    monkeypatch.setattr(cli, "render", spy_render)


def test_one_shot_forwards_timestamps_layout_and_caption_words(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _record_one_shot(monkeypatch)
    recorded: dict = {}
    _record_render(monkeypatch, recorded)

    assert main(["some.mp3", "--timestamps", "--layout", "sentence", "--caption-words", "4"]) == 0

    assert recorded["fmt"] == "md"
    assert recorded["layout"] == "sentence"
    assert recorded["timestamps"] is True
    assert recorded["words_per_caption"] == 4
    assert "rendered" in capsys.readouterr().out


def test_one_shot_output_suffix_beats_the_format_flag(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """`-o calls.srt` must render SRT even when `--format md` is given."""
    _record_one_shot(monkeypatch)
    out = tmp_path / "calls.srt"

    assert main(["some.mp3", "--format", "md", "-o", str(out)]) == 0

    assert Path.read_text(out).startswith("1\n00:00:00,000")


def test_one_shot_output_without_extension_gets_the_format(tmp_path, monkeypatch) -> None:
    _record_one_shot(monkeypatch)
    out = tmp_path / "call"

    assert main(["some.mp3", "-o", str(out)]) == 0

    assert (tmp_path / "call.md").exists()


def test_one_shot_unknown_output_extension_warns_but_writes(
    tmp_path, monkeypatch, capsys
) -> None:
    _record_one_shot(monkeypatch)
    out = tmp_path / "call.mp3"

    assert main(["some.mp3", "-o", str(out)]) == 0

    assert out.exists(), "the user-chosen filename must still be written"
    assert "privet" in out.read_text()
    err = capsys.readouterr().err
    assert "'mp3' is not a transcript format" in err


def test_resolve_format_prefers_the_output_extension() -> None:
    from transcriber.cli import _resolve_format

    assert _resolve_format("md", Path("out.srt")) == "srt"
    assert _resolve_format("md", Path("out.txt")) == "txt"
    assert _resolve_format("txt", Path("out")) == "txt"
    assert _resolve_format("md", Path("out.unknown")) == "md"


def test_one_shot_json_without_words_omits_word_arrays(tmp_path, monkeypatch) -> None:
    _record_one_shot(monkeypatch)
    js = tmp_path / "call.json"
    out = tmp_path / "call.md"

    assert main(["some.mp3", "--json", str(js), "-o", str(out)]) == 0

    payload = json.loads(js.read_text())
    assert payload["segments"] == [{"start": 0.0, "end": 2.0, "text": "privet"}]


def test_watch_rejects_an_unknown_format(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["watch", "--format", "pdf"])

    assert excinfo.value.code == 2
    assert "Unsupported format" in capsys.readouterr().err
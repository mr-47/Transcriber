import pytest

from transcriber.config import (
    TranscriberSettings,
    _env_float,
    _env_int,
)


@pytest.mark.parametrize(
    "raw,expected",
    [("0.7", 0.7), ("", 0.7), ("2", 2.0), ("junk", 0.7)],
)
def test_env_float(monkeypatch, raw, expected):
    monkeypatch.setenv("T", raw)
    assert _env_float("T", 0.7) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("2", 2), ("", 1), ("junk", 1), ("-1", -1)],
)
def test_env_int(monkeypatch, raw, expected):
    monkeypatch.setenv("T", raw)
    assert _env_int("T", 1) == expected


def test_settings_defaults(monkeypatch):
    for key in (
        "TRANSCRIBER_WHISPER_MODEL",
        "TRANSCRIBER_WHISPER_DEVICE",
        "TRANSCRIBER_WHISPER_COMPUTE_TYPE",
        "TRANSCRIBER_DIARIZATION_MODEL",
        "TRANSCRIBER_HF_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = TranscriberSettings()
    assert settings.whisper_model == "small"
    assert settings.whisper_device == "auto"
    assert settings.whisper_compute_type == ""
    assert settings.diarization_model == "pyannote/speaker-diarization-3.1"
    assert settings.hf_token == ""
    assert settings.diarization_enabled is False
    assert settings.resolve_compute_type() == "default"


def test_empty_env_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_WHISPER_MODEL", "")
    monkeypatch.setenv("TRANSCRIBER_WHISPER_DEVICE", "")
    monkeypatch.setenv("TRANSCRIBER_DIARIZATION_MODEL", "")
    settings = TranscriberSettings()
    assert settings.whisper_model == "small"
    assert settings.whisper_device == "auto"
    assert settings.diarization_model == "pyannote/speaker-diarization-3.1"


def test_settings_read_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_WHISPER_MODEL", "base")
    monkeypatch.setenv("TRANSCRIBER_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("TRANSCRIBER_WHISPER_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("TRANSCRIBER_DIARIZATION_MODEL", "custom/model")
    monkeypatch.setenv("TRANSCRIBER_HF_TOKEN", "secret-token")
    settings = TranscriberSettings()
    assert settings.whisper_model == "base"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "float16"
    assert settings.diarization_model == "custom/model"
    assert settings.hf_token == "secret-token"
    assert settings.diarization_enabled is True
    assert settings.resolve_compute_type() == "float16"


def test_diarization_knob_defaults_are_off(monkeypatch):
    for key in (
        "TRANSCRIBER_DIAR_THRESHOLD",
        "TRANSCRIBER_DIAR_SPEAKERS",
        "TRANSCRIBER_DIAR_MIN_ON",
        "TRANSCRIBER_DIAR_MIN_OFF",
        "TRANSCRIBER_DIAR_MIN_REGION",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = TranscriberSettings()
    assert settings.diarization_threshold == -1.0
    assert settings.diarization_num_speakers == -1
    assert settings.diarization_min_duration_on == -1.0
    assert settings.diarization_min_duration_off == -1.0
    assert settings.diarization_min_region_seconds == 1.0


def test_diarization_knobs_read_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_DIAR_THRESHOLD", "0.6")
    monkeypatch.setenv("TRANSCRIBER_DIAR_SPEAKERS", "3")
    monkeypatch.setenv("TRANSCRIBER_DIAR_MIN_ON", "0.4")
    monkeypatch.setenv("TRANSCRIBER_DIAR_MIN_OFF", "0.9")
    monkeypatch.setenv("TRANSCRIBER_DIAR_MIN_REGION", "0")
    settings = TranscriberSettings()
    assert settings.diarization_threshold == 0.6
    assert settings.diarization_num_speakers == 3
    assert settings.diarization_min_duration_on == 0.4
    assert settings.diarization_min_duration_off == 0.9
    assert settings.diarization_min_region_seconds == 0.0


def test_resolve_whisper_model_passes_hf_names(monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_WHISPER_MODEL", "large-v3")
    assert TranscriberSettings().resolve_whisper_model() == "large-v3"


def test_resolve_whisper_model_passes_hf_org_model_names(monkeypatch):
    """`org/model` ids contain a slash but are model ids, not paths."""
    repo = "Systran/faster-whisper-large-v3"
    monkeypatch.setenv("TRANSCRIBER_WHISPER_MODEL", repo)
    assert TranscriberSettings().resolve_whisper_model() == repo


def test_resolve_whisper_model_accepts_existing_path(monkeypatch, tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    monkeypatch.setenv("TRANSCRIBER_WHISPER_MODEL", str(model))
    assert TranscriberSettings().resolve_whisper_model() == str(model)


def test_resolve_whisper_model_rejects_missing_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIBER_WHISPER_MODEL", str(tmp_path / "nope" / "model.bin"))
    with pytest.raises(FileNotFoundError) as exc:
        TranscriberSettings().resolve_whisper_model()
    assert "TRANSCRIBER_WHISPER_MODEL" in str(exc.value)
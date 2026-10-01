import contextlib
import io
import wave

from fastapi.testclient import TestClient

from transcriber.api import create_app
from transcriber.models import Segment, Transcript, Utterance, Word


class _FakeTranscriber:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.settings = _FakeSettings()

    def transcribe(self, audio_path: str, language=None, word_timestamps: bool = True) -> Transcript:
        self.calls.append((audio_path, language, word_timestamps))
        return Transcript(
            language="en",
            language_probability=0.99,
            duration=1.5,
            segments=[
                Segment(
                    start=0.0,
                    end=0.5,
                    text="hello",
                    words=[Word(start=0.0, end=0.2, word="hello")],
                )
            ],
            utterances=[Utterance(speaker="SPEAKER_00", start=0.0, end=0.5, text="hello")],
        )

    def close(self) -> None:
        self.closed = True


class _FakeSettings:
    whisper_model = "small"
    whisper_device = "cpu"
    whisper_compute_type = "int8"
    hf_token = "test-token"

    def resolve_compute_type(self) -> str:
        return self.whisper_compute_type

    def resolve_whisper_model(self) -> str:
        return self.whisper_model

    @property
    def diarization_enabled(self) -> bool:
        return bool(self.hf_token)


def _audio_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


@contextlib.contextmanager
def _client(fake: _FakeTranscriber):
    app = create_app(transcriber=fake, default_language="en")
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_status_and_configuration():
    fake = _FakeTranscriber()
    with _client(fake) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["diarization_enabled"] is True
    assert body["model"] == "small"
    assert body["model_path"] is None
    assert body["device"] == "cpu"
    assert body["compute_type"] == "int8"


def test_health_survives_an_unresolvable_model_path():
    fake = _FakeTranscriber()

    def boom():
        raise FileNotFoundError("model gone")

    fake.settings.resolve_whisper_model = boom
    with _client(fake) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_path"] is None


def test_health_reports_a_local_model_path():
    fake = _FakeTranscriber()
    local = "/srv/models/model.bin"
    fake.settings.whisper_model = local
    fake.settings.resolve_whisper_model = lambda: local
    with _client(fake) as client:
        body = client.get("/health").json()

    assert body["model_path"] == local


def test_transcribe_success():
    with _client(_FakeTranscriber()) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("call.wav", _audio_bytes(), "audio/wav")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "en"
    assert body["utterances"][0]["speaker"] == "SPEAKER_00"
    assert body["segments"][0]["words"][0]["word"] == "hello"


def test_transcribe_without_words_drops_them():
    fake = _FakeTranscriber()
    with _client(fake) as client:
        response = client.post(
            "/transcribe?words=false",
            files={"file": ("call.wav", _audio_bytes(), "audio/wav")},
        )
    assert response.status_code == 200
    assert response.json()["segments"][0]["words"] == []
    assert fake.calls[0][2] is False, "words=false must not ask the model for word timestamps"


def test_transcribe_requests_word_timestamps_by_default():
    fake = _FakeTranscriber()
    with _client(fake) as client:
        client.post("/transcribe", files={"file": ("call.wav", _audio_bytes(), "audio/wav")})
    assert fake.calls[0][2] is True


def test_transcribe_rejects_unknown_suffix():
    with _client(_FakeTranscriber()) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("call.pdf", b"%PDF", "application/pdf")},
        )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_transcribe_rejects_empty_upload():
    with _client(_FakeTranscriber()) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("call.wav", b"", "audio/wav")},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


def test_transcribe_uses_default_language():
    fake = _FakeTranscriber()
    with _client(fake) as client:
        client.post("/transcribe", files={"file": ("call.wav", _audio_bytes(), "audio/wav")})
    assert fake.calls[0][1] == "en"


def test_transcribe_passes_none_when_no_default_language_is_configured():
    fake = _FakeTranscriber()
    app = create_app(transcriber=fake)
    with TestClient(app) as client:
        client.post("/transcribe", files={"file": ("call.wav", _audio_bytes(), "audio/wav")})
    assert fake.calls[0][1] is None


def test_transcribe_rejects_a_malformed_words_param():
    with _client(_FakeTranscriber()) as client:
        response = client.post(
            "/transcribe?words=banana",
            files={"file": ("call.wav", _audio_bytes(), "audio/wav")},
        )
    assert response.status_code == 422


def test_transcribe_passes_engine_http_exceptions_through():
    from fastapi import HTTPException

    fake = _FakeTranscriber()

    def teapot(*args, **kwargs):
        raise HTTPException(status_code=418, detail="engine is a teapot")

    fake.transcribe = teapot
    with _client(fake) as client:
        response = client.post(
            "/transcribe", files={"file": ("call.wav", _audio_bytes(), "audio/wav")}
        )

    assert response.status_code == 418
    assert response.json()["detail"] == "engine is a teapot"


def test_transcribe_query_language_overrides_default():
    fake = _FakeTranscriber()
    with _client(fake) as client:
        client.post(
            "/transcribe?language=de",
            files={"file": ("call.wav", _audio_bytes(), "audio/wav")},
        )
    assert fake.calls[0][1] == "de"


def test_transcribe_missing_file_is_422():
    with _client(_FakeTranscriber()) as client:
        response = client.post("/transcribe")
    assert response.status_code == 422


def test_transcribe_accepts_upper_case_extension():
    fake = _FakeTranscriber()
    with _client(fake) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("call.WAV", _audio_bytes(), "audio/wav")},
        )
    assert response.status_code == 200
    assert fake.calls[0][1] == "en"


def test_transcribe_accepts_m4a():
    fake = _FakeTranscriber()
    with _client(fake) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("call.m4a", _audio_bytes(), "audio/mp4")},
        )
    assert response.status_code == 200


def test_transcribe_filename_without_extension_is_a_400():
    with _client(_FakeTranscriber()) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("call", _audio_bytes(), "audio/wav")},
        )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]



def test_transcribe_failure_leaves_no_temp_files(tmp_path, monkeypatch):
    import tempfile

    fake = _FakeTranscriber()

    def boom(*args, **kwargs):
        raise RuntimeError("transcribe exploded")

    fake.transcribe = boom
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with _client(fake) as client:
        client.post(
            "/transcribe", files={"file": ("call.wav", _audio_bytes(), "audio/wav")}
        )
    assert list(tmp_path.iterdir()) == [], f"temp files leaked: {list(tmp_path.iterdir())}"


def test_injected_transcriber_is_not_closed():
    fake = _FakeTranscriber()
    app = create_app(transcriber=fake, default_language="en")
    with TestClient(app) as client:
        client.get("/health")
    assert fake.closed is False


def test_owned_transcriber_is_closed_on_shutdown(monkeypatch):
    import transcriber.api as api_module
    from transcriber.config import TranscriberSettings

    observed = {"closed": False}

    class SpyTranscriber(api_module.Transcriber):
        def close(self) -> None:
            observed["closed"] = True
            super().close()

    monkeypatch.setattr(api_module, "Transcriber", SpyTranscriber)

    app = create_app(settings=TranscriberSettings())
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model"] == "small"

    assert observed["closed"] is True, "the lifespan must close a transcriber it owns"
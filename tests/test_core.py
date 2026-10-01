import sys
import types

import pytest

from transcriber import core
from transcriber.config import TranscriberSettings
from transcriber.core import (
    CUBLAS_SONAME,
    Transcriber,
    _preload_cublas,
    _torch_cuda_usable,
)


class _FakeSpec:
    def __init__(self, *roots):
        self.submodule_search_locations = list(roots)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    _torch_cuda_usable.cache_clear()
    _preload_cublas.cache_clear()
    yield
    _torch_cuda_usable.cache_clear()
    _preload_cublas.cache_clear()


def _no_kernel_image(monkeypatch):
    """Make CUDA look available while every kernel launch fails.

    Reproduces a torch build with no kernels for the installed GPU, such as
    Pascal on a CUDA 13 build, where the driver and device are fine.
    """
    torch = pytest.importorskip("torch")

    real_zeros = torch.zeros

    def fake_zeros(*args, **kwargs):
        if any(isinstance(value, str) and value.startswith("cuda") for value in kwargs.values()):
            raise torch.AcceleratorError(
                "CUDA error: no kernel image is available for execution on the device"
            )
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", fake_zeros)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_probe_accepts_working_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    assert _torch_cuda_usable() is True


def test_probe_rejects_missing_kernels(monkeypatch):
    _no_kernel_image(monkeypatch)

    assert _torch_cuda_usable() is False


def test_probe_rejects_absent_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _torch_cuda_usable() is False


def test_probe_result_is_cached(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def fake_zeros(*args, **kwargs):
        calls.append(kwargs.get("device"))
        return torch.tensor(0.0)

    monkeypatch.setattr(torch, "zeros", fake_zeros)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    _torch_cuda_usable()
    _torch_cuda_usable()

    assert len(calls) == 1


def _transcriber_with_fake_pyannote(monkeypatch, recorder):
    class _Pipeline:
        @classmethod
        def from_pretrained(cls, name, token=None):
            return cls()

        def to(self, device):
            recorder.append(device)
            return self

        def __call__(self, path):
            return iter(())

    _install_pyannote(monkeypatch, _Pipeline)

    settings = TranscriberSettings()
    settings.hf_token = "hf_test"
    return Transcriber(settings)


def _install_pyannote(monkeypatch, pipeline_cls):
    """Point the lazy ``pyannote.audio`` import at a stub pipeline class."""
    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = pipeline_cls
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)


def _fake_segment(start, end, text, words=None):
    return types.SimpleNamespace(start=start, end=end, text=text, words=words)


def _fake_info(language="en", probability=0.9, duration=3.0):
    return types.SimpleNamespace(
        language=language, language_probability=probability, duration=duration
    )


class _StubWhisper:
    """Drop-in for faster-whisper's model, recording how it was called."""

    def __init__(self, segments=(), info=None, first_error=None, record=None):
        self.segments = segments
        self.info = info or _fake_info()
        self.first_error = first_error
        self.record = record if record is not None else {}
        self.calls = 0

    def transcribe(self, path, language=None, word_timestamps=False):
        self.calls += 1
        self.record["language"] = language
        self.record["word_timestamps"] = word_timestamps
        if self.first_error is not None and self.calls == 1:
            raise self.first_error
        return iter(self.segments), self.info


def test_diarization_uses_gpu_when_usable(monkeypatch):
    pytest.importorskip("torch")
    if not _torch_cuda_usable():
        pytest.skip("torch CUDA unusable on this machine")

    moved = []
    transcriber = _transcriber_with_fake_pyannote(monkeypatch, moved)
    transcriber._get_diarization_pipeline()

    assert [str(device) for device in moved] == ["cuda"]


def test_diarization_stays_on_cpu_without_kernels(monkeypatch):
    _no_kernel_image(monkeypatch)

    moved = []
    transcriber = _transcriber_with_fake_pyannote(monkeypatch, moved)
    pipeline = transcriber._get_diarization_pipeline()

    assert pipeline is not None
    assert moved == []


def _fake_loader(monkeypatch, resolvable, shipped):
    """Replace ctypes.CDLL with a loader that tracks what was requested.

    ``resolvable`` is the set of sonames the loader already knows, standing in
    for a host CUDA 12 install. ``shipped`` maps an nvidia package root to the
    libraries it contains.
    """
    loaded = []
    known = set(resolvable)

    def fake_cdll(name, *args, **kwargs):
        if name in known:
            loaded.append(name)
            return object()
        if any(name == lib for libs in shipped.values() for lib in libs):
            loaded.append(name)
            return object()
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(core.ctypes, "CDLL", fake_cdll)
    return loaded


def test_preload_skips_when_already_on_loader_path(monkeypatch):
    loaded = _fake_loader(monkeypatch, resolvable={CUBLAS_SONAME}, shipped={})

    assert _preload_cublas() is True
    assert loaded == [CUBLAS_SONAME]


def test_preload_loads_shipped_cublas(monkeypatch, tmp_path):
    lib = tmp_path / "cublas" / "lib" / CUBLAS_SONAME
    lib.parent.mkdir(parents=True)
    lib.touch()
    loaded = _fake_loader(monkeypatch, resolvable=set(), shipped={str(tmp_path): {str(lib)}})

    monkeypatch.setattr(core, "find_spec", lambda name: _FakeSpec(str(tmp_path)))

    assert _preload_cublas() is True
    assert loaded == [str(lib)]


def test_preload_reports_missing_cublas(monkeypatch, tmp_path):
    _fake_loader(monkeypatch, resolvable=set(), shipped={})
    monkeypatch.setattr(core, "find_spec", lambda name: _FakeSpec(str(tmp_path)))

    assert _preload_cublas() is False


def test_preload_warns_when_cublas_cannot_be_loaded(monkeypatch, tmp_path, caplog):
    import logging

    lib = tmp_path / "cublas" / "lib" / CUBLAS_SONAME
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"not a real shared object")

    def exploding_cdll(name, *args, **kwargs):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(core.ctypes, "CDLL", exploding_cdll)
    monkeypatch.setattr(core, "find_spec", lambda name: _FakeSpec(str(tmp_path)))

    with caplog.at_level(logging.WARNING, logger="transcriber.core"):
        assert _preload_cublas() is False

    assert "Could not preload" in caplog.text


def test_preload_runs_once(monkeypatch, tmp_path):
    lib = tmp_path / "cublas" / "lib" / CUBLAS_SONAME
    lib.parent.mkdir(parents=True)
    lib.touch()
    loaded = _fake_loader(monkeypatch, resolvable=set(), shipped={str(tmp_path): {str(lib)}})
    monkeypatch.setattr(core, "find_spec", lambda name: _FakeSpec(str(tmp_path)))

    Transcriber(TranscriberSettings())
    Transcriber(TranscriberSettings())

    assert loaded == [str(lib)]


def test_transcriber_preloads_cublas(monkeypatch, tmp_path):
    lib = tmp_path / "cublas" / "lib" / CUBLAS_SONAME
    lib.parent.mkdir(parents=True)
    lib.touch()
    loaded = _fake_loader(monkeypatch, resolvable=set(), shipped={str(tmp_path): {str(lib)}})
    monkeypatch.setattr(core, "find_spec", lambda name: _FakeSpec(str(tmp_path)))

    Transcriber(TranscriberSettings())

    assert loaded == [str(lib)]


def test_transcriber_drops_models_on_close():
    transcriber = Transcriber(TranscriberSettings())
    transcriber._whisper_model = object()
    transcriber._diarization_pipeline = object()

    transcriber.close()

    assert transcriber._whisper_model is None
    assert transcriber._diarization_pipeline is None


def test_transcriber_is_context_manager():
    transcriber = Transcriber(TranscriberSettings())
    transcriber._whisper_model = object()
    with transcriber as managed:
        assert managed is transcriber

    assert transcriber._whisper_model is None
    assert transcriber._diarization_pipeline is None


def _ready_transcriber(settings=None):
    """A Transcriber whose models are pre-stubbed, so no heavy imports run."""
    transcriber = Transcriber(settings or TranscriberSettings())
    return transcriber


def test_transcribe_without_diarization_builds_transcript():
    settings = TranscriberSettings()
    settings.hf_token = ""
    transcriber = _ready_transcriber(settings)
    whisper = _StubWhisper(
        segments=[
            _fake_segment(0.0, 1.0, " Hello ", [types.SimpleNamespace(start=0.0, end=0.5, word="Hello")]),
            _fake_segment(1.0, 2.0, "   ", []),
            _fake_segment(2.0, 3.0, "world.", None),
        ]
    )
    transcriber._whisper_model = whisper

    result = transcriber.transcribe("/tmp/a.wav", language="de")

    assert whisper.record["language"] == "de"
    assert whisper.record["word_timestamps"] is True
    assert result.language == "en"
    assert result.duration == 3.0
    assert len(result.segments) == 2, "blank-text segments must be dropped"
    assert result.segments[0].words[0].word == "Hello"
    assert result.segments[1].words == []
    assert result.utterances[0].text == "Hello world."
    assert result.utterances[0].speaker == "SPEAKER_00"


def test_transcribe_without_diarization_defaults_to_speaker_00():
    settings = TranscriberSettings()
    settings.hf_token = ""
    transcriber = _ready_transcriber(settings)
    transcriber._whisper_model = _StubWhisper(
        segments=[_fake_segment(0.0, 1.0, "a"), _fake_segment(1.0, 2.0, "b")]
    )

    result = transcriber.transcribe("/tmp/a.wav")

    assert [u.speaker for u in result.utterances] == ["SPEAKER_00"]
    assert result.utterances[0].text == "a b"


def test_transcribe_can_skip_word_timestamps():
    settings = TranscriberSettings()
    settings.hf_token = ""
    transcriber = _ready_transcriber(settings)
    whisper = _StubWhisper(segments=[_fake_segment(0.0, 1.0, "a")])
    transcriber._whisper_model = whisper

    transcriber.transcribe("/tmp/a.wav", word_timestamps=False)

    assert whisper.record["word_timestamps"] is False
    assert whisper.record["language"] is None


def test_whisper_model_is_built_once_and_cached(monkeypatch):
    transcriber = _ready_transcriber()
    built = []
    model = object()

    def build(self, model_name, device, compute_type):  # noqa: ANN001
        built.append((model_name, device, compute_type))
        return model

    monkeypatch.setattr(Transcriber, "_build_whisper_model", build)

    assert transcriber._get_whisper_model() is model
    assert transcriber._get_whisper_model() is model
    assert built == [("small", "auto", "default")]


def test_non_int8_compute_type_falls_back_to_int8(monkeypatch):
    settings = TranscriberSettings()
    settings.whisper_compute_type = "float16"
    transcriber = _ready_transcriber(settings)
    calls = []
    model = object()

    def build(self, model_name, device, compute_type):  # noqa: ANN001
        calls.append(compute_type)
        if compute_type != "int8":
            raise ValueError("float16 is not supported on this device")
        return model

    monkeypatch.setattr(Transcriber, "_build_whisper_model", build)

    assert transcriber._get_whisper_model() is model
    assert calls == ["float16", "int8"]


def test_int8_compute_type_error_is_not_swallowed(monkeypatch):
    settings = TranscriberSettings()
    settings.whisper_compute_type = "int8"
    transcriber = _ready_transcriber(settings)

    def build(self, *args):  # noqa: ANN001
        raise ValueError("int8 also failed")

    monkeypatch.setattr(Transcriber, "_build_whisper_model", build)
    with pytest.raises(ValueError):
        transcriber._get_whisper_model()


def test_transcribe_reloads_whisper_on_cuda_error(monkeypatch, caplog):
    usage = {}
    transcriber = _ready_transcriber()
    transcriber._whisper_model = _StubWhisper(
        segments=[_fake_segment(0.0, 1.0, "recovered")],
        first_error=RuntimeError("CUDA error: an illegal instruction was executed"),
    )

    def reload(self):  # noqa: ANN001 - bound method replacement
        usage["reloaded"] = True
        return _StubWhisper(segments=[_fake_segment(0.0, 1.0, "recovered")])

    monkeypatch.setattr(Transcriber, "_reload_whisper_cpu", reload)

    result = transcriber.transcribe("a.wav")

    assert usage["reloaded"] is True
    assert result.segments[0].text == "recovered"


def test_transcribe_reloads_whisper_on_lazy_cuda_error(monkeypatch):
    """A CUDA error on first consumption must also trigger the CPU reload.

    faster-whisper's ``transcribe`` returns a lazy generator and does its first
    GPU encode on the first ``next()``, so failures can surface while the
    segments are being consumed rather than when the generator is created.
    """
    usage = {}
    transcriber = _ready_transcriber()

    class _LazyCudaWhisper:
        def transcribe(self, path, language=None, word_timestamps=False):
            def segments():
                raise RuntimeError("CUDA error: an illegal instruction was executed")
                yield None  # pragma: no cover

            return segments(), _fake_info()

    transcriber._whisper_model = _LazyCudaWhisper()

    def reload(self):  # noqa: ANN001 - bound method replacement
        usage["reloaded"] = True
        return _StubWhisper(segments=[_fake_segment(0.0, 1.0, "recovered")])

    monkeypatch.setattr(Transcriber, "_reload_whisper_cpu", reload)

    result = transcriber.transcribe("a.wav")

    assert usage["reloaded"] is True
    assert result.segments[0].text == "recovered"


def test_transcribe_reraise_non_cuda_error():
    transcriber = _ready_transcriber()
    transcriber._whisper_model = _StubWhisper(first_error=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        transcriber.transcribe("a.wav")


def test_transcribe_cuda_error_on_cpu_device_is_not_recovered():
    settings = TranscriberSettings()
    settings.whisper_device = "cpu"
    transcriber = _ready_transcriber(settings)
    transcriber._whisper_model = _StubWhisper(
        first_error=RuntimeError("CUDA error: device-side assert")
    )

    with pytest.raises(RuntimeError):
        transcriber.transcribe("a.wav")


def test_transcribe_diarizes_with_turn_attribution(monkeypatch):
    turns = []
    _install_pyannote(monkeypatch, _turn_pipeline_recorder(turns))
    settings = TranscriberSettings()
    settings.hf_token = "hf_test"
    transcriber = _ready_transcriber(settings)
    transcriber._whisper_model = _StubWhisper(
        segments=[_fake_segment(0.0, 2.0, "hello")]
    )

    result = transcriber.transcribe("/tmp/a.wav")

    assert result.utterances[0].text == "hello"
    assert result.utterances[0].speaker == "SPEAKER_00"
    assert result.utterances[0].start == 0.0
    assert turns == ["/tmp/a.wav"]


def test_num_speakers_is_forwarded_to_pipeline(monkeypatch):
    seen = {}
    _install_pyannote(monkeypatch, _empty_turn_pipeline(seen))
    settings = TranscriberSettings()
    settings.hf_token = "hf_test"
    settings.diarization_num_speakers = 3
    transcriber = _ready_transcriber(settings)
    transcriber._whisper_model = _StubWhisper()

    transcriber.transcribe("/tmp/a.wav")

    assert seen == {"num_speakers": 3}


def test_num_speakers_is_omitted_when_not_set(monkeypatch):
    seen = {}
    _install_pyannote(monkeypatch, _empty_turn_pipeline(seen))
    settings = TranscriberSettings()
    settings.hf_token = "hf_test"
    transcriber = _ready_transcriber(settings)
    transcriber._whisper_model = _StubWhisper()

    transcriber.transcribe("/tmp/a.wav")

    assert seen == {}


def test_build_whisper_model_forwards_args(monkeypatch):
    seen = {}
    fw = types.ModuleType("faster_whisper")

    class _Model:
        def __init__(self, model, device=None, compute_type=None):
            seen.update(model=model, device=device, compute_type=compute_type)

    fw.WhisperModel = _Model
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    transcriber = _ready_transcriber()

    built = transcriber._build_whisper_model("base", "cpu", "int8")

    assert seen == {"model": "base", "device": "cpu", "compute_type": "int8"}
    assert built is not None


def test_reload_whisper_cpu_builds_int8_on_cpu(monkeypatch):
    seen = {}
    model = object()

    def build(self, model_name, device, compute_type):  # noqa: ANN001
        seen.update(model=model_name, device=device, compute_type=compute_type)
        return model

    monkeypatch.setattr(Transcriber, "_build_whisper_model", build)
    transcriber = _ready_transcriber()

    assert transcriber._reload_whisper_cpu() is model
    assert seen == {"model": "small", "device": "cpu", "compute_type": "int8"}
    assert transcriber._whisper_model is model


def test_diarization_pipeline_gpu_move_failure_is_non_fatal(monkeypatch, caplog):
    import logging

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, name, token=None):
            return cls()

        def to(self, device):
            raise RuntimeError("boom")

        def __call__(self, path, **kwargs):
            return iter(())

    _install_pyannote(monkeypatch, _Pipeline)
    monkeypatch.setattr(core, "_torch_cuda_usable", lambda: True)
    settings = TranscriberSettings()
    settings.hf_token = "hf_test"
    transcriber = _ready_transcriber(settings)

    with caplog.at_level(logging.WARNING, logger="transcriber.core"):
        pipeline = transcriber._get_diarization_pipeline()

    assert pipeline is not None
    assert "Could not move diarization pipeline to GPU" in caplog.text


def test_diarization_pipeline_is_built_once(monkeypatch):
    created = []

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, name, token=None):
            created.append(name)
            return cls()

        def to(self, device):
            return self

        def __call__(self, path, **kwargs):
            return iter(())

    _install_pyannote(monkeypatch, _Pipeline)
    settings = TranscriberSettings()
    settings.hf_token = "hf_test"
    transcriber = _ready_transcriber(settings)

    first = transcriber._get_diarization_pipeline()
    second = transcriber._get_diarization_pipeline()

    assert first is second
    assert created == [settings.diarization_model]


def test_diarization_returns_none_when_disabled():
    settings = TranscriberSettings()
    settings.hf_token = ""
    transcriber = _ready_transcriber(settings)

    assert transcriber._get_diarization_pipeline() is None
    assert transcriber.settings.diarization_enabled is False


def _empty_turn_pipeline(seen):
    """A pyannote stub that records the kwargs and yields no turns."""

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, name, token=None):
            return cls()

        def to(self, device):
            return self

        def __call__(self, path, **kwargs):
            seen.update(kwargs)
            return iter(())

    return _Pipeline


def _turn_pipeline_recorder(turns):
    class _Pipeline:
        @classmethod
        def from_pretrained(cls, name, token=None):
            return cls()

        def to(self, device):
            return self

        def __call__(self, path, **kwargs):
            turns.append(path)
            return iter(
                (
                    (
                        types.SimpleNamespace(start=0.0, end=2.0),
                        None,
                        "SPEAKER_00",
                    ),
                )
            )

    return _Pipeline


def test_tune_pipeline_applies_knobs():
    settings = TranscriberSettings()
    settings.diarization_threshold = 0.6
    settings.diarization_min_duration_on = 0.4
    settings.diarization_min_duration_off = 0.9
    pipeline = types.SimpleNamespace(
        hyperparameters={
            "clustering": {"threshold": 0.5},
            "segmentation": {"min_duration_on": 0.3, "min_duration_off": 0.7},
        }
    )

    Transcriber._tune_pipeline(pipeline, settings)

    assert pipeline.hyperparameters["clustering"]["threshold"] == 0.6
    assert pipeline.hyperparameters["segmentation"]["min_duration_on"] == 0.4
    assert pipeline.hyperparameters["segmentation"]["min_duration_off"] == 0.9


def test_tune_pipeline_leaves_defaults_untouched(caplog):
    settings = TranscriberSettings()  # all knobs at -1
    pipeline = types.SimpleNamespace(hyperparameters={"clustering": {}, "segmentation": {}})

    Transcriber._tune_pipeline(pipeline, settings)

    assert pipeline.hyperparameters == {"clustering": {}, "segmentation": {}}
    assert caplog.text == ""


def test_tune_pipeline_warns_when_section_missing(caplog):
    import logging

    settings = TranscriberSettings()
    settings.diarization_threshold = 0.6
    pipeline = types.SimpleNamespace(hyperparameters={})

    with caplog.at_level(logging.WARNING, logger="transcriber.core"):
        Transcriber._tune_pipeline(pipeline, settings)

    assert "Could not tune diarization" in caplog.text


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CUDA error: device-side assert triggered", True),
        ("ctranslate2: cublas failed to initialize", True),
        ("cudnn error: CUDNN_STATUS_MAPPING_ERROR", True),
        ("cudart error: out of memory", True),
        ("no NVIDIA driver found for this device", True),
        ("some unrelated failure", False),
        ("", False),
    ],
)
def test_is_cuda_error(message, expected):
    assert core._is_cuda_error(RuntimeError(message)) is expected

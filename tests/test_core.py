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

    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = _Pipeline
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)

    settings = TranscriberSettings()
    settings.hf_token = "hf_test"
    return Transcriber(settings)


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

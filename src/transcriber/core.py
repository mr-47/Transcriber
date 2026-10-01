from __future__ import annotations

import ctypes
import logging
import threading
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

from .config import TranscriberSettings
from .merge import absorb_tiny_turns, assign_speakers, build_utterances
from .models import Segment, SpeakerTurn, Transcript, Word

logger = logging.getLogger(__name__)

CUBLAS_SONAME = "libcublas.so.12"


@lru_cache(maxsize=1)
def _preload_cublas() -> bool:
    """Make ``libcublas.so.12`` resolvable for CTranslate2's bare ``dlopen``.

    CTranslate2 was built against CUDA 12 and loads cuBLAS by soname with no
    search path of its own, so it fails unless the loader already knows that
    name. The copy shipped in this environment lives under
    ``nvidia/cublas/lib``, which is not a default loader directory.

    Setting ``LD_LIBRARY_PATH`` fixes it only until the next reboot, and
    assigning ``os.environ`` at runtime does not work at all because glibc
    resolves its search path when the process starts. Loading the library here
    by absolute path is durable instead: a later ``dlopen`` of the same soname
    matches the already-loaded object and succeeds.

    Returns True when cuBLAS is available, whether it was already on the
    loader path or preloaded from here.
    """
    try:
        ctypes.CDLL(CUBLAS_SONAME)
        return True
    except OSError:
        pass

    spec = find_spec("nvidia")
    roots = list(spec.submodule_search_locations or ()) if spec else []
    for root in roots:
        candidate = Path(root) / "cublas" / "lib" / CUBLAS_SONAME
        if not candidate.is_file():
            continue
        try:
            ctypes.CDLL(str(candidate))
        except OSError as exc:
            logger.warning("Could not preload %s: %s", candidate, exc)
            continue
        logger.info("Preloaded %s so CTranslate2 can reach cuBLAS", candidate)
        return True

    logger.warning(
        "%s is neither on the loader path nor shipped alongside this environment; "
        "CUDA whisper inference will fall back to CPU",
        CUBLAS_SONAME,
    )
    return False


def _is_cuda_error(exc: BaseException) -> bool:
    """Best-effort check whether an exception stems from a broken CUDA backend."""
    message = str(exc).lower()
    return any(token in message for token in ("cuda", "cublas", "cudnn", "cudart", "driver"))


@lru_cache(maxsize=1)
def _torch_cuda_usable() -> bool:
    """Report whether torch can actually launch CUDA kernels on this GPU.

    ``torch.cuda.is_available()`` only reports driver-level presence, so it
    returns True even when the installed build ships no kernels for the device's
    compute capability (Pascal on a CUDA 13 build, for instance). Moving a module
    with ``.to("cuda")`` is a plain copy that succeeds regardless, so such a
    mismatch stays invisible until the first real kernel launch. Probe with an
    actual matmul and force a sync so the launch error surfaces here instead.

    The result is cached: the answer cannot change within a process, and the
    probe should not run once per request.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        probe = torch.zeros(8, 8, device="cuda")
        float((probe @ probe).sum().item())
        del probe
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch CUDA is unusable, keeping torch work on CPU: %s", exc)
        return False


class Transcriber:
    """Transcribes audio files and attributes speech to speakers.

    Models are loaded lazily on first use and cached; a single lock serializes
    inference since whisper/pyannote model binaries are not safe for concurrent
    use within one process.
    """

    def __init__(self, settings: TranscriberSettings | None = None) -> None:
        _preload_cublas()
        self._settings = settings or TranscriberSettings()
        self._whisper_model = None
        self._diarization_pipeline = None
        self._lock = threading.Lock()

    @property
    def settings(self) -> TranscriberSettings:
        return self._settings

    def _build_whisper_model(self, model: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        return WhisperModel(model, device=device, compute_type=compute_type)

    def _get_whisper_model(self):
        if self._whisper_model is None:
            device = self._settings.whisper_device
            compute_type = self._settings.resolve_compute_type()
            model = self._settings.resolve_whisper_model()
            logger.info(
                "Loading whisper model %s (device=%s, compute_type=%s)",
                model,
                device,
                compute_type,
            )
            try:
                self._whisper_model = self._build_whisper_model(model, device, compute_type)
            except ValueError as exc:
                if compute_type == "int8":
                    raise
                logger.warning(
                    "compute_type %r unavailable on this device (%s); retrying with int8",
                    compute_type,
                    exc,
                )
                self._whisper_model = self._build_whisper_model(model, device, "int8")
        return self._whisper_model

    def _reload_whisper_cpu(self):
        logger.warning("CUDA inference unavailable; reloading whisper model on CPU")
        self._whisper_model = self._build_whisper_model(
            self._settings.resolve_whisper_model(), "cpu", "int8"
        )
        return self._whisper_model

    @staticmethod
    def _tune_pipeline(pipeline, settings: TranscriberSettings) -> None:
        """Apply the optional diarization tuning knobs to a pyannote pipeline.

        A knob at its default (-1) leaves the pretrained hyperparameter alone.
        pyannote stores its configuration in ``hyperparameters``, so tuning is
        a dict edit, not a re-instantiation; values that cannot reach that
        structure (unknown versions) are skipped with a warning rather than
        failing the whole transcription.
        """
        items = [
            ("clustering", "threshold", settings.diarization_threshold),
            ("segmentation", "min_duration_on", settings.diarization_min_duration_on),
            ("segmentation", "min_duration_off", settings.diarization_min_duration_off),
        ]
        tuned = []
        hyperparameters = getattr(pipeline, "hyperparameters", {})
        for section, key, value in items:
            if value is None or value < 0:
                continue
            try:
                hyperparameters[section][key] = value
                tuned.append(f"{section}.{key}={value}")
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Could not tune diarization %s.%s=%s: %s", section, key, value, exc
                )
        if tuned:
            logger.info("Tuning diarization: %s", ", ".join(tuned))

    def _get_diarization_pipeline(self):
        if not self._settings.diarization_enabled:
            return None
        if self._diarization_pipeline is None:
            from pyannote.audio import Pipeline

            logger.info("Loading diarization pipeline %s", self._settings.diarization_model)
            pipeline = Pipeline.from_pretrained(
                self._settings.diarization_model, token=self._settings.hf_token
            )
            self._tune_pipeline(pipeline, self._settings)
            try:
                import torch

                if _torch_cuda_usable():
                    pipeline = pipeline.to(torch.device("cuda"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not move diarization pipeline to GPU: %s", exc)
            self._diarization_pipeline = pipeline
        return self._diarization_pipeline

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> Transcript:
        with self._lock:
            whisper = self._get_whisper_model()

            def run_whisper(model):
                # faster-whisper's ``transcribe`` returns a lazy generator: feature
                # extraction and the first GPU encode happen on the first ``next()``,
                # so a CUDA failure can surface while the segments are consumed, not
                # only when the generator is created. Consume here, inside the guard,
                # so a reload picks the failure up either way.
                segments_iter, info = model.transcribe(
                    audio_path,
                    language=language,
                    word_timestamps=word_timestamps,
                )
                segments = [
                    Segment(
                        start=s.start,
                        end=s.end,
                        text=s.text.strip(),
                        words=[Word(w.start, w.end, w.word) for w in (s.words or [])],
                    )
                    for s in segments_iter
                    if s.text and s.text.strip()
                ]
                return segments, info

            try:
                segments, info = run_whisper(whisper)
            except RuntimeError as exc:
                if self._settings.whisper_device == "cpu" or not _is_cuda_error(exc):
                    raise
                whisper = self._reload_whisper_cpu()
                segments, info = run_whisper(whisper)

            turns: list[SpeakerTurn] = []
            pipeline = self._get_diarization_pipeline()
            if pipeline is not None:
                kwargs = {}
                if self._settings.diarization_num_speakers > 0:
                    kwargs["num_speakers"] = self._settings.diarization_num_speakers
                for turn, _, label in pipeline(audio_path, **kwargs):
                    turns.append(
                        SpeakerTurn(start=turn.start, end=turn.end, speaker=str(label))
                    )
                turns = absorb_tiny_turns(
                    turns, self._settings.diarization_min_region_seconds
                )

            speakers = assign_speakers(segments, turns)
            utterances = build_utterances(segments, speakers)

            return Transcript(
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
                segments=segments,
                utterances=utterances,
            )

    def close(self) -> None:
        """Release loaded models so their GPU/CPU memory can be freed.

        Drops the references faster-whisper and pyannote hold on their
        allocation caches; a process would anyway -- this matters for a
        long-running API worker that wants to hand VRAM back before shutdown.
        """
        self._whisper_model = None
        self._diarization_pipeline = None

    def __enter__(self) -> "Transcriber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
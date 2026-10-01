"""REST API, a thin HTTP skin over the same :class:`Transcriber` the CLI uses.

Design notes:

* The transcriber is created once per app (lifespan) and reused, so models load
  on the first request and stay warm afterwards. Creating it per request would
  pay the model load every time.
* Transcription is CPU/GPU-bound and synchronous, so it runs in a worker thread
  (``anyio.to_thread.run_sync``) instead of blocking the event loop.
* ``Transcriber`` serialises calls internally with a lock, because whisper and
  pyannote model binaries are not safe for concurrent use within one process.
  Concurrent uploads queue up rather than fighting over the GPU.
* Uploads are streamed to a temp file, because faster-whisper needs a seekable
  path. The temp file is always removed, including on failure.
* There is **no authentication**. Bind to loopback, or put this behind a reverse
  proxy, before exposing it to a network.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .config import SUPPORTED_AUDIO_SUFFIXES, TranscriberSettings
from .core import Transcriber
from .models import Transcript

logger = logging.getLogger(__name__)

UPLOAD_CHUNK = 1024 * 1024


class HealthResponse(BaseModel):
    status: str
    diarization_enabled: bool
    model: str
    model_path: str | None = None
    device: str
    compute_type: str


class WordOut(BaseModel):
    start: float
    end: float
    word: str


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordOut] = Field(default_factory=list)


class UtteranceOut(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptOut(BaseModel):
    """Same shape as the JSON written by ``--json`` and ``calls-results/``."""

    language: str | None
    language_probability: float | None
    duration: float
    segments: list[SegmentOut]
    utterances: list[UtteranceOut]
    text: str


def _to_out(result: Transcript, words: bool) -> TranscriptOut:
    payload = result.to_dict(include_words=words)
    return TranscriptOut(**payload)


def create_app(
    transcriber: Transcriber | None = None,
    settings: TranscriberSettings | None = None,
    default_language: str | None = None,
) -> FastAPI:
    """Build the app. Pass ``transcriber`` to inject a stub (tests); pass
    ``settings`` to configure the real one, and ``default_language`` to apply a
    language hint to requests that omit ``?language=``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = transcriber is None
        app.state.transcriber = transcriber or Transcriber(settings or TranscriberSettings())
        app.state.owns_transcriber = owned
        try:
            yield
        finally:
            # Release loaded models so a reload does not leak GPU/CPU memory.
            if owned:
                app.state.transcriber.close()

    app = FastAPI(
        title="Transcriber",
        version=__version__,
        description="Audio transcription with speaker diarization (faster-whisper + pyannote)",
        lifespan=lifespan,
    )

    def get_transcriber() -> Transcriber:
        try:
            return app.state.transcriber
        except AttributeError:  # pragma: no cover - only if lifespan is bypassed
            raise HTTPException(status_code=503, detail="Service is still starting up")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        engine = get_transcriber()
        model_path: str | None = None
        try:
            resolved = engine.settings.resolve_whisper_model()
            # Only an absolute path is a local model; `org/model` ids also
            # contain a separator but are hub references, not paths.
            if Path(resolved).is_absolute():
                model_path = resolved
        except FileNotFoundError:
            model_path = None
        return HealthResponse(
            status="ok",
            diarization_enabled=engine.settings.diarization_enabled,
            model=engine.settings.whisper_model,
            model_path=model_path,
            device=engine.settings.whisper_device,
            compute_type=engine.settings.resolve_compute_type(),
        )

    @app.post("/transcribe", response_model=TranscriptOut, tags=["transcription"])
    async def handle_transcribe(
        request: Request,
        file: UploadFile = File(..., description="Audio file to transcribe"),  # noqa: B008
        language: str | None = Query(
            default=None,
            description=(
                "Optional language hint for whisper, e.g. 'en', 'de', 'ru'. Omit to use "
                f"the app default ({default_language or 'auto-detect'})."
            ),
        ),
        words: bool = Query(default=True, description="Include per-word timestamps"),
    ) -> TranscriptOut:
        engine = get_transcriber()

        suffix = os.path.splitext(file.filename or "")[1].lower()
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix or 'unknown'}'. Supported: {', '.join(sorted(SUPPORTED_AUDIO_SUFFIXES))}",
            )

        # The handle stays open for the whole upload: the `with` block is what
        # flushes the last chunk to disk before faster-whisper reads the file.
        # Cleanup is guaranteed for the *entire* request, including a client that
        # disconnects mid-upload (a chunk read that raises before transcription).
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                try:
                    size = 0
                    while chunk := await file.read(UPLOAD_CHUNK):
                        size += len(chunk)
                        tmp.write(chunk)
                finally:
                    await file.close()

            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            logger.info(
                "POST /transcribe %s (%d bytes) from %s",
                file.filename,
                size,
                request.client.host if request.client else "unknown",
            )
            try:
                result = await anyio.to_thread.run_sync(
                    engine.transcribe,
                    str(tmp_path),
                    language or default_language,
                    words,
                )
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as 500 below
                # The real error goes to the server log; the client gets a type
                # name only, so no model paths or internals leak into the body.
                logger.exception("Transcription failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed ({type(exc).__name__})",
                ) from exc

            return _to_out(result, words)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return app


app = create_app()
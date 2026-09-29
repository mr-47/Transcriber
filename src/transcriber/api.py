from __future__ import annotations

import os
import tempfile

import anyio
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from .config import SUPPORTED_AUDIO_SUFFIXES, TranscriberSettings
from .core import Transcriber
from .models import Transcript

SUPPORTED_SUFFIXES = SUPPORTED_AUDIO_SUFFIXES


class HealthResponse(BaseModel):
    status: str
    diarization_enabled: bool


def create_app(transcriber: Transcriber | None = None) -> FastAPI:
    if transcriber is None:
        transcriber = Transcriber(TranscriberSettings())

    app = FastAPI(title="Transcriber", version="0.1.0", description="Audio transcription with speaker diarization")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", diarization_enabled=transcriber.diarization_enabled)

    @app.post("/transcribe", response_model=Transcript, tags=["transcription"])
    async def handle_transcribe(
        file: UploadFile = File(..., description="Audio file to transcribe"),  # noqa: B008
        language: str | None = Query(
            default=None,
            description="Optional language hint for whisper, e.g. 'en', 'de', 'ja'",
        ),
    ) -> Transcript:
        suffix = os.path.splitext(file.filename or "")[1].lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix or 'unknown'}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
            )

        with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as tmp:
            tmp_path = tmp.name
            try:
                while chunk := await file.read(1024 * 1024):
                    tmp.write(chunk)
            finally:
                await file.close()

        try:
            return await anyio.to_thread.run_sync(
                transcriber.transcribe, tmp_path, language
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return app


app = create_app()
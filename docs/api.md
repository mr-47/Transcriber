# REST API

A FastAPI app (`transcriber.api:app`) providing a thin HTTP skin over the same
`Transcriber` engine the CLI uses.

```bash
.venv/bin/uvicorn transcriber.api:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation at http://localhost:8000/docs

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service status, diarization state, model/device/compute-type |
| `POST /transcribe` | Multipart audio upload -> transcript JSON |

### `GET /health`

Response fields:

- `status` — always `"ok"` while the app is up
- `diarization_enabled` — whether `TRANSCRIBER_HF_TOKEN` is set
- `model`, `model_path`, `device`, `compute_type` — what the running instance
  would use; `model_path` is present only for an absolute local path

`/health` responds before any model is loaded, so it is a cheap readiness/health
probe that does not force the model-load delay.

### `POST /transcribe`

Query parameters:

| Parameter | Default | Description |
|---|---|---|
| `language` | auto | Language hint (`en`, `de`, `ru`, ...); omitted means auto-detection |
| `words` | `true` | Include word timestamps under `segments[].words` |

Examples:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.mp3" \
  -F "language=en"

# omit word timestamps to shrink the response
curl -X POST "http://localhost:8000/transcribe?words=false" \
  -F "file=@meeting.mp3"
```

Validation happens before any model runs:

- an unknown file suffix (not one of `mp3 wav flac ogg opus m4a aac wma mp4
  webm mka mkv`) is rejected with HTTP 400;
- an empty upload is rejected with HTTP 400.

A transcription failure is a HTTP 500 whose body says only the exception type
name, so no model paths or internals leak to the client; the full error goes to
the server log.

The response body is the same JSON schema that `--json` and the watcher's
`<name>.json` write: `language`, `language_probability`, `duration`, `segments`,
`utterances`, `text`. See [output-formats.md](output-formats.md).

## One process by design

The app creates one `Transcriber` in its lifespan and reuses it, so models load
on the **first request** and stay warm afterwards. Transcription runs
synchronously in a worker thread (`anyio.to_thread.run_sync`) rather than
blocking the event loop, and `Transcriber.transcribe` serializes every call
behind one lock: whisper/pyannote binaries are not safe for concurrent use
within a process, so concurrent uploads **queue up** behind that lock instead of
competing for the GPU. Do not run multiple workers against the same model — each
worker would load its own copy.

Uploads stream to a seekable temporary file (faster-whisper needs a path). The
temp file is removed in a `finally` covering the entire request, including a
client that disconnects mid-upload.

## Shutdown

On shutdown the app closes the transcriber it created, dropping the model
references so loaded GPU/CPU memory can be freed. An injected transcriber (used
by the test suite) is never closed.

## Security

The API has **no authentication** — the module docstring says so deliberately.
The contract is "bind loopback, or sit behind an authenticated reverse proxy".
There is also no upload-size limit; an unbounded request can fill the temporary
filesystem. Do not expose it directly to the internet.
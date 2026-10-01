# Troubleshooting

## `transcriber: command not found`

The command lives inside the virtualenv. Activate it in every new terminal:

```bash
source .venv/bin/activate
```

or call it by full path (`.venv/bin/transcriber ...`), or add the venv to
`PATH` once:

```bash
echo 'export PATH="$HOME/OpenCode/Transcriber/.venv/bin:$PATH"' >> ~/.bashrc
exec bash
```

## First run pauses for a long time

The first transcription downloads the model (~460 MB for `small`) into the
Hugging Face cache. That pause is expected, not a hang — logging is configured
before model loading so the download is visible.

## `WARNING ... reloading whisper model on CPU`

The message `CUDA inference unavailable; reloading whisper model on CPU` means
the `cuda` extra is not installed in that environment. Reinstall:

```bash
.venv/bin/pip install -e ".[cuda]"
```

## `WARNING ... compute type inferred from the saved model is float16 ... converted to float32`

Expected on a GPU without fast FP16 (Pascal/Maxwell-era cards such as a GTX
1070). This is the documented compute downgrade, **not** the CPU-reload path —
transcription still uses the GPU. Set `TRANSCRIBER_WHISPER_COMPUTE_TYPE=float16`
only if you know the card is eligible.

## `no kernel image is available for execution on the device`

The installed `torch` build ships no kernels for your card's compute capability
(Pascal/Maxwell/Volta on a CUDA 13 build). This usually bites **diarization**,
whose GPU move the code probes separately. Install a CUDA 12.6 build — see
[installation.md](installation.md).

## Diarization pipeline download fails with an access error (401/403)

The pyannote models are gated. You must accept their terms on the model cards
*and* provide a read token. Without a token, diarization is simply disabled (to
the single `SPEAKER_00`), so a 401/403 means the token is stale or the account
has not accepted the terms.

## Nothing is attributed to more than one speaker

Check `TRANSCRIBER_HF_TOKEN`. Empty or unset disables diarization entirely;
`--no-diarization` forces it off. Set the token and re-run.

## Too many / too few speaker labels

Tuning knobs live in [configuration.md](configuration.md). `TRANSCRIBER_DIAR_*`
values affect clustering; `TRANSCRIBER_DIAR_MIN_REGION` (default `1.0`) folds
sub-second singleton speakers into their temporally nearest neighbour.

Be aware that pinned `TRANSCRIBER_DIAR_SPEAKERS` can split a two-party call
~98%/2% *before* attribution runs, and no overlap rule recovers a speaker
missing from the rest of the file — see [diarization.md](diarization.md).

## A file stays in `calls-inbox/`

Either the watcher is waiting for it to stop changing (readiness is checked
against size and mtime), or it has never been decoded. After
`MAX_STALLED_SCANS` (3) *settled* scans it moves to `calls-failed/`. Note that
`watch --once` usually exits before the fail path can trip — a held file is
deferred, not failed; run a continuous `transcriber watch` to see it fail.

## A file sits in `calls-process/`

An interrupted run. `recover_stale()` moves leftovers back to `calls-inbox/` on
the next watch start. Do not manually move a file while an active worker is
still processing it.

## A result is named `call_m4a.*` instead of `call.*`

A stem collision between `call.mp3` and `call.m4a`. Both can coexist — see
[folder-workflow.md](folder-workflow.md).

## An unknown `-o` extension silently yielded markdown once

It no longer happens silently: `-o report.mp3` now prints a warning to stderr
and writes the selected format anyway. Use a `.md/.txt/.html/.srt` extension.

## The API returns 400 / 500

- 400: unsupported file suffix or an empty upload (checked before any model
  runs).
- 500: the transcription failed; the body carries only the exception type name,
  the full error is in the server log. Do not leak model paths/internals into
  the body.

## `transcriber watch --format bogus`

Unknown `--format` values are a `parser.error`, i.e. exit code 2 — never a
silent `md`.

## Is this environment "one machine"? Read this first

Two `DeprecationWarning`s from `starlette.testclient` during the test suite are
expected third-party noise, not failures. `python` may not be on `PATH` in a
dev checkout — use `.venv/bin/*` or `python3`.
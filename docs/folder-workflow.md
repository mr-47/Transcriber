# Folder Workflow

`transcriber watch` moves audio through four directories below the base
directory (default: current directory, override with `--dir`).

| Folder | Purpose |
|---|---|
| `calls-inbox/` | New calls waiting to be processed |
| `calls-process/` | Call currently being transcribed (in transit) |
| `calls-failed/` | Calls that could not be processed, plus a `<name>.txt` error report |
| `calls-results/` | Successfully transcribed calls with their transcripts |

All four folders are created automatically on start.

## Movement semantics

Files are **moved** between folders, never copied. On success the audio keeps
its name and lands in `calls-results/` next to `<name>.json` (always with word
timestamps) and the requested transcript format(s). On failure the audio moves
to `calls-failed/` with a `<name>.txt` report containing the error and
traceback.

The invariant: **a file is never silently dropped and never stranded.** In
particular, post-transcription finalise (stem selection, JSON, rendered
formats, move to results) is wrapped so that *any* write or render failure
routes the audio to `calls-failed/` with a report — nothing stays in
`calls-process/`.

## Files still being written

The watcher does not fail a file that is still being written. Readiness is a
state machine per file name:

1. First sighting: a real decode probe (`probe_audio`, built on PyAV). Readable
   files are processed immediately; unreadable ones are recorded and held.
2. Later sightings: if size **or** mtime changed, the source is still writing
   (or the file was replaced at the same size), so it is held without even
   probing — this also catches header-first formats the probe cannot spot.
3. Once the stamp settles: probe again. Readable -> process; still unreadable
   after `MAX_STALLED_SCANS` (3) settled scans -> fail.

The mtime check exists because a source that *replaces* a fixed-size stream
(e.g. a circular capture) would otherwise never trip the size gate.

`probe_audio` never guesses: if PyAV itself is missing it raises a
`RuntimeError` naming the fix, rather than returning `False` (which would
methodically file every inbox item as failed). The `(frame) -> ImportError`
guard is written to raise, never to guess.

## Filename collisions

Two recordings can share a stem but not an extension:

```text
call.mp3
call.m4a
```

Both want `call.json` / `call.md`. Ownership is read off the **audio files
sitting next to the transcripts**, never guessed: `call.*` belongs to whichever
`call.<ext>` audio is present. The second conflicting input is disambiguated by
folding its extension into the stem:

```text
call_m4a.json
call_m4a.md
```

A collision that numbered fallback cannot dodge (an identical stem and
extension on a re-run over an existing result) writes `call_m4a_2`, `_3`, ...
A `RuntimeError` after exhausting 2..999 is intentional — recording workflows
should never reach it.

Re-running an existing success overwrites in place: the transcript of a
moved-back recording is written under the original stem, not accumulated.

Failure reports disambiguate the same way, with two extra rules:

- if the failed input is itself a `.txt` (the unsupported-file-type case), the
  report is `<name>.error.txt` so it cannot destroy the file it describes;
- a report claims its name even after its audio is gone (`report_claims`), so a
  reason is never overwritten.

## Crash recovery

At watch start, leftover files in `calls-process/` from an interrupted run are
moved back to `calls-inbox/` (`recover_stale`). Unsupported file types are moved
to `calls-failed/` with a reason.

`recover_stale` returns the recovered list, and the caller reports it. Do not
drop that return value — the tests rely on it.

## Usage

```bash
transcriber watch                      # poll calls-inbox every 5s forever
transcriber watch --dir /data/calls    # use a different base directory
transcriber watch --once               # drain the inbox once, then exit
transcriber watch --interval 1         # poll faster
transcriber watch --language ru        # hint the language on every file
transcriber watch --model tiny         # override the model size
transcriber watch --no-diarization     # single speaker

# multiple formats (comma-separated or repeat the flag; default: md)
transcriber watch --once --format txt,md,html,srt
transcriber watch --once --format txt --format html
```

`--format` is an append flag that accepts comma-separated lists, skips empty
parts, and rejects an unknown value with `parser.error` (exit code 2) rather
than silently writing `md`.

## Output behavior

Watch mode always writes `<name>.json` with word timestamps, because it is the
durable machine-readable archive. The rendered formats are the same renderers
the CLI uses — see [output-formats.md](output-formats.md).

> Note: `watch --once` can defer a held file and exit without it. The
> `MAX_STALLED_SCANS = 3` fail path needs three *settled* scans to trip, and a
> `--once` run usually exits after one. Hold-backs in `--once` are deferred, not
> failed — run a continuous `transcriber watch` to exercise the fail path.
# Known Limitations

## REST API security

The API has no authentication and no upload-size limit. Uploads stream to a
temporary file, so an unbounded request can fill the temporary filesystem. Keep
it bound to loopback, or put it behind an authenticated reverse proxy before
exposing it to any network.

## One process, one model, one lock

`Transcriber.transcribe` serializes everything behind a single lock because the
whisper/pyannote binaries are not safe for concurrent use within one process.
API requests queue up behind that lock rather than fighting over the GPU. Do
not run multiple API workers against one model: each loads its own copy.

## Speaker identity

Speaker labels (`SPEAKER_00`, ...) are local to each recording and stable only
within one file. Transcriber performs diarization, not persistent speaker
recognition — nothing links `SPEAKER_00` across two files.

## Diarization quality

- Diarization requires a token for gated pyannote models; without it there is a
  single `SPEAKER_00`.
- Attribution is segment-level, not word-level; speech spanning a speaker
  transition is assigned by the overlap rule, not per word.
- Overlapping speech is attributed with *latest-start wins* among covering
  turns. That keeps one long turn from swallowing a file, but it cannot repair
  a degenerate partition: pinned `TRANSCRIBER_DIAR_SPEAKERS` can split a
  two-party call ~98%/2% before attribution ever runs.
- Long meetings diarize less cleanly than short two-party calls; the
  `TRANSCRIBER_DIAR_MIN_REGION` cleanup pass is a mitigation, not a cure.

## Fixed speaker count

`TRANSCRIBER_DIAR_SPEAKERS` is passed straight through as pyannote
`num_speakers`. It is reliable only when the caller actually knows the speaker
count; forcing `2` on a two-party call has been observed to produce a ~98%/2%
split that collapses the transcript into one block. When in doubt, leave it
unset.

## GPU scope

- faster-whisper CUDA needs the `cuda` extra (the cuBLAS runtime) and falls
  back to `cpu,int8` otherwise.
- The float16->float32 compute downgrade on Pascal/Maxwell-era cards is
  expected, not an error.
- Windows support is **parse-tested under Wine only**; end-to-end Windows
  installation and GPU inference have not been verified on real Windows
  hardware.

## Models and first run

The first transcription downloads the Whisper model (`small` ~460 MB) plus,
with a token, the diarization pipeline. That download is not part of the
installers. There is no offline mode beyond what the Hugging Face cache already
holds.

## Word timestamps in the one-shot CLI

Word-level timestamps are only computed when `--json` and `--words` are both
present, by deliberate design. Do not expect `segments[].words` in `--json`
output without `--words`.

## Edge cases that are intentional

- A `.txt` input failing produces `<name>.error.txt` so the report cannot
  destroy the file it describes.
- Two files with an identical stem *and* extension colliding 998 times over
  raises `RuntimeError` — recording workflows should never reach it.
- `av` is a hard dependency of the folder workflow; a missing PyAV raises rather
  than looking like "corrupt audio".
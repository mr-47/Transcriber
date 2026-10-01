# Speaker Diarization

Transcriber uses [pyannote.audio](https://github.com/pyannote/pyannote-audio)
for speaker diarization: a pipeline (`pyannote/speaker-diarization-3.1` by
default) yields speaker turns, which are then merged with the whisper segments.

## Token gate

Diarization is enabled by `TRANSCRIBER_HF_TOKEN` and nothing else:
`bool(hf_token)`. The models are gated — you must accept their terms on the
Hugging Face model cards and provide a read token (see
[installation.md](installation.md)). An empty token disables diarization, so
every utterance is attributed to the single `SPEAKER_00`. The same gate backs
the CLI's `--no-diarization` flag (it clears the token), so an empty
`TRANSCRIBER_HF_TOKEN=""` disables diarization too — that is a feature.

## Labels and identity

Output labels are `SPEAKER_00`, `SPEAKER_01`, ... assigned in clustering order
and stable only within one recording.

## Tuning knobs

All knobs default to "leave the pipeline's pretrained hyperparameters alone"
(`-1`); only values `>= 0` are applied. pyannote stores its configuration in
`pipeline.hyperparameters`, so tuning is a *dict edit*, not a
re-instantiation — a version change that drops a known key is caught and
skipped with a warning rather than failing the transcription.

| Variable | hyperparameters slot | Effect |
|---|---|---|
| `TRANSCRIBER_DIAR_THRESHOLD` | `clustering.threshold` | Lower splits more speakers, higher merges them |
| `TRANSCRIBER_DIAR_MIN_ON` | `segmentation.min_duration_on` | Minimum speaker-region length |
| `TRANSCRIBER_DIAR_MIN_OFF` | `segmentation.min_duration_off` | Minimum silence that closes a region |
| `TRANSCRIBER_DIAR_SPEAKERS` | passed as pyannote `num_speakers` at call time (only when set) | Forces exactly that many clusters |

The GPU move is separate and guarded: pyannote runs on CPU when
`_torch_cuda_usable()` says the torch build cannot actually launch kernels on
the device, and warns. This never fails the transcription.

## Tiny-turn absorption

`TRANSCRIBER_DIAR_MIN_REGION` (default `1.0`) runs a post-processing pass,
`absorb_tiny_turns`. A sub-second turn cannot carry a reliable embedding and
becomes a singleton label — but on a dense two-party call, sub-second turns are
exactly the interjections that mark a speaker change. Dropping them merged whole
minutes of dialogue into single-speaker blocks once. The pass therefore:

- keeps **every** turn (attribution keeps its evidence),
- computes each speaker's total speech,
- reassigns only the label of any speaker under the floor to the temporally
  nearest surviving speaker.

`0` disables the pass. Do not "clean up" by dropping tiny turns.

## Attribution rules

`assign_speakers` aligns whisper segment midpoints to turns. Overlapping turns
resolve with **latest-start wins**; fallbacks are the nearest turn, then the
previously assigned speaker, then `SPEAKER_00`. Consecutive segments of the
same speaker merge into utterances.

### Attribution is label-only; it cannot repair a bad partition

If a pinned `TRANSCRIBER_DIAR_SPEAKERS` splits a two-party call ~98%/2% by
speech (pyannote `num_speakers`), no overlap rule recovers the speaker who is
missing from the rest of the file. Do not chase that symptom in `merge.py`; the
cleanup passes above are the fix.

## Limitations

- Attribution is segment-level, not word-level; there is no per-word speaker
  labelling.
- Overlapping speech is handled only by the overlap rule, which keeps one turn
  from swallowing a file but cannot conjure a speaker out of a degenerate
  partition.
- Long multi-party meetings cluster less cleanly than short two-speaker calls.
- Nothing survives between files: `SPEAKER_00` in one recording is unrelated to
  `SPEAKER_00` in another.
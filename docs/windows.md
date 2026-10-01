# Windows Support

Windows shares the same Python code, CLI, folder workflow, API, diarization
logic, output formats and environment variables as Linux.

## Two launchers

| Script | Purpose |
|---|---|
| `install.cmd` | One-command setup: check Python, create `.venv`, install deps, offer the `cuda` extra, create the `calls-*` folders |
| `run.cmd` | Start the folder watcher; forwards `%*` to `transcriber watch` |

`run.cmd` mirrors `run.sh`: default formats `md,txt,html` from
`TRANSCRIBER_FORMATS`, everything else forwarded to `transcriber watch`. So
`run.cmd --once` drains the inbox and exits, `run.cmd --format srt` changes the
written formats, and so on.

Models are **not** downloaded by `install.cmd`; they pull into the Hugging Face
cache on first run (`%USERPROFILE%\.cache\huggingface`), exactly as on Linux.

## `install.cmd` behavior

Checks, in order:

1. **Python prerequisites** — `python` must be on PATH and >= 3.10 (compared
   via `sys.version_info`, not by text-parsing `Python 3.11.9`, which sorts
   3.10 below 3.9). Probes are spelled with `if errorlevel`, not `&&`/`||`.
2. **GPU probe** — `nvidia-smi --query-gpu=name` for the card name. If found,
   prompts to install the `cuda` extra (~600 MB wheel). The probe is a
   single-line `for /f ... do if not defined GPUNAME set "GPUNAME=%%G"` line,
   deliberately not a multi-line parenthesized block (Wine's `cmd` runs a
   `for /f` body even with no output and fails to substitute the loop variable
   inside such a block).
3. **venv** — a working `.venv` is kept; a broken/moved one is replaced.
4. **dependencies** — `.[dev]`, or `.[dev,cuda]` when the flag/choice says so.
5. **`calls-*` folders** — created with `.gitkeep` files.

Flags: `--cuda`, `--no-cuda`, `--help`. Anything else starting with a dash is
rejected with exit code 1 (sliced with delayed expansion `!ARG:~0,1!`, because
`findstr /c` treats its pattern as a literal and cannot express "starts with").
Set the token persistently with `setx TRANSCRIBER_HF_TOKEN hf_...`.

`install.cmd` enables delayed expansion; `run.cmd` must **not** — it forwards
`%*`, and with `EnableDelayedExpansion` an `!` in a path or argument would be
consumed instead of passed through. That is why its probes are flat
`if ... set` lines rather than nested blocks needing `!VAR!`.

## CRLF requirement

Both `.cmd` files are 100% CRLF and must stay that way — real `cmd.exe`
misparses LF-only batch, most visibly at `goto`/labels, which is exactly what
the option parsing and `:help` depend on. `.gitattributes` pins
`*.cmd text eol=crlf`; verify with Python (`data.count("\n")` vs
`data.count("\r\n")`), not `awk`.

## Verification status

Verified:

- Windows batch logic under Wine 9.0 (`cmd.exe`): `install.cmd --help` exits 0,
  `install.cmd --bogus` exits 1, the no-argument run bails at the Python
  prerequisite, `run.cmd` reports ".venv is missing".

Not verified (Wine cannot be):

- end-to-end installation on real Windows hardware;
- real `nvidia-smi` GPU detection (Wine's `cmd` behaves differently on empty
  `for /f` output, so only the *consumer* of `GPUNAME` — the install prompt —
  is exercised);
- GPU transcription on a real Windows CUDA setup.

Keep this limitation visible until real-hardware validation is complete.
Linux remains the primary development and verification platform.
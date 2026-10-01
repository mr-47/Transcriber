@echo off
REM Start the folder watcher: drop a recording into calls-inbox\ and the
REM transcript appears in calls-results\.
REM
REM Usage:
REM   run.cmd                  watch, writing md + txt + html
REM   run.cmd --once           drain the inbox and exit (good for scheduled tasks)
REM
REM Anything you pass is handed to `transcriber watch`, so every flag works:
REM   run.cmd --format srt --language en
setlocal EnableExtensions
cd /d "%~dp0"

REM A venv that will not start is almost always one whose checkout moved, not
REM a broken install. Recreate it with install.cmd rather than patching paths.
if not exist ".venv\Scripts\python.exe" (
  echo error: .venv is missing. Run install.cmd first. 1>&2
  exit /b 1
)

for %%d in (calls-inbox calls-process calls-failed calls-results) do if not exist "%%d" mkdir "%%d"

REM activate.bat sets VIRTUAL_ENV and prepends Scripts to PATH, which is all a
REM child process needs; there is no environment to undo afterwards.
call ".venv\Scripts\activate.bat"

REM A venv can exist while the install into it failed, and then batch's own
REM "not recognized" message names nothing actionable. The console script is
REM looked for by path, not on PATH, so this does not depend on a PATH search.
if not exist ".venv\Scripts\transcriber.exe" if not exist ".venv\Scripts\transcriber-script.py" (
  echo error: transcriber is not installed in .venv. Re-run install.cmd. 1>&2
  exit /b 1
)

REM Without a token every recording is attributed to a single speaker. Saying
REM so once per run is cheaper than the surprise of a transcript with no labels.
if not defined TRANSCRIBER_HF_TOKEN (
  echo note: TRANSCRIBER_HF_TOKEN is not set, speaker diarization is disabled; 1>&2
  echo       set it with `setx TRANSCRIBER_HF_TOKEN hf_...` to enable labels. 1>&2
  echo.
)

REM run.sh's ${TRANSCRIBER_FORMATS:-md,txt,html} has no cmd shorthand, so the
REM default is spelled out. The env var is watched directly rather than copied,
REM so an empty TRANSCRIBER_FORMATS also falls back to this default.
if not defined TRANSCRIBER_FORMATS ( set "FORMATS=md,txt,html" ) else ( set "FORMATS=%TRANSCRIBER_FORMATS%" )

echo Watching calls-inbox\ ... transcripts land in calls-results\
echo Ctrl-C to stop.
echo.

transcriber watch --format "%FORMATS%" %*
exit /b %ERRORLEVEL%
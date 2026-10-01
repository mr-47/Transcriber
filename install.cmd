@echo off
REM One-command setup for Transcriber on Windows.
REM
REM Creates the virtualenv, installs the dependencies, detects an NVIDIA GPU
REM and offers the CUDA extra faster-whisper needs, and creates the calls-*
REM working folders. Safe to re-run: anything already present is left alone.
REM
REM Models are NOT downloaded here: Transcriber fetches its Whisper and
REM diarization weights into the Hugging Face cache on the first run, the same
REM way it does on Linux.
REM
REM Usage:
REM   install.cmd                probe for an NVIDIA GPU and offer the CUDA extra
REM   install.cmd --cuda         install the CUDA extra unconditionally
REM   install.cmd --no-cuda      never offer it (the default when no GPU is found)
REM
REM Then: run.cmd
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "CUDA="
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--cuda" ( set "CUDA=1" & shift & goto parse )
if /i "%~1"=="--no-cuda" ( set "CUDA=0" & shift & goto parse )
if /i "%~1"=="--help" goto help
if /i "%~1"=="-h" goto help
REM Anything else starting with a dash is a typo, not a flag. Sliced out with
REM delayed expansion rather than tested with findstr, which treats its /c
REM argument as a literal and so cannot express "starts with" at all.
set "ARG=%~1"
if "!ARG:~0,1!"=="-" (
  echo error: unknown option: %~1 1>&2
  exit /b 1
)
echo error: unexpected argument: %~1 1>&2
exit /b 1
:parsed

call :step "Checking prerequisites"
REM Each tool is probed by running it, not with "where": what matters is
REM whether it can be executed. Both checks are spelled with "if errorlevel"
REM rather than && / ||, which some emulators mishandle with redirection.
python --version >nul 2>&1
if errorlevel 1 ( echo error: Python is required. Install it from python.org and tick "Add to PATH". 1>&2 & exit /b 1 )
REM 3.10 is the floor pyproject.toml declares. Compared against sys.version_info
REM rather than parsing "Python 3.11.9" as text, which sorts 3.10 below 3.9 and
REM rejects 3.100. Checked before the venv exists, so a failure names the
REM interpreter rather than surfacing later as pip refusing a wheel.
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 ( echo error: Python 3.10 or later is required; pyproject.toml declares the floor. 1>&2 & exit /b 1 )
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "PYVER=%%V"
echo   python %PYVER%

REM faster-whisper drives the GPU through CTranslate2's CUDA backend, which on a
REM plain "pip install ." is missing the cuBLAS runtime it dlopen()s by soname --
REM the `cuda` extra provides it. The extra is ~600 MB, so it is offered rather
REM than assumed. nvidia-smi is probed for the card name; its absence also
REM covers the machine with no usable NVIDIA GPU, but a CUDA setup where it is
REM not on PATH is missed, which is what --cuda is for.
set "GPUNAME="
REM Single-line `do if not defined ... set`, deliberately not a multi-line
REM parenthesised block: a for-variable inside one is expanded correctly by
REM cmd but not by Wine's cmd, which also runs the body even when the command
REM produced no output. That combination makes the loop look like it detects
REM a GPU on a machine with none.
for /f "delims=" %%G in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do if not defined GPUNAME set "GPUNAME=%%G"
if defined GPUNAME echo   gpu: !GPUNAME!
if not defined GPUNAME echo   gpu: none detected; pass --cuda to install the extra anyway
if not defined CUDA if defined GPUNAME (
  echo.
  set /p "CHOICE=   Install the CUDA extra (needed for GPU inference, ~600 MB)? [y/N]: "
  if /i "!CHOICE!"=="y" set "CUDA=1"
  if /i "!CHOICE!"=="yes" set "CUDA=1"
)

call :step "Creating the virtualenv"
REM A venv whose checkout was moved has absolute paths baked into its scripts,
REM so a broken one is replaced. A working one is always kept.
if exist ".venv\Scripts\python.exe" (
  echo   .venv already exists, keeping it
) else (
  rmdir /s /q .venv 2>nul
  python -m venv .venv || ( echo error: could not create .venv 1>&2 & exit /b 1 )
  echo   created .venv
)

call :step "Installing dependencies"
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
REM The dev extras match install.sh; the CUDA extra is only worth its download
REM size on a machine that is actually going to use the GPU.
if "!CUDA!"=="1" (
  ".venv\Scripts\python.exe" -m pip install --quiet -e ".[dev,cuda]" || ( echo error: dependency install failed 1>&2 & exit /b 1 )
) else (
  ".venv\Scripts\python.exe" -m pip install --quiet -e ".[dev]" || ( echo error: dependency install failed 1>&2 & exit /b 1 )
)
for /f "delims=" %%V in ('".venv\Scripts\python.exe" -c "import transcriber; print(transcriber.__version__)"') do set "VER=%%V"
echo   transcriber %VER%

call :step "Creating the working folders"
for %%d in (calls-inbox calls-process calls-failed calls-results) do (
  if not exist "%%d" mkdir "%%d"
  if not exist "%%d\.gitkeep" type nul > "%%d\.gitkeep"
)
echo   drop recordings into calls-inbox\

if defined TRANSCRIBER_HF_TOKEN (
  echo   diarization: enabled (TRANSCRIBER_HF_TOKEN is set)
) else (
  echo.
  echo   Note: TRANSCRIBER_HF_TOKEN is not set, speaker diarization is disabled.
  echo   Set it persistently with:  setx TRANSCRIBER_HF_TOKEN hf_...
  echo   Accept the gated-model terms first - see README "Diarization setup".
)

echo.
echo Setup finished.  Start the watcher with:  run.cmd
echo.
exit /b 0

REM ---------------------------------------------------------------------
:step
echo.
echo ==^> %~1
goto :eof

:help
REM Written out rather than parsed back out of the header above: findstr /c
REM treats its argument as a literal, so an anchored pattern is impossible.
echo One-command setup for Transcriber on Windows.
echo.
echo Creates the virtualenv, installs the dependencies, and creates the
echo calls-* working folders. Safe to re-run: anything already present is
echo left alone. Models download on the first run, not here.
echo.
echo Usage:
echo   install.cmd                probe for an NVIDIA GPU and offer the CUDA extra
echo   install.cmd --cuda         install the CUDA extra unconditionally
echo   install.cmd --no-cuda      never offer it ^(the default when no GPU is found^)
echo.
echo Then: run.cmd
goto :eof
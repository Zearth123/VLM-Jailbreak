@echo off
rem ============================================================
rem  YuFeng-XGuard-Reason-0.6B launcher
rem  ASCII only: cmd.exe reads .bat/.cmd in the OEM codepage,
rem  and non-ASCII here would break parsing.
rem
rem    xguard.cmd --self-test
rem    xguard.cmd --text "some text"
rem    xguard.cmd --items artifacts\items.json --out artifacts\out.jsonl
rem    xguard.cmd --items ... --out ... --reason
rem ============================================================
setlocal
set "HERE=%~dp0"

rem --- deps: torch 2.14.1+cpu / transformers 5.20.1 / accelerate ---
set "LIBS=D:\VLMjalibreak\.tmp\pylibs"
if not exist "%LIBS%\torch" if exist "D:\VLMjalibreak\.pylibs\torch" set "LIBS=D:\VLMjalibreak\.pylibs"

set "PY=C:\Users\Zearth\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if not exist "%PY%" for %%I in (python.exe) do set "PY=%%~$PATH:I"
if not exist "%PY%" (
  echo [xguard] python not found; edit PY in xguard.cmd 1>&2
  exit /b 1
)

set "PYTHONPATH=%LIBS%;%HERE%;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if "%~1"=="" goto usage

"%PY%" "%HERE%tools\xguard_cli.py" %*
exit /b %ERRORLEVEL%

:usage
echo YuFeng-XGuard-Reason-0.6B  (29-category risk classifier; "sec" = Safe)
echo.
echo   xguard.cmd --self-test
echo   xguard.cmd --text "some text"
echo   xguard.cmd --items IN.json --out OUT.jsonl
echo   xguard.cmd --items IN.json --out OUT.jsonl --reason
echo   xguard.cmd --help
echo.
echo   CPU wheel installed: add --dtype float32 for speed/stability.
exit /b 0

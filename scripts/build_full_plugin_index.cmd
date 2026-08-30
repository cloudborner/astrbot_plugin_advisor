@echo off
setlocal
cd /d "%~dp0.."
set "ADVISOR_PYTHON=.venv\Scripts\python.exe"
if not exist "%ADVISOR_PYTHON%" set "ADVISOR_PYTHON=python"
"%ADVISOR_PYTHON%" scripts\build_full_plugin_index.py %*
set "ADVISOR_EXIT=%ERRORLEVEL%"
echo.
if "%ADVISOR_EXIT%"=="0" (
  echo Full plugin source index completed successfully.
) else (
  echo Pipeline stopped or completed with failures. Re-run to resume.
)
pause
exit /b %ADVISOR_EXIT%

@echo off
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%CD%\venv\Lib\site-packages"
set "APP_URL=http://127.0.0.1:5001/login"
set "BUNDLED_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if exist "%BUNDLED_PYTHON%" (
    start "" "%APP_URL%"
    "%BUNDLED_PYTHON%" app.py
    goto :end
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    start "" "%APP_URL%"
    python app.py
    goto :end
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    start "" "%APP_URL%"
    py app.py
    goto :end
)

echo Python was not found. Install Python 3.10 or 3.11, then run this file again.
pause

:end
endlocal

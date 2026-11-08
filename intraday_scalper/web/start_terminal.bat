@echo off
setlocal EnableExtensions
chcp 65001 >nul
title SOXL-SOXS Terminal Launcher
set "HERE=%~dp0"
set "ROOT=%HERE%..\.."
pushd "%ROOT%" >nul
set "PY=%ROOT%\intraday_scalper\.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:8080/"

if not exist "%PY%" (
  echo [ERROR] Python venv not found:
  echo   %PY%
  pause
  popd >nul
  exit /b 1
)

call :check_health
if defined HEALTH_OK goto already

echo ============================================================
echo  SOXL / SOXS Intraday Terminal - starting
echo ============================================================
echo Server: %URL%
echo Starting background server (first start may take 10-40s)...
start "SOXL-SOXS-TERMINAL" /min "%PY%" -m intraday_scalper.web.terminal_server --port 8080

set /a TRIES=0
:wait
timeout /t 2 /nobreak >nul
call :check_health
if defined HEALTH_OK goto ready
set /a TRIES+=1
if %TRIES% GEQ 30 goto fail
goto wait

:ready
echo Server is ready. Opening browser...
start "" "%URL%"
echo.
echo Terminal is running in the background window "SOXL-SOXS-TERMINAL".
echo Double-click stop_terminal.bat (or close that window) to stop it.
timeout /t 3 /nobreak >nul
popd >nul
exit /b 0

:already
echo Server already running at %URL%
echo Opening browser...
start "" "%URL%"
popd >nul
exit /b 0

:fail
echo [ERROR] Server did not become ready in time.
echo Try running the server manually to see the error:
echo   "%PY%" -m intraday_scalper.web.terminal_server --port 8080
pause
popd >nul
exit /b 1

:check_health
set "HEALTH_OK="
set "CODE="
for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "try{(Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%URL%healthz').StatusCode}catch{0}"`) do set "CODE=%%H"
if "%CODE%"=="200" set "HEALTH_OK=1"
exit /b 0

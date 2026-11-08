@echo off
setlocal EnableExtensions
chcp 65001 >nul
title SOXL-SOXS Terminal Stopper
echo ============================================================
echo  SOXL / SOXS Intraday Terminal - stopping
echo ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'intraday_scalper.web.terminal_server' }; if (-not $procs) { Write-Host 'No running terminal server found.' } else { foreach ($p in $procs) { Write-Host ('Stopping PID ' + $p.ProcessId + ' ...'); & taskkill /F /T /PID $p.ProcessId 2>&1 | Out-Null } }"

echo.
echo Done. If the browser page is still open, refresh it to confirm it is offline.
timeout /t 3 /nobreak >nul

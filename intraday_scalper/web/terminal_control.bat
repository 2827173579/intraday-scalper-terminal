@echo off
setlocal EnableExtensions
chcp 65001 >nul
title SOXL-SOXS Terminal Control

:menu
cls
echo ============================================================
echo   SOXL / SOXS Intraday Terminal - Control Panel
echo ============================================================
echo   URL: http://127.0.0.1:8080/
echo.
echo   [1] Start / open terminal
echo   [2] Stop terminal
echo   [3] Just open browser
echo   [0] Exit
echo.
set /p CHOICE=Choose [1/2/3/0]: 

if "%CHOICE%"=="1" goto start
if "%CHOICE%"=="2" goto stop
if "%CHOICE%"=="3" goto open
if "%CHOICE%"=="0" exit /b 0
echo Invalid choice.
timeout /t 2 /nobreak >nul
goto menu

:start
call "%~dp0start_terminal.bat"
goto menu

:stop
call "%~dp0stop_terminal.bat"
goto menu

:open
start "" "http://127.0.0.1:8080/"
timeout /t 2 /nobreak >nul
goto menu

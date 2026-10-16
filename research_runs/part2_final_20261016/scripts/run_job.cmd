@echo off
REM usage: run_job.cmd NAME script.py [args...]  -> logs\NAME.log, logs\NAME.exitcode
cd /d "D:\PROJECT\VeriFace AI\research_runs\part2_final_20261016"
set HF_HUB_OFFLINE=1
set NAME=%1
shift
set ARGS=
:loop
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto loop
:run
echo START %DATE% %TIME% ARGS=%ARGS% > logs\%NAME%.exitcode
"C:\Users\anshu\AppData\Local\Programs\Python\Python313\python.exe" -I -B %ARGS% > logs\%NAME%.log 2>&1
echo EXIT=%ERRORLEVEL% %DATE% %TIME% >> logs\%NAME%.exitcode

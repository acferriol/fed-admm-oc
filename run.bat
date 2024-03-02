@echo off
start py server.py
timeout /t 10 /nobreak > nul

for /l %%i in (1,1,4) do (
    start py client.py
)
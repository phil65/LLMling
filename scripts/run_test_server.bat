@echo off
:: scripts/run_test_server.bat
"%~dp0\..\\.venv\Scripts\uv.exe" run "%~dp0\..\\.venv\Scripts\python.exe" -m llmling.test_server

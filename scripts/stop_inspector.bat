@echo off
:: scripts/stop_inspector.bat

:: Kill any running inspector processes
taskkill /f /im node.exe /fi "WINDOWTITLE eq MCP Inspector*"

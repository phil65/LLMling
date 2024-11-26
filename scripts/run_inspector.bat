@echo off
:: scripts/run_inspector.bat

:: Set environment variables properly for Windows
set "CLIENT_PORT=5173"
set "SERVER_PORT=3000"

:: Start the MCP Inspector with our server
start "MCP Inspector" cmd /c "npx -y @modelcontextprotocol/inspector uvx --directory . run uv run llmling src/llmling/config_resources/test.yml"

:: Wait a moment
timeout /t 2

:: Open the browser
start http://localhost:5173

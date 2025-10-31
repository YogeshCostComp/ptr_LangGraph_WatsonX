# LangGraph Agent Startup Script
# This script starts the API server and opens the React chat UI

Write-Host "Starting LangGraph ReAct Agent..." -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to the project directory
Set-Location $scriptDir

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Python found" -ForegroundColor Green

# Check if Node/npm is available
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: npm is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] npm found" -ForegroundColor Green

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "[WARN] .env file not found" -ForegroundColor Yellow
} else {
    Write-Host "[OK] .env file found" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting API Server..." -ForegroundColor Cyan

# Start the API server in a new PowerShell window
$serverProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir'; python run_api.py" -PassThru

# Wait a bit for the server to start
Write-Host "Waiting for API server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Check if server is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 2 -ErrorAction Stop
    Write-Host "[OK] API Server is running on http://localhost:8000" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Server might still be starting..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting React UI..." -ForegroundColor Cyan

# Start the React UI in a new PowerShell window
$chatUiDir = Join-Path $scriptDir "chat-ui-react"
$uiProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$chatUiDir'; npm run dev" -PassThru

Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "URLs:" -ForegroundColor Cyan
Write-Host "   React UI:       http://localhost:3000"
Write-Host "   API Server:     http://localhost:8000"
Write-Host "   API Docs:       http://localhost:8000/docs"
Write-Host "   Chat Endpoint:  http://localhost:8000/v1/chat/completions"
Write-Host ""
Write-Host "Tips:" -ForegroundColor Yellow
Write-Host "   - React UI will open automatically at http://localhost:3000"
Write-Host "   - API server is running on http://localhost:8000"
Write-Host "   - Both are running in separate windows"
Write-Host "   - Close those windows to stop the services"
Write-Host ""
Write-Host "Waiting 5 seconds for React UI to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Open the React UI in default browser
Write-Host "Opening React UI in browser..." -ForegroundColor Cyan
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "Press any key to exit this window..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

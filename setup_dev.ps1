#!/usr/bin/env pwsh
# PowerShell script to set up the easy_glm development environment

Write-Host "Setting up easy_glm development environment..." -ForegroundColor Green

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Please install uv first:" -ForegroundColor Yellow
    Write-Host "  curl -LsSf https://astral.sh/uv/install.sh | sh" -ForegroundColor Yellow
    Write-Host "  Or visit: https://github.com/astral-sh/uv" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Blue
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Blue
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    & .\venv\Scripts\Activate.ps1
} else {
    & source venv/bin/activate
}

# Install dependencies using uv
Write-Host "Installing dependencies with uv..." -ForegroundColor Blue
uv pip install -r requirements-dev.txt

# Install the package in editable mode
Write-Host "Installing package in editable mode..." -ForegroundColor Blue
uv pip install -e .

Write-Host "Setup complete! 🎉" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Yellow
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
} else {
    Write-Host "  source venv/bin/activate" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "To start Jupyter notebook, run:" -ForegroundColor Yellow
Write-Host "  jupyter notebook" -ForegroundColor Cyan 
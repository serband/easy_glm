#!/bin/bash
# Shell script to set up the easy_glm development environment on Unix/Linux/macOS

set -e  # Exit on any error

echo "🚀 Setting up easy_glm development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Or visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set UV_PYTHON to use the venv python
export UV_PYTHON="$(pwd)/venv/bin/python"

# Install dependencies using uv
echo "Installing dependencies with uv..."
uv pip install -r requirements-dev.txt

# Install the package in editable mode
echo "Installing package in editable mode..."
uv pip install -e .

echo "✅ Setup complete! 🎉"
echo ""
echo "To activate the environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "To start Jupyter notebook, run:"
echo "   jupyter notebook"
echo ""
echo "To run tests, use:"
echo "   pytest" 
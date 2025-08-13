#!/usr/bin/env python3
"""
Cross-platform setup script for easy_glm development environment.
Uses uv and venv for dependency management.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def run_command(cmd, shell=False):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except FileNotFoundError:
        return False, "", "Command not found"

def check_uv_installed():
    """Check if uv is installed."""
    success, _, _ = run_command(["uv", "--version"])
    return success

def create_venv():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        success, stdout, stderr = run_command([sys.executable, "-m", "venv", "venv"])
        if not success:
            print(f"Error creating virtual environment: {stderr}")
            return False
    return True

def get_activation_command():
    """Get the appropriate activation command for the platform."""
    if platform.system() == "Windows":
        return r"venv\Scripts\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install dependencies using uv."""
    print("Installing dependencies with uv...")

    # Get the appropriate pip command for the virtual environment
    if platform.system() == "Windows":
        pip_cmd = ["uv", "pip", "install"]
    else:
        pip_cmd = ["uv", "pip", "install"]

    # Install development dependencies
    success, stdout, stderr = run_command(pip_cmd + ["-r", "requirements-dev.txt"])
    if not success:
        print(f"Error installing dependencies: {stderr}")
        return False

    # Install package in editable mode
    print("Installing package in editable mode...")
    success, stdout, stderr = run_command(pip_cmd + ["-e", "."])
    if not success:
        print(f"Error installing package: {stderr}")
        return False

    return True

def main():
    """Main setup function."""
    print("🚀 Setting up easy_glm development environment...")

    # Check if uv is installed
    if not check_uv_installed():
        print("❌ uv is not installed. Please install uv first:")
        print("   For Unix/Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   For Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        print("   Or visit: https://github.com/astral-sh/uv")
        sys.exit(1)

    # Create virtual environment
    if not create_venv():
        print("❌ Failed to create virtual environment")
        sys.exit(1)

    # Set UV_PYTHON environment variable to use the venv python
    if platform.system() == "Windows":
        python_path = str(Path.cwd() / "venv" / "Scripts" / "python.exe")
    else:
        python_path = str(Path.cwd() / "venv" / "bin" / "python")

    os.environ["UV_PYTHON"] = python_path

    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)

    print("✅ Setup complete! 🎉")
    print()
    print("To activate the environment, run:")
    print(f"   {get_activation_command()}")
    print()
    print("To start Jupyter notebook, run:")
    print("   jupyter notebook")
    print()
    print("To run tests, use:")
    print("   pytest")

if __name__ == "__main__":
    main()

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
        result = subprocess.run(
            cmd, shell=shell, check=True, capture_output=True, text=True
        )
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

    pip_cmd = ["uv", "pip", "install"]

    success, stdout, stderr = run_command(pip_cmd + ["-r", "requirements-dev.txt"])
    if not success:
        print(f"Error installing dependencies: {stderr}")
        return False

    print("Installing package in editable mode...")
    success, stdout, stderr = run_command(pip_cmd + ["-e", "."])
    if not success:
        print(f"Error installing package: {stderr}")
        return False

    if platform.system() == "Windows":
        python_path = str(Path.cwd() / "venv" / "Scripts" / "python.exe")
    else:
        python_path = str(Path.cwd() / "venv" / "bin" / "python")

    _ensure_importable(python_path)

    return True


def _ensure_importable(python_path: str) -> None:
    """Ensure easy_glm is importable.

    On some Python versions (notably 3.14), editable installs via ``.pth``
    files are silently broken.  As a fallback, symlink the source package
    directly into site-packages.
    """
    check = subprocess.run(
        [python_path, "-c", "import easy_glm"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return

    print("Editable install not importable — creating symlink fallback ...")
    site_packages = _get_site_packages(python_path)
    src_pkg = Path.cwd() / "src" / "easy_glm"
    dest = site_packages / "easy_glm"

    if platform.system() == "Windows":
        _symlink_windows(src_pkg, dest)
    else:
        dest.unlink(missing_ok=True)
        dest.symlink_to(src_pkg, target_is_directory=True)

    check2 = subprocess.run(
        [python_path, "-c", "import easy_glm"],
        capture_output=True,
        text=True,
    )
    if check2.returncode != 0:
        print("Warning: symlink didn't fix it — set PYTHONPATH=src manually")


def _symlink_windows(src: Path, dest: Path) -> None:
    try:
        os.symlink(src, dest, target_is_directory=True)
    except OSError:
        try:
            subprocess.run(
                ["mklink", "/J", str(dest), str(src)],
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            print(
                "Warning: Could not create symlink on Windows. "
                "You may need to run as Administrator or enable Developer Mode."
            )


def _get_site_packages(python_path: str) -> Path:
    result = subprocess.run(
        [python_path, "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def main():
    """Main setup function."""
    print("🚀 Setting up easy_glm development environment...")

    # Check if uv is installed
    if not check_uv_installed():
        print("❌ uv is not installed. Please install uv first:")
        print(
            "   For Unix/Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
        print(
            '   For Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
        )
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

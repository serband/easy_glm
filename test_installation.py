#!/usr/bin/env python3
"""
Test script to verify easy_glm installation and basic functionality.
"""

import subprocess
import sys


def test_installation():
    """Test that easy_glm can be installed and imported."""
    print("Testing easy_glm installation...")

    # Use the current Python interpreter (already in the venv if activated)
    venv_python = sys.executable

    try:
        # Test importing easy_glm
        result = subprocess.run(
            [
                venv_python,
                "-c",
                "import easy_glm; print('✅ easy_glm imported successfully')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)

        # Test loading external dataframe
        result = subprocess.run(
            [
                venv_python,
                "-c",
                "import easy_glm; df = easy_glm.load_external_dataframe(); print(f'✅ Loaded dataframe with {df.shape[0]} rows and {df.shape[1]} columns')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)

        print("✅ All tests passed!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        print(f"stderr: {e.stderr}")
        return False


if __name__ == "__main__":
    if test_installation():
        print("\n🎉 Installation verification completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Installation verification failed!")
        sys.exit(1)

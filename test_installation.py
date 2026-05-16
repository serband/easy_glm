#!/usr/bin/env python3
"""
Test script to verify easy_glm installation and basic functionality.
"""

import subprocess
import sys


def verify_installation():
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

        # Test core functionality without network access
        result = subprocess.run(
            [
                venv_python,
                "-c",
                (
                    "import easy_glm, polars as pl; "
                    "df = pl.DataFrame({'x': [1.0, 2.0], 'cat': ['A', 'B']}); "
                    "blueprint = easy_glm.generate_blueprint(df); "
                    "print(f'✅ Generated blueprint for {len(blueprint)} columns')"
                ),
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
    if verify_installation():
        print("\n🎉 Installation verification completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Installation verification failed!")
        sys.exit(1)

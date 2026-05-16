#!/usr/bin/env python3
"""
Verification script for easy_glm Git installation.
This script verifies that the package can be installed and used correctly
when installed directly from Git using: uv pip install git+https://github.com/serband/easy_glm.git
"""


def verify_installation():
    """Verify that easy_glm works correctly when installed from Git."""
    print("🔍 Verifying easy_glm Git installation...")

    try:
        # Test 1: Import the package
        print("1. Testing package import...")
        import easy_glm

        print("   ✅ Package imported successfully")

        # Test 2: Build a tiny synthetic dataframe without network access
        print("2. Testing synthetic data setup...")
        import polars as pl

        sample_df = pl.DataFrame(
            {
                "VehAge": [1.0, 3.0, 7.0, 10.0],
                "Region": ["North", "South", "North", "Urban"],
                "Exposure": [1.0, 0.5, 0.75, 1.0],
                "ClaimNb": [0, 1, 0, 2],
                "traintest": [1, 1, 0, 1],
            }
        )
        print("   ✅ Synthetic dataframe created")

        # Test 3: Test core functionality
        print("3. Testing core functionality...")
        # Generate blueprint
        blueprint = easy_glm.generate_blueprint(sample_df)
        print("   ✅ Blueprint generation successful")

        # Test 4: Prepare data
        predictors = ["VehAge", "Region"]
        _ = easy_glm.prepare_data(
            df=sample_df,
            modelling_variables=predictors,
            additional_columns=["Exposure", "ClaimNb", "traintest"],
            formats=blueprint,
            traintest_column="traintest",
            table_name="test_cars",
        )
        print("   ✅ Data preparation successful")

        print(
            "\n🎉 All tests passed! The package is working correctly when installed from Git."
        )
        return True

    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        return False


if __name__ == "__main__":
    if verify_installation():
        print("\n✅ Git installation verification completed successfully!")
    else:
        print("\n💥 Git installation verification failed!")
        exit(1)

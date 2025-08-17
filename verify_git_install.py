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
        
        # Test 2: Load external dataframe
        print("2. Testing data loading...")
        df = easy_glm.load_external_dataframe()
        print(f"   ✅ Loaded dataframe with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Test 3: Test core functionality
        print("3. Testing core functionality...")
        import polars as pl
        import numpy as np
        
        # Create a small sample for faster testing
        sample_df = df.head(1000)
        
        # Add train/test column
        sample_df = sample_df.with_columns(
            (pl.Series(np.random.rand(sample_df.height)) < 0.7)
            .cast(pl.Int8)
            .alias("traintest")
        )
        
        # Generate blueprint
        blueprint = easy_glm.generate_blueprint(sample_df)
        print("   ✅ Blueprint generation successful")
        
        # Test 4: Prepare data
        predictors = ["VehAge", "Region", "VehGas"]
        prepped = easy_glm.prepare_data(
            df=sample_df,
            modelling_variables=predictors,
            additional_columns=["Exposure", "ClaimNb", "traintest"],
            formats=blueprint,
            traintest_column="traintest",
            table_name="test_cars",
        )
        print("   ✅ Data preparation successful")
        
        print("\n🎉 All tests passed! The package is working correctly when installed from Git.")
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

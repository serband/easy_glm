from easy_glm.core.blueprint import generate_blueprint


def test_generate_blueprint_basic(synthetic_insurance_data):
    """
    Test that generate_blueprint returns expected keys and value types.
    """
    blueprint = generate_blueprint(synthetic_insurance_data)
    assert isinstance(blueprint, dict)
    # Check some expected columns
    for col in ["VehAge", "DrivAge", "Region"]:
        assert col in blueprint
        assert isinstance(blueprint[col], list)
        # Numeric columns should have floats/ints, categoricals should have strings
        if all(isinstance(x, float | int) for x in blueprint[col]):
            assert all(isinstance(x, float | int) for x in blueprint[col])
        else:
            assert all(isinstance(x, str) for x in blueprint[col])
    # No error messages for these columns
    for col in ["VehAge", "Region"]:
        assert not (
            isinstance(blueprint[col], str) and blueprint[col].startswith("Error:")
        )


if __name__ == "__main__":
    test_generate_blueprint_basic()
    print("generate_blueprint test passed.")

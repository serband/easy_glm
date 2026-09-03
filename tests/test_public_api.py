"""Small public-import contracts."""


def test_rate_model_is_exported_from_the_package_root():
    import easy_glm
    from easy_glm.engine.rate_model import RateModel

    assert easy_glm.RateModel is RateModel

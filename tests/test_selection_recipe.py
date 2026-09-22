"""A saved screen must not acquire new defaults when exported later."""

from copy import deepcopy

import pytest

from easy_glm.workflow.selection_recipe import resolved_selection_options


@pytest.mark.parametrize("family,link", [("gaussian", "log"), ("binomial", "logit")])
def test_legacy_missing_settings_keep_original_link_and_full_scoring(family, link):
    assert resolved_selection_options({"family": family}, {}, legacy=True) == {
        "family": family,
        "link": link,
        "importance_sample_pct": 100.0,
    }


def test_saved_resolved_results_preserve_explicit_gaussian_identity():
    assert (
        resolved_selection_options(
            {"family": "gaussian", "link": None}, {"link": "identity"}, legacy=True
        )["link"]
        == "identity"
    )


def test_new_recipe_pins_defaults_without_mutating_inputs():
    options = {"family": "gaussian", "link": None}
    result = {"link": "identity", "importance_sample_pct": 30.0}
    before = deepcopy((options, result))
    assert resolved_selection_options(options, result) == {
        "family": "gaussian",
        "link": "identity",
        "importance_sample_pct": 30.0,
    }
    assert (options, result) == before


def test_explicit_options_survive_changed_defaults_and_conflicting_result():
    options = {"family": "gaussian", "link": "log", "importance_sample_pct": 100.0}
    assert (
        resolved_selection_options(
            options, {"link": "identity", "importance_sample_pct": 30.0}, legacy=True
        )
        == options
    )

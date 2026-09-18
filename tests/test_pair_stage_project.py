from __future__ import annotations

from easy_glm.workflow.project import (
    Adjustment,
    Interaction,
    PairStageConfig,
    Project,
)


def project() -> Project:
    p = Project()
    p.data.roles = {
        "target": "target",
        "main": "predictor",
        "A": "predictor",
        "B": "predictor",
        "C": "predictor",
        "offset": "offset",
    }
    cfg = p.new_model("m")
    cfg.penalty.alpha = 0.1
    cfg.pair_stages = [PairStageConfig("s1", "A", "B")]
    return p


def test_pair_only_predictors_roundtrip_and_legacy_format() -> None:
    p = project()
    p.models["m"].predictors = ["main"]
    assert not p.validate("m")
    encoded = p.to_dict()
    assert encoded["version"] == 3
    reloaded = Project.from_dict(encoded)
    assert reloaded.models["m"].pair_stages[0].a == "A"
    assert len(reloaded.models["m"].pair_stages[0].candidates) == 2
    reloaded.models["m"].pair_stages.clear()
    assert reloaded.to_dict()["version"] == 3
    assert reloaded.models["m"].pair_method == "sequential_catboost"
    assert Project().to_dict()["version"] == 2


def test_pair_validation_rejects_reversed_duplicate_mixing_and_role() -> None:
    p = project()
    cfg = p.models["m"]
    cfg.pair_stages.append(PairStageConfig("s2", "B", "A"))
    cfg.interactions.append(Interaction("main", "C"))
    p.data.roles["A"] = "ignore"
    errors = p.validate("m")
    assert any("listed twice" in e for e in errors)
    assert any("cannot be mixed" in e for e in errors)
    assert any("must have predictor role" in e for e in errors)
    cfg.interactions.clear()
    cfg.pair_stages.pop()
    cfg.family = "gaussian"
    assert any("Poisson/log and Tweedie/log" in e for e in p.validate("m"))


def test_rename_and_role_removal_update_stages_and_adjustments() -> None:
    p = project()
    cfg = p.models["m"]
    cfg.adjustments.append(
        Adjustment("pair:s1", None, 10.0, 1.5, "R", "R", True, "s1", 0, 0)
    )
    p.rename_column("A", "A_new")
    assert cfg.pair_stages[0].a == "A_new"
    assert cfg.pair_stages[0].stage_id == "s1"
    p.apply_role_change("A_new", "ignore")
    assert cfg.pair_stages == []
    assert cfg.adjustments == []


def test_invalid_pair_adjustment_and_search_settings_report_errors() -> None:
    p = project()
    cfg = p.models["m"]
    cfg.pair_stages[0].candidates.clear()
    cfg.adjustments.append(
        Adjustment("pair:s1", None, None, 2, cell=True, stage_id="s1")
    )
    errors = p.validate("m")
    assert any("teacher candidate" in e for e in errors)
    assert any("canonical axis rows" in e for e in errors)

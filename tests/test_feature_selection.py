"""One-way shadow selection preserves the training and design contracts."""

from __future__ import annotations

import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from easy_glm.core.design import CategoricalEncoder, StepEncoder
from easy_glm.workflow import feature_selection as selection
from easy_glm.workflow.project import Derived, Project, VariableDesign


def _project() -> Project:
    project = Project()
    project.data.roles = {
        "y": "target",
        "w": "weight",
        "offset": "offset",
        "split": "split",
        "id": "id",
        "x": "predictor",
        "category": "predictor",
    }
    project.data.split.mode = "column"
    project.data.split.column = "split"
    project.data.split.train_value = 1
    project.design.variables["x"] = VariableDesign(
        knots=[0.5, 1.5],
        null_indicator=False,
        penalty_weight=2.0,
        monotone="increasing",
    )
    project.design.variables["category"] = VariableDesign(
        kind="categorical", levels=["a", "b"], penalty_weight=3.0
    )
    return project


def _frame() -> pl.DataFrame:
    n = 20
    return pl.DataFrame(
        {
            "x": [None if i % 7 == 0 else float(i % 3) for i in range(n)],
            "category": [None if i % 5 == 0 else "ab"[i % 2] for i in range(n)],
            "y": [float(i % 3) for i in range(n)],
            "w": [1.0 + (i % 2) for i in range(n)],
            "offset": [0.1] * n,
            "id": list(range(n)),
            "split": [1] * 16 + [0] * 4,
            "extra": list(range(n)),
        }
    )


def test_training_only_clones_and_operational_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, frame = _project(), _frame()
    project.data.roles["extra"] = "unassigned"
    before = pickle.dumps(project), frame.clone()
    seen: list[tuple[pl.DataFrame, object, dict[str, object]]] = []

    def fake_fit(data: pl.DataFrame, spec: object, target: str, **kwargs: object):
        assert data.height == 16
        assert target == "y"
        assert kwargs["weight_col"] == "w"
        assert kwargs["offset_col"] == "offset"
        assert kwargs["divide_target_by_weight"] is True
        assert kwargs["cv"] == 5 and kwargs["n_alphas"] == 6
        assert kwargs["sparse"] is True
        real = next(name for name in spec.main_effects if not name.startswith("__"))
        original = spec[real]
        for name in spec.main_effects:
            if name == real or name.startswith("__selection_random"):
                continue
            clone = spec[name]
            expected = original.to_dict()
            expected["variable"] = name
            assert clone.to_dict() == expected
        if real == "x":
            assert isinstance(original, StepEncoder)
            assert original.knots == [0.5, 1.5]
            assert original.penalty_weight == 2.0
            assert kwargs["monotone"] == dict.fromkeys(
                [real, *[n for n in spec.main_effects if "shadow" in n]],
                "increasing",
            )
        if real == "category":
            assert isinstance(original, CategoricalEncoder)
            assert original.levels == ["a", "b"]
            assert original.penalty_weight == 3.0
        seen.append((data, spec, kwargs))
        return SimpleNamespace(alpha=0.2)

    def fake_importance(fit: object, data: pl.DataFrame, **kwargs: object):
        assert data.height == 16
        assert set(kwargs["protected_columns"]) >= {"y", "w", "offset", "split", "id"}
        names = seen[-1][1].main_effects
        return pl.DataFrame(
            {
                "variable": list(names),
                "importance": [2.0] + [0.1] * (len(names) - 1),
                "std": [0.01] * len(names),
            }
        )

    monkeypatch.setattr(selection, "fit_glm", fake_fit)
    monkeypatch.setattr(selection, "permutation_importance", fake_importance)
    result = selection.select_variables(
        project, frame, divide_target_by_weight=True, n_alphas=6
    )
    assert result["training_rows"] == 16
    assert result["candidate_count"] == result["tested_count"] == 3
    assert {row["variable"] for row in result["rows"]} == {"x", "category", "extra"}
    assert all(row["status"] == "signal" for row in result["rows"])
    assert (
        next(row for row in result["rows"] if row["variable"] == "extra")["role"]
        == "unassigned"
    )
    assert pickle.dumps(project) == before[0]
    assert frame.equals(before[1])


def test_strict_threshold_and_failed_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, frame = _project(), _frame()
    project.data.roles["extra"] = "ignore"
    values = iter([0.0, -0.1])

    def fake_fit(data: pl.DataFrame, spec: object, target: str, **kwargs: object):
        if "category" in spec.main_effects:
            raise RuntimeError("candidate fit failed")
        return SimpleNamespace(alpha=0.1)

    def fake_importance(fit: object, data: pl.DataFrame, **kwargs: object):
        value = next(values)
        return pl.DataFrame(
            {
                "variable": ["x", *[c for c in data.columns if "__selection_" in c]],
                "importance": [value, 0.0, 0.0, 0.0, 0.0, 0.0],
                "std": [0.0] * 6,
            }
        )

    monkeypatch.setattr(selection, "fit_glm", fake_fit)
    monkeypatch.setattr(selection, "permutation_importance", fake_importance)
    first = selection.select_variables(project, frame)
    assert first["rows"][0]["status"] == "no_signal"
    assert first["rows"][1]["status"] == "failed"
    assert first["rows"][1]["importance"] is None
    second = selection.select_variables(project, frame)
    assert second["rows"][0]["importance"] == -0.1
    assert second["rows"][0]["threshold"] == 0.0


def test_global_invalid_input_and_cancellation() -> None:
    project, frame = _project(), _frame()
    with pytest.raises(ValueError, match="Weights"):
        selection.select_variables(project, frame.with_columns(pl.lit(0).alias("w")))
    with pytest.raises(ValueError, match="Target"):
        selection.select_variables(project, frame.with_columns(pl.lit(None).alias("y")))
    with pytest.raises(InterruptedError):
        selection.select_variables(project, frame, cancelled=lambda: True)


@pytest.mark.parametrize(
    "option",
    [
        {"family": "unknown"},
        {"family": "binomial", "link": "log"},
        {"link": "logit"},
        {"tweedie_power": 1.7},
        {"l1_ratio": 0},
        {"l1_ratio": float("nan")},
        {"n_alphas": 2.5},
        {"n_alphas": True},
        {"repeats": "5"},
        {"seed": -1},
        {"seed": 1.5},
    ],
)
def test_invalid_options_fail_whole_job(option: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        selection.select_variables(_project(), _frame(), **option)


def test_cancel_after_last_importance_does_not_return_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, frame = _project(), _frame()
    project.data.roles["category"] = "ignore"
    project.data.roles["extra"] = "ignore"
    state = {"cancelled": False}
    monkeypatch.setattr(
        selection, "fit_glm", lambda *args, **kwargs: SimpleNamespace(alpha=0.1)
    )

    def importance(*args: object, **kwargs: object) -> pl.DataFrame:
        state["cancelled"] = True
        return pl.DataFrame()

    monkeypatch.setattr(selection, "permutation_importance", importance)
    with pytest.raises(InterruptedError, match="cancelled"):
        selection.select_variables(project, frame, cancelled=lambda: state["cancelled"])


def test_convergence_warning_is_failed_not_no_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, frame = _project(), _frame()
    project.data.roles["category"] = "ignore"
    project.data.roles["extra"] = "ignore"

    def unconverged(*args: object, **kwargs: object) -> SimpleNamespace:
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
        return SimpleNamespace(alpha=0.1)

    monkeypatch.setattr(selection, "fit_glm", unconverged)
    result = selection.select_variables(project, frame)
    assert result["rows"][0]["status"] == "failed"
    assert "ConvergenceWarning" in result["rows"][0]["reason"]
    assert result["rows"][0]["monotone"] == "increasing"


def test_unusable_and_oversized_candidates_remain_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, frame = _project(), _frame()
    project.data.roles["extra"] = "ignore"
    frame = frame.with_columns(pl.lit(None).alias("x"))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("No fit should start")

    monkeypatch.setattr(selection, "fit_glm", forbidden)
    monkeypatch.setattr(selection, "_MAX_DESIGN_BYTES", 1)
    result = selection.select_variables(project, frame)
    by_name = {row["variable"]: row for row in result["rows"]}
    assert by_name["x"]["status"] == "skipped"
    assert by_name["category"]["status"] == "failed"
    assert "Compact design" in by_name["category"]["reason"]
    assert result["tested_count"] == 0


def test_design_column_limit_precedes_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    project, frame = _project(), _frame()
    project.data.roles["category"] = "ignore"
    project.data.roles["extra"] = "ignore"
    monkeypatch.setattr(selection, "_MAX_DESIGN_COLUMNS", 1)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Oversized design must not fit")

    monkeypatch.setattr(selection, "fit_glm", forbidden)
    row = selection.select_variables(project, frame)["rows"][0]
    assert row["status"] == "failed"
    assert "Design has" in row["reason"]
    assert row["monotone"] == "increasing"


def test_renamed_source_included_but_derived_only_columns_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, frame = _project(), _frame()
    project.data.roles["x"] = "ignore"
    project.data.roles["category"] = "ignore"
    project.data.renames["extra"] = "renamed_extra"
    project.data.roles["renamed_extra"] = "predictor"
    project.data.derived = [
        Derived("derived_predictor", "pl.col('renamed_extra') * 2"),
        Derived("derived_unassigned", "pl.col('renamed_extra') * 3"),
    ]
    project.data.roles["derived_predictor"] = "predictor"
    seen: list[str] = []

    def fake_fit(data: pl.DataFrame, spec: object, target: str, **kwargs: object):
        seen.append(next(iter(spec.main_effects)))
        return SimpleNamespace(alpha=0.1)

    def fake_importance(fit: object, data: pl.DataFrame, **kwargs: object):
        names = [seen[-1], *[name for name in data.columns if "__selection_" in name]]
        return pl.DataFrame(
            {"variable": names, "importance": [1.0] + [0.0] * 5, "std": [0.0] * 6}
        )

    monkeypatch.setattr(selection, "fit_glm", fake_fit)
    monkeypatch.setattr(selection, "permutation_importance", fake_importance)
    result = selection.select_variables(project, frame)
    assert seen == ["renamed_extra"]
    assert result["candidate_count"] == 1
    assert [row["variable"] for row in result["rows"]] == ["renamed_extra"]


def test_seed_does_not_depend_on_candidate_order() -> None:
    assert selection._seed(42, "x") == selection._seed(42, "x")
    assert selection._seed(42, "x") != selection._seed(42, "category")
    assert selection._seed(42, "x") != selection._seed(43, "x")


def test_real_signal_noise_and_holdout_invariance() -> None:
    rng = np.random.default_rng(2024)
    n = 220
    x = rng.integers(0, 3, n)
    noise = rng.integers(0, 3, n)
    y = rng.poisson(np.exp(-0.2 + x))
    frame = pl.DataFrame(
        {"x": x, "noise": noise, "y": y, "split": [1] * 180 + [0] * 40}
    )
    project = Project()
    project.data.roles = {
        "x": "predictor",
        "noise": "predictor",
        "y": "target",
        "split": "split",
    }
    project.data.split.mode = "column"
    project.data.split.column = "split"
    project.data.split.train_value = 1
    first = selection.select_variables(project, frame, n_alphas=5, repeats=2)
    changed_holdout = frame.with_columns(
        pl.when(pl.col("split") == 0)
        .then(pl.lit(9999))
        .otherwise(pl.col("y"))
        .alias("y"),
        pl.when(pl.col("split") == 0)
        .then(pl.lit(9999))
        .otherwise(pl.col("x"))
        .alias("x"),
    )
    second = selection.select_variables(project, changed_holdout, n_alphas=5, repeats=2)
    assert first["rows"] == second["rows"]
    by_name = {row["variable"]: row for row in first["rows"]}
    assert by_name["x"]["status"] == "signal"
    assert by_name["noise"]["status"] == "no_signal"


@pytest.mark.parametrize("family", ["gaussian", "gamma", "tweedie", "binomial"])
def test_supported_family_real_fit_smoke(family: str) -> None:
    rng = np.random.default_rng(441)
    n = 90
    x = rng.integers(0, 2, n)
    if family == "gaussian":
        y = 2.0 + 1.5 * x + rng.normal(0, 0.1, n)
    elif family == "gamma":
        y = np.exp(0.3 + 0.8 * x + rng.normal(0, 0.08, n))
    elif family == "tweedie":
        y = np.where(rng.random(n) < 0.15, 0.0, np.exp(0.3 + 0.8 * x))
    else:
        y = rng.binomial(1, np.where(x == 0, 0.15, 0.75))
    frame = pl.DataFrame({"x": x, "y": y, "split": [1] * n})
    project = Project()
    project.data.roles = {"x": "predictor", "y": "target", "split": "split"}
    project.data.split.mode = "column"
    project.data.split.column = "split"
    project.data.split.train_value = 1
    with threadpool_limits(limits=1):
        result = selection.select_variables(
            project, frame, family=family, n_alphas=3, repeats=1
        )
    assert result["candidate_count"] == result["tested_count"] == 1
    row = result["rows"][0]
    assert row["status"] in ("signal", "no_signal")
    assert row["importance"] is not None and np.isfinite(row["importance"])
    assert row["reason"] == ""

"""Piece G — the compact design matrix, and that it changes no number.

The whole point of piece G is that a 5M-row book fits in memory *without the
answers moving*. So almost every test here is an equivalence test: the compact
tabmat ``SplitMatrix`` against the dense float64 matrix it stands for, and a
fit on one against a fit on the other.

The acceptance rule, stated once
--------------------------------
The two representations are **not** bit-identical and are not meant to be: they
add the same numbers in a different order (``np.bincount`` over bin codes
versus a BLAS dot product), and the weighted column standard deviations that
set the per-band / per-cell ``P1`` come from tabmat in one case and from numpy
in the other. Both differences are at the last bit of a float64. So the rule
is:

1. predictions agree to **1e-10 relative**, and
2. the **non-zero set of the coefficients is identical**; if a coefficient ever
   differed it would have to be one sitting exactly on the lasso threshold, and
   the test would report which — no such difference has been seen (the largest
   observed coefficient difference on these designs is ~1e-14).

Cross-validation gets one extra allowance: the chosen alpha must be the same
grid point or one step away, because the two paths score the folds through the
same arithmetic and could in principle straddle a tie.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tabmat as tm

from easy_glm import DesignSpec, fit_glm, fit_two_stage, to_rate_model
from easy_glm.core.design import (
    SPARSE_ROW_THRESHOLD,
    _check_step_blocks_first,
    design_bytes,
    quantile_knots,
)
from easy_glm.core.fit import aggregate_rows, penalty_weights
from easy_glm.core.stepmatrix import StepMatrix, install_glum_shim

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "french_motor_50k.parquet"
#: predictions must match this closely between the two representations
PRED_RTOL = 1e-10
#: the matrix operations themselves are checked far tighter
OP_ATOL = 1e-12


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def book() -> pl.DataFrame:
    return pl.read_parquet(FIXTURE)


def synthetic_book(n: int, seed: int = 0) -> pl.DataFrame:
    """A small motor book: skewed numerics, a long tail of levels, ~1 % nulls
    in a numeric and a categorical, and an unseen level in the last rows."""
    rng = np.random.default_rng(seed)

    def levels(count: int, prefix: str) -> np.ndarray:
        weights = 1.0 / np.arange(1, count + 1) ** 1.2
        weights /= weights.sum()
        names = np.array([f"{prefix}{i:02d}" for i in range(count)])
        return names[rng.choice(count, n, p=weights)]

    driv_age = np.clip(rng.gamma(9.0, 4.5, n) + 18.0, 18, 95)
    veh_age = np.clip(rng.gamma(2.0, 3.0, n), 0, 40)
    bonus = np.clip(50 + rng.gamma(2.0, 12.0, n), 50, 230)
    density = np.exp(rng.normal(6.0, 1.6, n))
    exposure = np.clip(rng.beta(4.0, 2.0, n), 0.02, 1.0)
    log_mu = (
        -2.6
        + 0.012 * (bonus - 50)
        - 0.02 * np.clip(driv_age - 18, 0, 30)
        + 0.03 * np.clip(veh_age, 0, 12)
    )
    frame = pl.DataFrame(
        {
            "DrivAge": driv_age,
            "VehAge": veh_age,
            "BonusMalus": bonus,
            "Density": density,
            "Region": levels(12, "R"),
            "VehBrand": levels(8, "B"),
            "Exposure": exposure,
            "ClaimNb": rng.poisson(np.exp(log_mu) * exposure).astype(float),
            "logprem": np.log(np.clip(rng.gamma(5.0, 60.0, n), 40, 5_000)),
        }
    )
    null_numeric = rng.random(n) < 0.01
    null_category = rng.random(n) < 0.01
    return frame.with_columns(
        pl.when(pl.Series(null_numeric))
        .then(None)
        .otherwise(pl.col("VehAge"))
        .alias("VehAge"),
        pl.when(pl.Series(null_category))
        .then(None)
        .otherwise(pl.col("VehBrand"))
        .alias("VehBrand"),
    )


def scoring_frame(data: pl.DataFrame) -> pl.DataFrame:
    """Rows a scorer must survive: an unseen level, nulls, values past the
    training range in both directions."""
    out = data.tail(2_000)
    i = pl.arange(0, out.height)
    return out.with_columns(
        pl.when(i % 40 == 0)
        .then(pl.lit("UNSEEN"))
        .otherwise(pl.col("Region"))
        .alias("Region"),
        pl.when(i % 31 == 0)
        .then(pl.col("Density") * 100.0)
        .when(i % 31 == 1)
        .then(pl.lit(0.0))
        .otherwise(pl.col("Density"))
        .alias("Density"),
    )


# --------------------------------------------------------------------------
# 1. StepMatrix against the dense block it stands for
# --------------------------------------------------------------------------
class TestStepMatrix:
    """Every operation glum can ask for, against the dense matrix, to 1e-12."""

    @staticmethod
    def _pair(n_knots: int, n: int = 900, seed: int = 0):
        rng = np.random.default_rng(seed)
        code = rng.integers(0, n_knots + 2, n)  # includes the null code
        block = StepMatrix(code, n_knots, name="v")
        return block, block.toarray()

    @pytest.mark.parametrize("n_knots", [1, 3, 12])
    def test_matvec(self, n_knots):
        block, dense = self._pair(n_knots)
        beta = np.random.default_rng(1).normal(size=n_knots)
        np.testing.assert_allclose(block.matvec(beta), dense @ beta, atol=OP_ATOL)
        np.testing.assert_allclose(block @ beta, dense @ beta, atol=OP_ATOL)

    @pytest.mark.parametrize("n_knots", [3, 12])
    def test_matvec_with_column_subset(self, n_knots):
        block, dense = self._pair(n_knots)
        rng = np.random.default_rng(2)
        beta = rng.normal(size=n_knots)
        cols = np.sort(rng.choice(n_knots, n_knots // 2 + 1, replace=False)).astype(
            np.int32
        )
        np.testing.assert_allclose(
            block.matvec(beta, cols), dense[:, cols] @ beta[cols], atol=OP_ATOL
        )
        out = np.ones(dense.shape[0])
        block.matvec(beta, cols, out=out)
        np.testing.assert_allclose(out, 1.0 + dense[:, cols] @ beta[cols], atol=OP_ATOL)

    @pytest.mark.parametrize("n_knots", [1, 3, 12])
    def test_transpose_matvec_with_row_and_column_subsets(self, n_knots):
        block, dense = self._pair(n_knots)
        rng = np.random.default_rng(3)
        vec = rng.normal(size=dense.shape[0])
        np.testing.assert_allclose(
            block.transpose_matvec(vec), dense.T @ vec, atol=OP_ATOL
        )
        rows = np.sort(rng.choice(dense.shape[0], 300, replace=False)).astype(np.int32)
        cols = np.sort(rng.choice(n_knots, max(1, n_knots // 2), replace=False)).astype(
            np.int32
        )
        np.testing.assert_allclose(
            block.transpose_matvec(vec, rows, cols),
            dense[np.ix_(rows, cols)].T @ vec[rows],
            atol=OP_ATOL,
        )
        out = np.ones(n_knots)
        block.transpose_matvec(vec, rows, cols, out=out)
        expected = np.ones(n_knots)
        expected[cols] += dense[np.ix_(rows, cols)].T @ vec[rows]
        np.testing.assert_allclose(out, expected, atol=OP_ATOL)

    @pytest.mark.parametrize("n_knots", [1, 3, 12])
    def test_sandwich_with_row_and_column_subsets(self, n_knots):
        block, dense = self._pair(n_knots)
        rng = np.random.default_rng(4)
        d = rng.random(dense.shape[0])
        np.testing.assert_allclose(
            block.sandwich(d), dense.T @ (d[:, None] * dense), atol=OP_ATOL
        )
        rows = np.sort(rng.choice(dense.shape[0], 400, replace=False)).astype(np.int32)
        cols = np.sort(rng.choice(n_knots, max(1, n_knots // 2), replace=False)).astype(
            np.int32
        )
        sub = dense[np.ix_(rows, cols)]
        np.testing.assert_allclose(
            block.sandwich(d, rows, cols),
            sub.T @ (d[rows][:, None] * sub),
            atol=OP_ATOL,
        )

    def test_cross_sandwich_with_every_block_type(self):
        rng = np.random.default_rng(5)
        n = 900
        block, dense = self._pair(9, n=n)
        other_step = StepMatrix(rng.integers(0, 6, n), 4, name="w")
        categorical = tm.CategoricalMatrix(
            rng.integers(0, 5, n), categories=np.arange(5), drop_first=True
        )
        plain = tm.DenseMatrix(rng.normal(size=(n, 4)))
        d = rng.random(n)
        for other in (other_step, categorical, plain):
            right = other.toarray()
            np.testing.assert_allclose(
                block._cross_sandwich(other, d),
                dense.T @ (d[:, None] * right),
                atol=OP_ATOL,
                err_msg=type(other).__name__,
            )
            rows = np.sort(rng.choice(n, 500, replace=False)).astype(np.int32)
            left_cols = np.array([0, 3, 8], dtype=np.int32)
            right_cols = np.array([0, right.shape[1] - 1], dtype=np.int32)
            np.testing.assert_allclose(
                block._cross_sandwich(other, d, rows, left_cols, right_cols),
                dense[np.ix_(rows, left_cols)].T
                @ (d[rows][:, None] * right[np.ix_(rows, right_cols)]),
                atol=OP_ATOL,
                err_msg=f"{type(other).__name__} with subsets",
            )

    def test_cross_sandwich_refuses_an_unknown_block(self):
        block, _ = self._pair(3, n=50)
        with pytest.raises(TypeError, match="cross-sandwich"):
            block._cross_sandwich(
                tm.SparseMatrix(np.eye(50)), np.ones(50)
            )  # type: ignore[arg-type]

    def test_column_statistics_and_standardize(self):
        block, dense = self._pair(7)
        rng = np.random.default_rng(6)
        w = rng.random(dense.shape[0])
        w /= w.sum()
        means = block.transpose_matvec(w)
        np.testing.assert_allclose(means, w @ dense, atol=OP_ATOL)
        np.testing.assert_allclose(
            block._get_col_stds(w, means),
            np.sqrt(w @ (dense * dense) - (w @ dense) ** 2),
            atol=OP_ATOL,
        )
        standardized, out_means, out_stds = block.standardize(w, True, True)
        reference, ref_means, ref_stds = tm.DenseMatrix(dense).standardize(
            w, True, True
        )
        np.testing.assert_allclose(out_means, ref_means, atol=OP_ATOL)
        np.testing.assert_allclose(out_stds, ref_stds, atol=OP_ATOL)
        np.testing.assert_allclose(
            standardized.toarray(), reference.toarray(), atol=OP_ATOL
        )

    def test_row_subset_keeps_the_codes_and_the_values(self):
        block, dense = self._pair(6)
        rows = np.array([3, 17, 900 - 1, 42])
        sub = block[rows, :]
        assert isinstance(sub, StepMatrix)
        assert sub.nbytes == 4 * len(rows)  # 4 bytes a row, not 8 a column
        np.testing.assert_array_equal(sub.toarray(), dense[rows])

    def test_getcol_and_names(self):
        block, dense = self._pair(4)
        for j in range(4):
            np.testing.assert_array_equal(
                np.asarray(block.getcol(j).toarray()).ravel(), dense[:, j]
            )
        assert len(block.get_names("column")) == 4
        assert block.get_names("term") == ["v"] * 4
        block.set_names(["a", "b", "c", "d"])
        assert block.get_names("column") == ["a", "b", "c", "d"]

    def test_refuses_impossible_codes(self):
        with pytest.raises(ValueError, match="must lie in"):
            StepMatrix(np.array([0, 1, 9]), 3)
        with pytest.raises(ValueError, match="at least one knot"):
            StepMatrix(np.array([0, 1]), 0)


# --------------------------------------------------------------------------
# 2. the built matrix
# --------------------------------------------------------------------------
def full_spec(data: pl.DataFrame) -> DesignSpec:
    """A design using every encoder kind at once."""
    return DesignSpec.from_data(
        data,
        ["DrivAge", "VehAge", "BonusMalus", "Density", "Region", "VehBrand"],
        knots={
            "DrivAge": quantile_knots(data["DrivAge"], 10),
            "VehAge": quantile_knots(data["VehAge"], 8),
            "BonusMalus": quantile_knots(data["BonusMalus"], 8),
            "Density": quantile_knots(data["Density"], 4),
        },
        linear=["Density"],
        min_level_share=0.005,
        interactions=[("DrivAge", "Region")],
        min_cell_exposure=0.004,
    )


class TestBuild:
    def test_the_two_representations_are_the_same_matrix(self, book):
        spec = full_spec(book)
        dense = spec.build(book, sparse=False)
        compact = spec.build(book, sparse=True)
        assert compact.shape == dense.shape == (book.height, spec.n_features)
        assert compact.dtype == np.float64
        # exactly equal, not merely close: both are built from the same codes
        np.testing.assert_array_equal(compact.toarray(), dense)

    def test_a_continuous_term_and_a_categorical_only_design_also_match(self, book):
        for spec in (
            DesignSpec.from_data(
                book, ["Density"], linear=["Density"], knots={"Density": []}
            ),
            DesignSpec.from_data(book, ["Region", "VehBrand"]),
            DesignSpec.from_data(book, ["DrivAge"], null_indicator=False),
        ):
            np.testing.assert_array_equal(
                spec.build(book, sparse=True).toarray(), spec.build(book, sparse=False)
            )

    def test_unseen_levels_and_nulls_land_in_the_same_columns(self):
        data = synthetic_book(4_000, seed=7)
        spec = full_spec(data)
        score = scoring_frame(data)
        assert (score["Region"] == "UNSEEN").sum() > 0
        assert score["VehAge"].null_count() > 0
        np.testing.assert_array_equal(
            spec.build(score, sparse=True).toarray(), spec.build(score, sparse=False)
        )

    def test_block_order_puts_the_step_blocks_first(self, book):
        compact = full_spec(book).build(book, sparse=True)
        kinds = [type(m).__name__ for m in compact.matrices]
        last_step = max(i for i, k in enumerate(kinds) if k == "StepMatrix")
        assert all(k == "StepMatrix" for k in kinds[: last_step + 1]), kinds
        assert "DenseMatrix" in kinds and "CategoricalMatrix" in kinds

    def test_the_block_order_rule_is_enforced_not_assumed(self):
        rng = np.random.default_rng(0)
        step = StepMatrix(rng.integers(0, 4, 100), 2, name="v")
        plain = tm.DenseMatrix(rng.normal(size=(100, 2)))
        bad = tm.SplitMatrix([plain, step], [np.arange(0, 2), np.arange(2, 4)])
        with pytest.raises(RuntimeError, match="must come before"):
            _check_step_blocks_first(bad)

    def test_design_bytes_match_the_published_formula(self, book):
        spec = full_spec(book)
        compact = spec.build(book, sparse=True)
        assert design_bytes(compact) == spec.expected_design_bytes(book.height)
        # and the compact matrix is a small fraction of the dense one
        assert design_bytes(compact) * 5 < design_bytes(spec.build(book, sparse=False))

    def test_the_default_is_decided_by_row_count(self):
        wide = pl.DataFrame({"x": np.arange(SPARSE_ROW_THRESHOLD, dtype=float)})
        spec = DesignSpec.from_data(wide, ["x"], knots={"x": [10.0, 20.0, 30.0]})
        assert isinstance(spec.build(wide.head(SPARSE_ROW_THRESHOLD - 1)), np.ndarray)
        assert isinstance(spec.build(wide), tm.SplitMatrix)

    def test_the_golden_fixture_stays_on_the_dense_path(self, book):
        """The golden numbers are recorded from the dense fit, and a 50,000-row
        book is below the threshold — so the default path for it must not have
        moved. If this fails the golden test is being asked a new question."""
        assert book.height < SPARSE_ROW_THRESHOLD
        assert isinstance(full_spec(book).build(book), np.ndarray)


# --------------------------------------------------------------------------
# 3. the fit is the same fit
# --------------------------------------------------------------------------
def assert_same_fit(dense_fit, sparse_fit, score: pl.DataFrame, *, label: str) -> None:
    """The acceptance rule of the module docstring, in one place."""
    zero_dense, zero_sparse = dense_fit.coef == 0, sparse_fit.coef == 0
    if not np.array_equal(zero_dense, zero_sparse):
        differing = np.flatnonzero(zero_dense != zero_sparse)
        names = [dense_fit.spec.feature_names[i] for i in differing]
        sizes = [
            (float(dense_fit.coef[i]), float(sparse_fit.coef[i])) for i in differing
        ]
        raise AssertionError(
            f"{label}: the non-zero set differs on {names} with coefficients "
            f"{sizes}. Under the acceptance rule such a difference is allowed "
            "only for a coefficient on the lasso threshold (|value| < 1e-8) and "
            "must be listed here."
        )
    np.testing.assert_allclose(
        sparse_fit.predict(score),
        dense_fit.predict(score),
        rtol=PRED_RTOL,
        atol=0,
        err_msg=label,
    )


CASES: dict[str, dict] = {
    "steps_and_categoricals": {},
    "with_offset": {"offset_col": "logprem"},
    "monotone": {"monotone": {"BonusMalus": "increasing"}},
    "cross_validated": {"alpha": None, "cv": 3, "n_alphas": 6},
}


class TestFitEquivalence:
    """Same predictions and the same non-zero set, design after design."""

    @staticmethod
    def _fits(data, spec, *, two_stage=False, **kwargs):
        common = {
            "family": "poisson",
            "weight_col": "Exposure",
            "divide_target_by_weight": True,
            "alpha": 0.0005,
        }
        common.update(kwargs)
        common = {k: v for k, v in common.items() if v is not None or k != "alpha"}
        fit = fit_two_stage if two_stage else fit_glm
        return (
            fit(data, spec, "ClaimNb", sparse=False, **common),
            fit(data, spec, "ClaimNb", sparse=True, **common),
        )

    @pytest.mark.parametrize("case", list(CASES))
    def test_on_the_fifty_thousand_row_fixture(self, book, case):
        spec = DesignSpec.from_data(
            book,
            ["DrivAge", "VehAge", "BonusMalus", "Region", "VehGas"],
            knots={
                "DrivAge": quantile_knots(book["DrivAge"], 10),
                "VehAge": quantile_knots(book["VehAge"], 8),
                "BonusMalus": quantile_knots(book["BonusMalus"], 8),
            },
        )
        data = book.with_columns(pl.col("Exposure").log().alias("logprem"))
        dense, compact = self._fits(data, spec, **CASES[case])
        assert (dense.coef != 0).sum() > 5, "the case must actually fit something"
        assert_same_fit(dense, compact, data.tail(3_000), label=case)
        if case == "cross_validated":
            assert dense.alpha == pytest.approx(compact.alpha, rel=1e-8)

    def test_a_design_with_every_encoder_kind_fitted_in_two_stages(self, book):
        spec = full_spec(book)
        dense, compact = self._fits(book, spec, two_stage=True)
        cells = [
            c
            for f, c in zip(dense.spec.features, dense.coef, strict=True)
            if f.kind == "cell"
        ]
        assert cells and any(c != 0 for c in cells), "no interaction cell was fitted"
        assert_same_fit(dense, compact, scoring_frame(book), label="two_stage")
        np.testing.assert_allclose(
            compact.stage1.coef, dense.stage1.coef, rtol=0, atol=1e-10
        )

    def test_a_continuous_term_beside_a_step_term(self, book):
        spec = DesignSpec.from_data(
            book,
            ["DrivAge", "Density", "BonusMalus"],
            knots={
                "DrivAge": quantile_knots(book["DrivAge"], 10),
                "Density": [],
                "BonusMalus": quantile_knots(book["BonusMalus"], 6),
            },
            linear=["Density"],
        )
        assert spec["Density"].n_bands == 1  # the "continuous" kind
        dense, compact = self._fits(book, spec)
        assert_same_fit(dense, compact, scoring_frame(book), label="continuous")

    def test_on_a_three_hundred_thousand_row_book(self):
        """Above the row threshold, where the compact matrix is the default —
        with nulls, an unseen level and values beyond the training range."""
        data = synthetic_book(300_000, seed=11)
        spec = DesignSpec.from_data(
            data,
            ["DrivAge", "VehAge", "BonusMalus", "Density", "Region", "VehBrand"],
            knots={
                "DrivAge": quantile_knots(data["DrivAge"], 12),
                "VehAge": quantile_knots(data["VehAge"], 8),
                "BonusMalus": quantile_knots(data["BonusMalus"], 10),
                "Density": quantile_knots(data["Density"], 4),
            },
            linear=["Density"],
            interactions=[("DrivAge", "Region")],
            min_cell_exposure=0.004,
        )
        assert data.height >= SPARSE_ROW_THRESHOLD
        assert isinstance(spec.build(data), tm.SplitMatrix)  # the default path
        dense, compact = self._fits(data, spec, two_stage=True)
        assert_same_fit(dense, compact, scoring_frame(data), label="300k two-stage")

    def test_the_rate_model_still_reproduces_the_compact_fit(self, book):
        spec = full_spec(book)
        _, compact = self._fits(book, spec, two_stage=True)
        model = to_rate_model(compact, exposure_col="Exposure")
        score = scoring_frame(book)
        np.testing.assert_allclose(
            model.predict(score, exposure_col=None),
            compact.predict(score),
            rtol=1e-10,
            atol=0,
        )

    def test_penalty_weights_agree_between_the_representations(self, book):
        spec = full_spec(book)
        w = book["Exposure"].to_numpy()
        dense = penalty_weights(
            spec, spec.build(book, sparse=False), w, scale_predictors=True
        )
        compact = penalty_weights(
            spec, spec.build(book, sparse=True), w, scale_predictors=True
        )
        np.testing.assert_allclose(compact, dense, rtol=1e-12, atol=0)

    def test_glum_accepts_the_compact_matrix_without_densifying_it(self, book):
        """The shim is a private-API patch; if glum ever stops taking our block
        it will silently go through sklearn's ``check_array`` and densify. A
        two-block fit that keeps its blocks is the canary."""
        install_glum_shim()
        import glum._validation as validation

        spec = full_spec(book)
        compact = spec.build(book, sparse=True)
        checked = validation.check_array_tabmat_compliant(compact)
        assert isinstance(checked, tm.SplitMatrix)
        assert [type(m).__name__ for m in checked.matrices] == [
            type(m).__name__ for m in compact.matrices
        ]
        copied = validation.check_array_tabmat_compliant(compact, copy=True)
        assert isinstance(copied.matrices[0], StepMatrix)
        assert copied.matrices[0].code is not compact.matrices[0].code


# --------------------------------------------------------------------------
# 4. scoring never builds a matrix
# --------------------------------------------------------------------------
class TestScoringWithoutADesignMatrix:
    def test_predictions_match_glums_own_and_are_chunked(self, book):
        spec = full_spec(book)
        fit = fit_glm(
            book,
            spec.main_effects_spec(),
            "ClaimNb",
            family="poisson",
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.0005,
        )
        score = scoring_frame(book)
        through_glum = np.asarray(
            fit.model.predict(fit.spec.build(score, sparse=False)), dtype=float
        )
        np.testing.assert_allclose(fit.predict(score), through_glum, rtol=1e-12, atol=0)
        # the chunk size is an implementation detail, not an answer
        np.testing.assert_array_equal(
            fit.linear_predictor(score, chunk_rows=97),
            fit.linear_predictor(score, chunk_rows=10_000_000),
        )

    def test_predict_does_not_call_build(self, book, monkeypatch):
        spec = full_spec(book)
        fit = fit_two_stage(
            book,
            spec,
            "ClaimNb",
            family="poisson",
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.0005,
        )
        expected = fit.predict(book.head(500))

        def explode(*args, **kwargs):  # pragma: no cover - only if scoring regresses
            raise AssertionError("scoring must not build a design matrix")

        monkeypatch.setattr(DesignSpec, "build", explode)
        monkeypatch.setattr(DesignSpec, "build_dense", explode)
        monkeypatch.setattr(DesignSpec, "build_sparse", explode)
        np.testing.assert_array_equal(fit.predict(book.head(500)), expected)

    def test_contributions_equal_the_columns_they_stand_for(self, book):
        spec = full_spec(book)
        rng = np.random.default_rng(0)
        coef = rng.normal(size=spec.n_features)
        design = spec.build(book, sparse=False)
        np.testing.assert_allclose(
            spec.linear_predictor(book, coef, 1.25),
            1.25 + design @ coef,
            rtol=1e-12,
            atol=1e-12,
        )


# --------------------------------------------------------------------------
# 5. aggregation by identical design row
# --------------------------------------------------------------------------
class TestAggregation:
    @pytest.mark.parametrize("family", ["poisson", "gamma", "tweedie"])
    def test_aggregated_fit_equals_the_row_level_fit(self, book, family):
        data = book.with_columns(
            (pl.col("ClaimNb") + 0.5).alias("Cost")  # strictly positive for gamma
        )
        spec = DesignSpec.from_data(
            data,
            ["DrivAge", "VehGas", "Region"],
            knots={"DrivAge": [25.0, 40.0, 60.0]},
        )
        common = {
            "family": family,
            "weight_col": "Exposure",
            "divide_target_by_weight": True,
            "alpha": 0.001,
        }
        rows = fit_glm(data, spec, "Cost", **common)
        groups = fit_glm(data, spec, "Cost", aggregate=True, **common)
        np.testing.assert_allclose(groups.coef, rows.coef, rtol=0, atol=1e-12)
        assert groups.intercept == pytest.approx(rows.intercept, abs=1e-12)
        np.testing.assert_allclose(
            groups.predict(data), rows.predict(data), rtol=1e-12, atol=0
        )
        assert groups.n_train_rows == data.height
        assert groups.modal_bins == rows.modal_bins

    def test_aggregation_keeps_the_offset_in_the_key(self, book):
        data = book.with_columns(pl.col("Exposure").log().alias("logprem"))
        spec = DesignSpec.from_data(
            data, ["DrivAge", "Region"], knots={"DrivAge": [25.0, 45.0]}
        )
        common = {
            "family": "poisson",
            "weight_col": "Exposure",
            "divide_target_by_weight": True,
            "offset_col": "logprem",
            "alpha": 0.001,
        }
        rows = fit_glm(data, spec, "ClaimNb", **common)
        groups = fit_glm(data, spec, "ClaimNb", aggregate=True, **common)
        np.testing.assert_allclose(groups.coef, rows.coef, rtol=0, atol=1e-12)

    def test_a_linear_term_keeps_its_value_in_the_key(self, book):
        spec = DesignSpec.from_data(
            book, ["Density", "Region"], linear=["Density"], knots={"Density": []}
        )
        common = {
            "family": "poisson",
            "weight_col": "Exposure",
            "divide_target_by_weight": True,
            "alpha": 0.001,
        }
        rows = fit_glm(book, spec, "ClaimNb", **common)
        groups = fit_glm(book, spec, "ClaimNb", aggregate=True, **common)
        np.testing.assert_allclose(groups.coef, rows.coef, rtol=0, atol=1e-12)

    def test_grouping_totals_are_preserved(self, book):
        spec = DesignSpec.from_data(
            book, ["DrivAge", "VehGas"], knots={"DrivAge": [30.0, 50.0]}
        )
        y = book["ClaimNb"].to_numpy().astype(float)
        w = book["Exposure"].to_numpy()
        rows, y_bar, weight_sum, offset = aggregate_rows(spec, book, y, w, None)
        assert offset is None
        assert rows.height < book.height
        assert weight_sum.sum() == pytest.approx(w.sum(), rel=1e-12)
        assert (y_bar * weight_sum).sum() == pytest.approx((y * w).sum(), rel=1e-12)

    def test_aggregation_refuses_cross_validation(self, book):
        spec = DesignSpec.from_data(book, ["VehGas"])
        with pytest.raises(ValueError, match="cannot be combined with cv"):
            fit_glm(book, spec, "ClaimNb", cv=3, aggregate=True, family="poisson")


# --------------------------------------------------------------------------
# 6. progress
# --------------------------------------------------------------------------
class TestProgress:
    def test_the_callback_is_told_what_is_happening(self, book):
        seen: list[str] = []
        spec = DesignSpec.from_data(
            book, ["DrivAge", "Region"], knots={"DrivAge": [30.0, 50.0]}
        )
        fit_glm(
            book,
            spec,
            "ClaimNb",
            family="poisson",
            alpha=0.001,
            progress=seen.append,
        )
        assert seen, "no progress was reported"
        assert all("rows x" in message and "s" in message for message in seen)

    def test_both_stages_are_named(self, book):
        seen: list[str] = []
        spec = full_spec(book)
        fit_two_stage(
            book,
            spec,
            "ClaimNb",
            family="poisson",
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.0005,
            progress=seen.append,
        )
        assert any(m.startswith("Stage 1, main effects") for m in seen)
        assert any(m.startswith("Stage 2, interaction cells") for m in seen)

    def test_a_failing_callback_cannot_fail_a_fit(self, book):
        def broken(message: str) -> None:
            raise RuntimeError("the display fell over")

        spec = DesignSpec.from_data(book, ["VehGas"])
        fit = fit_glm(
            book, spec, "ClaimNb", family="poisson", alpha=0.001, progress=broken
        )
        assert np.isfinite(fit.intercept)


# --------------------------------------------------------------------------
# 7. the 5M-row budget (slow: run with `pytest -m slow`)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_five_million_rows_fit_inside_three_gigabytes():
    """The plan's headline promise, measured rather than argued.

    Runs ``scripts/bench_scale.py`` in its own process (the benchmark spawns a
    further subprocess per point) so the peak resident memory measured belongs
    to that fit and nothing else. The script fails if any run's peak is above
    the budget or if the design bytes miss the published formula.

    Needs about 3 GB free and takes a couple of minutes; it is deselected by
    the default ``addopts`` and runs only under ``pytest -m slow``.
    """
    budget = 3 * 1024**3  # scripts/bench_scale.py::BUDGET_5M_BYTES

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bench_scale.py"),
            "--sizes",
            "5000000",
            "--representations",
            "sparse",
            "--check-budget",
            str(budget),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "5,000,000" in completed.stdout


# --------------------------------------------------------------------------
# 8. everything piece A2 settled, on the compact path
# --------------------------------------------------------------------------
def two_stage_spec(data: pl.DataFrame) -> DesignSpec:
    """Mains plus one interaction, small enough to fit four times in a test."""
    return DesignSpec.from_data(
        data,
        ["DrivAge", "VehAge", "BonusMalus", "Region", "VehGas"],
        knots={
            "DrivAge": quantile_knots(data["DrivAge"], 8),
            "VehAge": quantile_knots(data["VehAge"], 6),
            "BonusMalus": quantile_knots(data["BonusMalus"], 6),
        },
        min_level_share=0.01,
        interactions=[("DrivAge", "Region")],
        min_cell_exposure=0.004,
    )


class TestTwoStageBehaviourOnTheCompactPath:
    """Piece A2's final behaviour was settled against the dense matrix. Each of
    its three load-bearing routes — an offset passed as an array, an
    :class:`EasyGLM` bundle holding both stages, and the exported script — is
    re-checked here with the fit forced through the ``SplitMatrix``."""

    FIT_KW = {
        "family": "poisson",
        "weight_col": "Exposure",
        "divide_target_by_weight": True,
    }

    def test_an_offset_array_reaches_both_stages(self, book):
        """A2 folds an ``offset=`` array into ``eta1`` by hand, because
        ``linear_predictor`` adds neither form of offset. That hand-folding runs
        on whatever the design is, so it has to be re-checked here: if the
        compact path ever changed what ``linear_predictor`` returns, stage 2
        would silently absorb the whole offset."""
        frame = book.with_columns(
            pl.Series("logprem", np.log(np.linspace(150.0, 900.0, book.height)))
        )
        spec = two_stage_spec(frame)
        array = frame["logprem"].to_numpy()
        by_column = fit_two_stage(
            frame,
            spec,
            "ClaimNb",
            alpha=0.001,
            offset_col="logprem",
            sparse=True,
            **self.FIT_KW,
        )
        by_array = fit_two_stage(
            frame,
            spec,
            "ClaimNb",
            alpha=0.001,
            offset=array,
            sparse=True,
            **self.FIT_KW,
        )
        np.testing.assert_allclose(
            by_array.stage1.coef, by_column.stage1.coef, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            by_array.stage2.coef, by_column.stage2.coef, rtol=1e-10, atol=1e-12
        )
        assert by_array.offset_col is None and by_column.offset_col == "logprem"
        # and the array route itself is the same model on either representation
        dense = fit_two_stage(
            frame,
            spec,
            "ClaimNb",
            alpha=0.001,
            offset=array,
            sparse=False,
            **self.FIT_KW,
        )
        score = scoring_frame(frame)
        offset = scoring_frame(frame)["logprem"].to_numpy()
        np.testing.assert_allclose(
            by_array.predict(score, offset=offset),
            dense.predict(score, offset=offset),
            rtol=PRED_RTOL,
            atol=0,
        )
        assert np.array_equal(by_array.coef == 0, dense.coef == 0)

    def test_easyglm_saves_and_loads_a_compact_two_stage_fit(self, book, tmp_path):
        """The v3 bundle writes both glum estimators and rebuilds the pair. The
        fit it was rebuilt from was made on the compact design; the rebuilt one
        must score identically, and it scores from the codes either way."""
        from easy_glm import EasyGLM, TwoStageFit, rate_tables

        spec = two_stage_spec(book)
        fit = fit_two_stage(
            book, spec, "ClaimNb", alpha=0.001, sparse=True, **self.FIT_KW
        )
        assert isinstance(fit, TwoStageFit)
        bundle = EasyGLM(
            fit, to_rate_model(fit, exposure_col="Exposure"), rate_tables(fit)
        )
        bundle.save(tmp_path / "compact_two_stage")
        assert (tmp_path / "compact_two_stage" / "glm_model_stage2.joblib").exists()

        loaded = EasyGLM.load(tmp_path / "compact_two_stage")
        assert isinstance(loaded.glm, TwoStageFit)
        np.testing.assert_array_equal(loaded.glm.coef, fit.coef)
        score = scoring_frame(book)
        np.testing.assert_array_equal(
            loaded.predict(score).to_numpy(), bundle.predict(score).to_numpy()
        )
        np.testing.assert_allclose(
            loaded.rate_model.predict(score, exposure_col=None),
            loaded.glm.predict(score),
            rtol=1e-10,
            atol=0,
        )

    def test_the_exported_script_reproduces_a_compact_run(
        self, book, tmp_path, monkeypatch
    ):
        """The workbench run goes through the ``SplitMatrix``; the script it
        exports does not say ``sparse=`` at all, so it takes whatever the row
        count gives it — the dense matrix, at 50,000 rows. The script is only
        honest if the two agree, so this runs it and compares.

        It also pins the A2 decision that the script's stages come from
        ``isinstance(run.fit, TwoStageFit)``: a run fitted the compact way is
        still a ``TwoStageFit`` and must still export two stages."""
        import easy_glm.core.design as design_module
        from easy_glm.core.fit import TwoStageFit
        from easy_glm.workflow import (
            Interaction,
            Project,
            prepare,
            run_model,
            to_script,
        )

        # force every build in this test through the compact path
        monkeypatch.setattr(design_module, "SPARSE_ROW_THRESHOLD", 1_000)
        built_sparse: list[int] = []
        original = DesignSpec.build_sparse
        monkeypatch.setattr(
            DesignSpec,
            "build_sparse",
            lambda self, data: (built_sparse.append(data.height), original(self, data))[
                1
            ],
        )

        project = Project(name="scale")
        project.data.source.type = "parquet"
        project.data.source.path = str(FIXTURE)
        project.data.roles = {
            "ClaimNb": "target",
            "Exposure": "weight",
            "IDpol": "id",
            "DrivAge": "predictor",
            "BonusMalus": "predictor",
            "Region": "predictor",
            "VehGas": "predictor",
        }
        project.data.split.mode = "random"
        project.data.split.column = "traintest"
        project.data.split.fraction = 0.7
        project.data.split.seed = 3
        project.new_model(
            "freq",
            family="poisson",
            divide_target_by_weight=True,
            predictors=["DrivAge", "BonusMalus", "Region", "VehGas"],
        )
        project.models["freq"].penalty.alpha = 0.002
        project.models["freq"].penalty.cv = None
        project.models["freq"].interactions = [
            Interaction("DrivAge", "Region", min_cell_exposure=0.01)
        ]

        data = prepare(project)
        run = run_model(project, data, "freq")
        assert built_sparse, "the run did not go through the compact design"
        assert isinstance(run.fit, TwoStageFit)

        source = to_script(project, "freq", run=run, output_prefix="compact_v1")
        assert "fit = TwoStageFit(stage1, stage2)" in source
        assert "sparse" not in source  # the script takes the row-count default
        script = tmp_path / "rebuild.py"
        script.write_text(source)
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]

        from easy_glm.engine import RateModel

        rebuilt = RateModel.from_json(tmp_path / "compact_v1.easyglm")
        holdout = data.filter(pl.col("traintest") == 0)
        np.testing.assert_allclose(
            rebuilt.predict(holdout, exposure_col=None),
            run.predict(holdout),
            rtol=PRED_RTOL,
            atol=0,
        )

    def test_stage_one_p1_does_not_reach_the_cells(self, book):
        """A2 stopped stage 1's ``P1`` (one weight per main-effect column) from
        being forwarded to stage 2, whose design has different columns
        altogether. The compact path computes ``P1`` from the matrix's own
        column statistics rather than from numpy, so the same question is worth
        asking again: stage 2's cells must be penalised by the cell rule, not by
        whatever the caller asked for on the mains."""
        spec = two_stage_spec(book)
        mains = spec.main_effects_spec()
        heavy = np.full(mains.n_features, 50.0)  # crush the mains
        with_p1 = fit_two_stage(
            book, spec, "ClaimNb", alpha=0.001, sparse=True, P1=heavy, **self.FIT_KW
        )
        plain = fit_two_stage(
            book, spec, "ClaimNb", alpha=0.001, sparse=True, **self.FIT_KW
        )
        # stage 1 really was penalised harder ...
        assert int((with_p1.stage1.coef != 0).sum()) < int(
            (plain.stage1.coef != 0).sum()
        )
        # ... and stage 2 still ran on its own columns, unpenalised by that P1
        assert with_p1.stage2.spec.n_features == plain.stage2.spec.n_features
        assert (with_p1.stage2.coef != 0).sum() > 0

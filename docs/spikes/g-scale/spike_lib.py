"""Shared pieces of the workstream-G scale spike (easy_glm 0.4).

Nothing here touches the easy_glm repo; ``easy_glm`` is only imported.

Contents
--------
* synthetic Poisson frequency data (cached as parquet per size)
* ``codes()``      -- per-variable integer bin / level codes for a DesignSpec
* ``predict_from_codes()`` -- float64 rate-table style scoring (what RateModel does)
* dense builders (float32/float64) written straight into a preallocated array
* SplitMatrix builder (CategoricalMatrix blocks + dense step block)
* aggregation by identical design row (the Emblem trick)
* ``StepMatrix`` -- prototype tabmat block for step columns via the cumsum trick
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sps
import tabmat as tm
from tabmat.matrix_base import MatrixBase

from easy_glm.core.design import CategoricalEncoder, DesignSpec, StepEncoder

STEP_VARS = ["driv_age", "veh_age", "bonus_malus", "density", "veh_power", "mileage"]
CAT_VARS = ["region", "brand", "area"]
PREDICTORS = STEP_VARS + CAT_VARS

FRENCH_PREDICTORS = [
    "DrivAge",
    "VehAge",
    "BonusMalus",
    "Density",
    "VehPower",
    "Region",
    "VehBrand",
    "VehGas",
    "Area",
]


# --------------------------------------------------------------------------
# synthetic data
# --------------------------------------------------------------------------
def make_data(n: int, seed: int = 0) -> pl.DataFrame:
    """Synthetic motor frequency data with mildly realistic marginals.

    Integer-valued rating factors (as real data), skewed level frequencies,
    ~1% nulls in two variables, a rare-level tail in the categoricals, and a
    Poisson claim count with exposure between 0.05 and 1.
    """
    rng = np.random.default_rng(seed)
    driv_age = np.clip(18 + rng.gamma(4.0, 6.5, n), 18, 95).astype(np.int32)
    veh_age = np.clip(rng.gamma(2.0, 3.5, n), 0, 35).astype(np.int32)
    bonus_malus = np.clip(50 + rng.exponential(18, n) * (rng.random(n) < 0.45), 50, 230)
    bonus_malus = bonus_malus.astype(np.int32)
    density = np.clip(np.exp(rng.normal(6.0, 1.6, n)), 1, 30000).astype(np.int32)
    veh_power = np.clip(rng.poisson(3.0, n) + 4, 4, 15).astype(np.int32)
    mileage = np.round(np.exp(rng.normal(9.3, 0.6, n)) / 500) * 500  # km, 500 km grid

    def skewed_levels(k: int, prefix: str) -> np.ndarray:
        p = np.exp(-np.arange(k) / (k / 3.0))
        p /= p.sum()
        return np.array([f"{prefix}{i}" for i in range(k)])[rng.choice(k, n, p=p)]

    region = skewed_levels(20, "R")
    brand = skewed_levels(12, "B")
    area = skewed_levels(10, "A")

    exposure = np.clip(rng.uniform(0.05, 1.0, n), 0.05, 1.0)

    # planted effects: step-ish in age, monotone in bonus_malus, brand/region loads
    lp = (
        np.log(0.07)
        + 0.6 * (driv_age < 25)
        + 0.15 * (driv_age < 30)
        - 0.15 * (driv_age > 60)
        + 0.012 * (bonus_malus - 50)
        + 0.08 * np.log1p(density) / 3
        - 0.01 * veh_age
        + 0.03 * (veh_power - 6)
        + 0.10 * (region == "R0")
        - 0.10 * (region == "R2")
        + 0.15 * (brand == "B1")
        + 0.05 * (area == "A3")
    )
    claims = rng.poisson(exposure * np.exp(lp)).astype(np.float64)

    df = pl.DataFrame(
        {
            "driv_age": driv_age,
            "veh_age": veh_age,
            "bonus_malus": bonus_malus,
            "density": density,
            "veh_power": veh_power,
            "mileage": mileage,
            "region": region,
            "brand": brand,
            "area": area,
            "exposure": exposure,
            "claims": claims,
        }
    )
    # ~1% nulls in two variables
    null_a = rng.random(n) < 0.01
    null_b = rng.random(n) < 0.01
    df = df.with_columns(
        pl.when(pl.Series(null_a)).then(None).otherwise(pl.col("veh_age")).alias("veh_age"),
        pl.when(pl.Series(null_b)).then(None).otherwise(pl.col("brand")).alias("brand"),
    )
    return df


# --------------------------------------------------------------------------
# integer codes per variable (the only thing every representation needs)
# --------------------------------------------------------------------------
def var_codes(spec: DesignSpec, df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Bin / level code per variable.

    step var with K knots: code 0..K = bin (searchsorted, right), K+1 = null
    categorical with L kept levels: 0 = reference, 1..L-1 = levels[1:], L = Other/null
    """
    out: dict[str, np.ndarray] = {}
    for var, enc in spec.encoders.items():
        if isinstance(enc, StepEncoder):
            x = df[var].cast(pl.Float64).to_numpy()
            knots = np.asarray(enc.knots)
            c = np.searchsorted(knots, x, side="right")
            c = np.where(np.isnan(x), len(knots) + 1, c)
            out[var] = c.astype(np.int32)
        else:
            assert isinstance(enc, CategoricalEncoder)
            L = len(enc.levels)
            mapping = {lvl: i for i, lvl in enumerate(enc.levels)}
            s = df[var].cast(pl.Utf8)
            c = s.replace_strict(mapping, default=L, return_dtype=pl.Int32).fill_null(L)
            out[var] = c.to_numpy().astype(np.int32)
    return out


def n_codes(spec: DesignSpec, var: str) -> int:
    enc = spec.encoders[var]
    if isinstance(enc, StepEncoder):
        return len(enc.knots) + 2
    return len(enc.levels) + 1


def predict_from_codes(
    spec: DesignSpec,
    codes: dict[str, np.ndarray],
    coef: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Float64 rate-table scoring: exp(intercept + sum of per-variable lookups).

    For a step variable the lookup is ``cumsum(step coefs)`` (with 0 for the
    lowest bin and the null coefficient for the null row), for a categorical
    it is ``[0, *coefs]``. This is exactly what ``RateModel.predict`` does.
    """
    coef = np.asarray(coef, dtype=np.float64)
    lp = np.full(len(next(iter(codes.values()))), float(intercept), dtype=np.float64)
    for var, sl in spec.slices().items():
        enc = spec.encoders[var]
        b = coef[sl]
        if isinstance(enc, StepEncoder):
            K = len(enc.knots)
            table = np.concatenate([[0.0], np.cumsum(b[:K])])
            if enc.null_indicator:
                table = np.concatenate([table, [b[K]]])
        else:
            table = np.concatenate([[0.0], b])
        lp += table[codes[var]]
    return np.exp(lp)


# --------------------------------------------------------------------------
# dense builders (write straight into the output array; no hstack transients)
# --------------------------------------------------------------------------
def build_dense_from_codes(
    spec: DesignSpec, codes: dict[str, np.ndarray], dtype=np.float64, n: int | None = None
) -> np.ndarray:
    n = len(next(iter(codes.values()))) if n is None else n
    out = np.empty((n, spec.n_features), dtype=dtype, order="F")
    for var, sl in spec.slices().items():
        enc = spec.encoders[var]
        c = codes[var]
        if isinstance(enc, StepEncoder):
            K = len(enc.knots)
            for j in range(K):
                # 1{x >= knot_j} == 1{bin >= j+1}, nulls (code K+1) excluded
                col = out[:, sl.start + j]
                np.copyto(col, (c >= j + 1) & (c <= K), casting="unsafe")
            if enc.null_indicator:
                np.copyto(out[:, sl.start + K], c == K + 1, casting="unsafe")
        else:
            L = len(enc.levels)
            for j in range(L):  # levels[1:] then Other
                np.copyto(out[:, sl.start + j], c == j + 1, casting="unsafe")
    return out


def step_block_dense(
    spec: DesignSpec, codes: dict[str, np.ndarray], dtype=np.float64
) -> tuple[np.ndarray, list[int]]:
    """Dense block holding all step (+ null) columns, and their global indices."""
    n = len(next(iter(codes.values())))
    idx: list[int] = []
    for var, sl in spec.slices().items():
        if isinstance(spec.encoders[var], StepEncoder):
            idx.extend(range(sl.start, sl.stop))
    out = np.empty((n, len(idx)), dtype=dtype, order="F")
    pos = 0
    for var, sl in spec.slices().items():
        enc = spec.encoders[var]
        if not isinstance(enc, StepEncoder):
            continue
        c = codes[var]
        K = len(enc.knots)
        for j in range(K):
            np.copyto(out[:, pos + j], (c >= j + 1) & (c <= K), casting="unsafe")
        if enc.null_indicator:
            np.copyto(out[:, pos + K], c == K + 1, casting="unsafe")
        pos += enc.n_features
    return out, idx


def cat_blocks(
    spec: DesignSpec, codes: dict[str, np.ndarray], dtype=np.float64
) -> list[tuple[tm.CategoricalMatrix, list[int]]]:
    """One CategoricalMatrix per categorical variable (drop_first drops the reference)."""
    blocks = []
    for var, sl in spec.slices().items():
        enc = spec.encoders[var]
        if isinstance(enc, CategoricalEncoder):
            L = len(enc.levels)
            m = tm.CategoricalMatrix(
                codes[var],
                categories=np.arange(L + 1),
                drop_first=True,
                dtype=dtype,
                column_name=var,
            )
            blocks.append((m, list(range(sl.start, sl.stop))))
    return blocks


def build_split(spec: DesignSpec, codes: dict[str, np.ndarray], dtype=np.float64) -> tm.SplitMatrix:
    dense, dense_idx = step_block_dense(spec, codes, dtype)
    mats = [tm.DenseMatrix(dense)]
    idxs = [np.asarray(dense_idx)]
    for m, idx in cat_blocks(spec, codes, dtype):
        mats.append(m)
        idxs.append(np.asarray(idx))
    return tm.SplitMatrix(mats, idxs)


# --------------------------------------------------------------------------
# aggregation by identical design row
# --------------------------------------------------------------------------
def aggregate(
    spec: DesignSpec,
    codes: dict[str, np.ndarray],
    y: np.ndarray,
    w: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Group rows with identical codes; return (codes_agg, ybar_agg, W_agg, group_of_row).

    ``y`` is the per-unit-of-weight target (e.g. claims / exposure) and ``w``
    the weight; the aggregate keeps ``W = sum w`` and ``ybar = sum(w*y) / W``,
    which leaves the weighted GLM objective unchanged up to a constant.
    """
    cols = {}
    for var, c in codes.items():
        k = n_codes(spec, var)
        dt = pl.UInt8 if k < 256 else pl.UInt16
        cols[var] = pl.Series(var, c).cast(dt)
    frame = pl.DataFrame(cols).with_columns(
        pl.Series("_w", w), pl.Series("_wy", w * y), pl.int_range(pl.len()).alias("_row")
    )
    keys = list(codes)
    agg = (
        frame.group_by(keys, maintain_order=False)
        .agg(pl.col("_w").sum(), pl.col("_wy").sum(), pl.col("_row").alias("_rows"))
        .with_row_index("_g")
    )
    # map each original row to its group (only needed to broadcast predictions back)
    group_of_row = np.empty(frame.height, dtype=np.int64)
    exploded = agg.select("_g", "_rows").explode("_rows")
    group_of_row[exploded["_rows"].to_numpy()] = exploded["_g"].to_numpy()
    codes_agg = {k: agg[k].cast(pl.Int32).to_numpy() for k in keys}
    W = agg["_w"].to_numpy().astype(np.float64)
    ybar = agg["_wy"].to_numpy().astype(np.float64) / W
    return codes_agg, ybar, W, group_of_row


def n_distinct_rows(spec: DesignSpec, codes: dict[str, np.ndarray]) -> int:
    return pl.DataFrame({k: pl.Series(k, v) for k, v in codes.items()}).n_unique()


# --------------------------------------------------------------------------
# StepMatrix: tabmat block for the K step columns of one variable
# --------------------------------------------------------------------------
def _rows_view(arr: np.ndarray, rows):
    return arr if rows is None else arr[rows]


class StepMatrix(MatrixBase):
    """Step columns ``1{bin >= j+1}``, j = 0..K-1, stored as one int code per row.

    ``code`` in 0..K is the bin; ``code == K+1`` (null) contributes zeros to all
    step columns (the null indicator lives in a separate dense column).

    Only the three operations glum needs are implemented with the cumulative-sum
    trick; the remaining abstract methods are minimal.

      X beta          = cumsum-table[code]
      X^T v           = reverse-cumsum(bincount(code, v))[1:K+1]
      X^T diag(d) X   = S[max(j,k)+1] with S = reverse-cumsum(bincount(code, d))
    """

    def __init__(self, code: np.ndarray, n_knots: int, dtype=np.float64, name: str = "step"):
        self.code = np.ascontiguousarray(code)
        self.K = int(n_knots)
        self.shape = (len(self.code), self.K)
        self.dtype = np.dtype(dtype)
        self._name = name

    # -- helpers -----------------------------------------------------------
    def _bincount(self, v: np.ndarray, rows) -> np.ndarray:
        """Per-bin sums of v (length K+2; index K+1 is the null bin)."""
        c = _rows_view(self.code, rows)
        v = _rows_view(np.asarray(v), rows)
        return np.bincount(c, weights=v, minlength=self.K + 2)[: self.K + 2]

    def _tail_sums(self, v: np.ndarray, rows) -> np.ndarray:
        """S[j] = sum_i v_i 1{code_i >= j}, j = 0..K (null rows excluded)."""
        bc = self._bincount(v, rows)[: self.K + 1]
        return np.cumsum(bc[::-1])[::-1]

    # -- the three operations ----------------------------------------------
    def matvec(self, other, cols=None, out=None):
        other = np.asarray(other)
        if other.ndim == 2:  # (K, m) -> (n, m)
            res = np.column_stack([self.matvec(other[:, i], cols) for i in range(other.shape[1])])
            if out is not None:
                out[:] += res
                return out
            return res
        beta = np.zeros(self.K, dtype=np.result_type(self.dtype, other.dtype))
        if cols is None:
            beta[:] = other
        else:
            beta[cols] = other[cols]
        table = np.concatenate([[0.0], np.cumsum(beta), [0.0]])  # index K+1 = null -> 0
        res = table[self.code]
        if out is not None:
            out[:] += res
            return out
        return res

    def transpose_matvec(self, vec, rows=None, cols=None, out=None):
        vec = np.asarray(vec)
        if vec.ndim == 2:
            res = np.column_stack([self.transpose_matvec(vec[:, i], rows, cols) for i in range(vec.shape[1])])
            if out is not None:
                out[:] += res
                return out
            return res
        S = self._tail_sums(vec, rows)  # length K+1
        full = S[1 : self.K + 1]
        if cols is None:
            res = full
        else:
            res = full[cols]
        if out is not None:
            if cols is None:
                out[:] += res
            else:
                out[cols] += res
            return out
        return res.astype(self.dtype, copy=False)

    def sandwich(self, d, rows=None, cols=None):
        d = np.asarray(d)
        S = self._tail_sums(d, rows)  # S[j], j=0..K
        j = np.arange(self.K)
        M = S[np.maximum(j[:, None], j[None, :]) + 1]
        if cols is not None:
            M = M[np.ix_(cols, cols)]
        return M.astype(self.dtype, copy=False)

    # -- cross products with other blocks (SplitMatrix calls left._cross_sandwich(right)) --
    def _cross_sandwich(self, other, d, rows=None, L_cols=None, R_cols=None):
        d = np.asarray(d)
        c = _rows_view(self.code, rows)
        dd = _rows_view(d, rows)
        if isinstance(other, StepMatrix):
            oc = _rows_view(other.code, rows)
            # joint bincount over (bin, bin'), weights d; then 2-D reverse cumsum
            T = np.bincount(
                c * (other.K + 2) + oc, weights=dd, minlength=(self.K + 2) * (other.K + 2)
            ).reshape(self.K + 2, other.K + 2)[: self.K + 1, : other.K + 1]
            S = np.cumsum(np.cumsum(T[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
            res = S[1:, 1:]
        elif isinstance(other, tm.CategoricalMatrix):
            oc = _rows_view(other.indices, rows)
            ncat = len(other.categories)
            T = np.bincount(
                c * ncat + oc, weights=dd, minlength=(self.K + 2) * ncat
            ).reshape(self.K + 2, ncat)[: self.K + 1]
            S = np.cumsum(T[::-1], axis=0)[::-1]
            res = S[1:]
            if other.drop_first:
                res = res[:, 1:]
        elif isinstance(other, tm.DenseMatrix):
            Y = _rows_view(other._array, rows)
            n = len(c)
            # per-bin column sums of d * Y via a sparse indicator matmul
            ind = sps.csr_matrix(
                (dd, (c, np.arange(n))), shape=(self.K + 2, n)
            )
            T = np.asarray(ind @ Y)[: self.K + 1]
            S = np.cumsum(T[::-1], axis=0)[::-1]
            res = S[1:]
        else:
            raise TypeError(f"StepMatrix cannot cross with {type(other)}")
        if L_cols is not None:
            res = res[L_cols]
        if R_cols is not None:
            res = res[:, R_cols]
        return np.asarray(res, dtype=self.dtype)

    # -- the rest of the abstract interface --------------------------------
    def getcol(self, i):
        i %= self.K
        return ((self.code >= i + 1) & (self.code <= self.K)).astype(self.dtype)[:, None]

    def toarray(self):
        j = np.arange(self.K)
        return ((self.code[:, None] >= j + 1) & (self.code[:, None] <= self.K)).astype(self.dtype)

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        return StepMatrix(self.code, self.K, dtype=dtype, name=self._name)

    def _get_col_stds(self, weights, col_means):
        mean = self.transpose_matvec(weights)  # 0/1 columns: E[x^2] = E[x]
        return np.sqrt(np.maximum(mean - col_means**2, 0)).astype(self.dtype)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
        else:
            row, col = item, slice(None)
        if not (col == slice(None) or (isinstance(col, slice) and col == slice(None, None, None))):
            return tm.DenseMatrix(self.toarray()[row, col])
        if isinstance(row, np.ndarray):
            row = row.ravel()
        return StepMatrix(self.code[row], self.K, self.dtype, self._name)

    def get_names(self, type="column", missing_prefix=None, indices=None):
        return [f"{self._name}>=k{j}" for j in range(self.K)]

    def set_names(self, names, type="column"):
        pass

    def __repr__(self):
        return f"StepMatrix(n={self.shape[0]}, K={self.K}, dtype={self.dtype})"


def build_split_stepmatrix(
    spec: DesignSpec, codes: dict[str, np.ndarray], dtype=np.float64
) -> tm.SplitMatrix:
    """SplitMatrix with StepMatrix blocks first (so their _cross_sandwich is used),
    then one dense block for the null indicators, then CategoricalMatrix blocks."""
    n = len(next(iter(codes.values())))
    mats, idxs = [], []
    null_cols: list[int] = []
    null_arrays: list[np.ndarray] = []
    for var, sl in spec.slices().items():
        enc = spec.encoders[var]
        if isinstance(enc, StepEncoder):
            K = len(enc.knots)
            mats.append(StepMatrix(codes[var], K, dtype, name=var))
            idxs.append(np.arange(sl.start, sl.start + K))
            if enc.null_indicator:
                null_cols.append(sl.start + K)
                null_arrays.append(codes[var] == K + 1)
    if null_cols:
        dense = np.empty((n, len(null_cols)), dtype=dtype, order="F")
        for j, a in enumerate(null_arrays):
            np.copyto(dense[:, j], a, casting="unsafe")
        mats.append(tm.DenseMatrix(dense))
        idxs.append(np.asarray(null_cols))
    for m, idx in cat_blocks(spec, codes, dtype):
        mats.append(m)
        idxs.append(np.asarray(idx))
    return tm.SplitMatrix(mats, idxs)


def patch_glum_validation() -> None:
    """Let glum accept any tabmat MatrixBase subclass inside a SplitMatrix.

    glum's ``check_array_tabmat_compliant`` only passes through the tabmat
    classes it knows; anything else goes to sklearn's ``check_array`` and is
    densified/rejected. This spike-only patch short-circuits unknown MatrixBase
    subclasses. In the product this is a small upstream PR to glum (or a
    subclass registered in that function).
    """
    import glum._glm as g
    import glum._validation as v

    orig = v.check_array_tabmat_compliant

    def patched(mat, drop_first=False, **kwargs):
        if isinstance(mat, StepMatrix):
            return mat
        if isinstance(mat, tm.SplitMatrix):
            new = [patched(m, drop_first=drop_first, **kwargs) for m in mat.matrices]
            return tm.SplitMatrix(new, mat.indices)
        return orig(mat, drop_first=drop_first, **kwargs)

    v.check_array_tabmat_compliant = patched
    g.check_array_tabmat_compliant = patched

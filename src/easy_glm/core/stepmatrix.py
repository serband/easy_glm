"""``StepMatrix`` — step (O-dummy) design columns stored as one bin index per row.

A step term with knots ``k_1 < ... < k_K`` contributes ``K`` columns
``1{x >= k_j}`` to the design matrix. Written out densely that is
``8 * n * K`` bytes; at 5M rows and 25 knots, one variable alone costs a
gigabyte. But the whole block is determined by a single number per row — the
**bin index** ``b_i = #{j : x_i >= k_j}`` in ``0..K`` (with ``K + 1`` reserved
for a null, which contributes zeros to every step column and carries its own
``is null`` column elsewhere). Storing that index as an ``int32`` costs
``4 * n`` bytes **regardless of the number of knots**.

Every operation glum asks a design matrix for can be computed from the index
with a cumulative sum, so nothing is ever expanded (see the spike report,
``docs/spikes/g-scale/SPIKE_REPORT.md`` appendix B):

``X beta``
    ``c = [0, cumsum(beta), 0]``; the answer is ``c[b]`` — one gather.
``X.T v``
    ``S[j] = sum_i v_i 1{b_i >= j}`` is the reverse cumulative sum of
    ``bincount(b, v)``; the answer is ``S[1..K]``.
``X.T diag(d) X``
    with the same ``S`` built from ``d``, entry ``(j, l)`` is
    ``S[max(j, l) + 1]``, because a row contributes to both columns exactly
    when its bin is at least the larger of the two.

Cross products with the other blocks of a :class:`tabmat.SplitMatrix` follow
the same idea in two dimensions (a joint ``bincount`` then a reverse cumulative
sum along the bin axis).

Two rules the rest of the package depends on
--------------------------------------------
* **float64 only.** The spike found that float32 designs stop converging past
  1M rows under glum 3.4.1, return a float32 ``coef_``, and segfault tabmat
  when handed an uncast float64 ``sample_weight``. The dtype argument exists
  because tabmat's interface has one; ``DesignSpec.build`` never passes
  anything but float64. See ``SPIKE_REPORT.md`` §4.2 before reconsidering.
* **StepMatrix blocks must come first** in a ``SplitMatrix``.
  ``SplitMatrix.sandwich`` only ever calls ``matrices[i]._cross_sandwich(
  matrices[j])`` for ``i < j``, and tabmat's own blocks raise ``TypeError`` on
  a block type they do not know. Putting the step blocks first means the cross
  products are always dispatched to *our* implementation, which does know about
  theirs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sps
import tabmat as tm
from tabmat import MatrixBase

__all__ = ["StepMatrix", "install_glum_shim"]


def _rows_view(arr: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
    """``arr`` restricted to ``rows`` (``None`` = every row, no copy)."""
    return arr if rows is None else arr[rows]


class StepMatrix(MatrixBase):
    """The ``K`` step columns of one variable, held as one bin index per row.

    Parameters
    ----------
    code : np.ndarray
        Bin index per row: ``0..K`` for a value (``0`` = below the first knot),
        ``K + 1`` for a null. Stored as ``int32``.
    n_knots : int
        ``K`` — the number of columns this block contributes.
    dtype : np.dtype
        Declared dtype of the block. Always ``float64`` in easy_glm (see the
        module docstring); the internal sums go through ``np.bincount``, which
        accumulates in float64 whatever the declared dtype is.
    name : str
        The variable's name, used to build column names.
    """

    _cross_dtype = np.float64

    def __init__(
        self,
        code: np.ndarray,
        n_knots: int,
        dtype: Any = np.float64,
        name: str = "step",
    ) -> None:
        code = np.ascontiguousarray(code, dtype=np.int32)
        if code.ndim != 1:
            raise ValueError("StepMatrix code must be a one-dimensional array")
        n_knots = int(n_knots)
        if n_knots < 1:
            raise ValueError("StepMatrix needs at least one knot")
        if code.size and (code.min() < 0 or code.max() > n_knots + 1):
            raise ValueError(
                f"StepMatrix codes must lie in 0..{n_knots + 1} "
                f"(bins 0..{n_knots}, {n_knots + 1} = null); got "
                f"{code.min()}..{code.max()}"
            )
        self.code = code
        self.K = n_knots
        self.shape = (int(code.shape[0]), n_knots)
        self.dtype = np.dtype(dtype)
        self._name = name
        self._colnames: list[str | None] = [f"{name}>=k{j + 1}" for j in range(n_knots)]
        self._terms: list[str | None] = [name] * n_knots

    # -- helpers -----------------------------------------------------------
    def _bincount(self, v: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
        """Per-bin sums of ``v``; length ``K + 2`` (index ``K + 1`` = nulls)."""
        c = _rows_view(self.code, rows)
        vv = _rows_view(np.asarray(v, dtype=np.float64), rows)
        return np.bincount(c, weights=vv, minlength=self.K + 2)[: self.K + 2]

    def _tail_sums(self, v: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
        """``S[j] = sum_i v_i 1{code_i >= j}`` for ``j = 0..K`` (nulls excluded).

        ``S[j]`` is the column sum of step column ``j`` weighted by ``v``,
        because column ``j`` is ``1{code >= j}`` for a non-null row.
        """
        bc = self._bincount(v, rows)[: self.K + 1]
        return np.cumsum(bc[::-1])[::-1]

    # -- the three operations glum needs -----------------------------------
    def matvec(
        self,
        other: np.ndarray | list,
        cols: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """``self[:, cols] @ other[cols]`` — a cumulative sum then a gather."""
        other = np.asarray(other)
        if other.ndim == 2:
            res2 = np.column_stack(
                [self.matvec(other[:, i], cols) for i in range(other.shape[1])]
            )
            if out is not None:
                out[:] += res2
                return out
            return res2
        if other.shape[0] != self.K:
            raise ValueError(f"shapes {self.shape} and {other.shape} not aligned")
        beta = np.zeros(self.K, dtype=np.result_type(self.dtype, other.dtype))
        if cols is None:
            beta[:] = other
        else:
            beta[cols] = other[cols]
        # table[b] = sum of the first b coefficients; null (K+1) contributes 0
        table = np.concatenate([[0.0], np.cumsum(beta), [0.0]])
        res = table[self.code]
        if out is not None:
            out[:] += res
            return out
        return res

    def transpose_matvec(
        self,
        vec: np.ndarray | list,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """``self[rows, cols].T @ vec[rows]`` — a bincount then a reverse cumsum."""
        vec = np.asarray(vec)
        if vec.ndim == 2:
            res2 = np.column_stack(
                [
                    self.transpose_matvec(vec[:, i], rows, cols)
                    for i in range(vec.shape[1])
                ]
            )
            if out is not None:
                if cols is None:
                    out[:] += res2
                else:
                    out[cols] += res2
                return out
            return res2
        full = self._tail_sums(vec, rows)[1 : self.K + 1]
        res = full if cols is None else full[cols]
        if out is not None:
            if cols is None:
                out[:] += res
            else:
                out[cols] += res
            return out
        return np.asarray(res, dtype=np.float64)

    def sandwich(
        self,
        d: np.ndarray,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(self[rows, cols].T * d[rows]) @ self[rows, cols]``.

        Entry ``(j, l)`` counts the rows whose bin is at least ``max(j, l) + 1``,
        so the whole ``K x K`` table is read off one length-``K + 1`` vector.
        """
        tails = self._tail_sums(np.asarray(d), rows)
        j = np.arange(self.K)
        table = tails[np.maximum(j[:, None], j[None, :]) + 1]
        if cols is not None:
            table = table[np.ix_(cols, cols)]
        return np.asarray(table, dtype=np.float64)

    # -- cross products with the other blocks ------------------------------
    def _cross_sandwich(
        self,
        other: MatrixBase,
        d: np.ndarray,
        rows: np.ndarray | None = None,
        L_cols: np.ndarray | None = None,  # noqa: N803 - tabmat's own
        R_cols: np.ndarray | None = None,  # noqa: N803 - parameter names
    ) -> np.ndarray:
        """``self[rows, L_cols].T @ diag(d[rows]) @ other[rows, R_cols]``.

        ``SplitMatrix.sandwich`` calls this for every pair of blocks with this
        one on the left, which is why step blocks have to be first: tabmat's
        own blocks would raise ``TypeError`` on a block type they do not know.
        """
        d = np.asarray(d, dtype=np.float64)
        c = _rows_view(self.code, rows)
        dd = _rows_view(d, rows)
        if isinstance(other, StepMatrix):
            oc = _rows_view(other.code, rows)
            # joint per-(bin, bin') sums of d, then a reverse cumsum on both axes
            joint = np.bincount(
                c.astype(np.int64) * (other.K + 2) + oc,
                weights=dd,
                minlength=(self.K + 2) * (other.K + 2),
            ).reshape(self.K + 2, other.K + 2)[: self.K + 1, : other.K + 1]
            tails = np.cumsum(np.cumsum(joint[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
            res = tails[1:, 1:]
        elif isinstance(other, tm.CategoricalMatrix):
            oc = _rows_view(other.indices, rows)
            ncat = len(other.categories)
            joint = np.bincount(
                c.astype(np.int64) * ncat + oc,
                weights=dd,
                minlength=(self.K + 2) * ncat,
            ).reshape(self.K + 2, ncat)[: self.K + 1]
            res = np.cumsum(joint[::-1], axis=0)[::-1][1:]
            if other.drop_first:
                res = res[:, 1:]
        elif isinstance(other, tm.DenseMatrix):
            right = _rows_view(np.asarray(other.unpack()), rows)
            n = len(c)
            # per-bin column sums of d * other, as one sparse indicator matmul
            indicator = sps.csr_matrix(
                (dd, (c, np.arange(n, dtype=np.int64))), shape=(self.K + 2, n)
            )
            joint = np.asarray(indicator @ right)[: self.K + 1]
            res = np.cumsum(joint[::-1], axis=0)[::-1][1:]
        else:
            raise TypeError(
                f"StepMatrix has no cross-sandwich with {type(other).__name__}. "
                "Only StepMatrix, CategoricalMatrix and DenseMatrix blocks are "
                "supported; see easy_glm.core.design.DesignSpec.build."
            )
        if L_cols is not None:
            res = res[L_cols]
        if R_cols is not None:
            res = res[:, R_cols]
        return np.asarray(res, dtype=np.float64)

    # -- the rest of the MatrixBase interface ------------------------------
    def getcol(self, i: int) -> tm.DenseMatrix:
        """Column ``i`` as a one-column dense block."""
        i = int(i) % self.K
        col = ((self.code >= i + 1) & (self.code <= self.K)).astype(self.dtype)
        return tm.DenseMatrix(
            col[:, None],
            column_names=[self._colnames[i]],
            term_names=[self._terms[i]],
        )

    def toarray(self) -> np.ndarray:
        """The dense ``(n, K)`` block this stands for (tests and small data)."""
        j = np.arange(self.K)
        code = self.code[:, None]
        return ((code >= j + 1) & (code <= self.K)).astype(self.dtype)

    def astype(self, dtype, order="K", casting="unsafe", copy=True) -> StepMatrix:
        """A block with the same codes and a new declared dtype."""
        return StepMatrix(self.code, self.K, dtype=dtype, name=self._name)

    def copy(self) -> StepMatrix:
        """A block with its own copy of the codes."""
        return StepMatrix(self.code.copy(), self.K, dtype=self.dtype, name=self._name)

    def _get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        """Weighted column standard deviations.

        The columns are 0/1, so ``E[x^2] = E[x]`` and the variance is
        ``mean - mean^2`` — tabmat's own trick for ``CategoricalMatrix``.
        ``standardize`` / ``unstandardize`` are the base-class implementations
        on top of this and ``transpose_matvec``; nothing here needs overriding.
        """
        mean = self.transpose_matvec(weights)
        return np.sqrt(np.maximum(mean - np.asarray(col_means) ** 2, 0.0))

    def __getitem__(self, item) -> StepMatrix | tm.DenseMatrix:
        """Row subsetting (what cross-validation does per fold) keeps the codes.

        A fold therefore costs ``4`` bytes per row rather than ``8 * K``.
        Column subsetting is rare and falls back to a dense block.
        """
        if isinstance(item, tuple):
            row, col = item
        else:
            row, col = item, slice(None)
        if not (isinstance(col, slice) and col == slice(None)):
            return tm.DenseMatrix(self.toarray()[row, col])
        if isinstance(row, np.ndarray):
            row = row.ravel()
        elif isinstance(row, int):
            row = [row]
        return StepMatrix(self.code[row], self.K, dtype=self.dtype, name=self._name)

    def get_names(
        self,
        type: str = "column",
        missing_prefix: str | None = None,
        indices: list[int] | None = None,
    ) -> list[str | None]:
        """Column (or term) names — one term, ``K`` columns."""
        if type == "column":
            return list(self._colnames)
        if type == "term":
            return list(self._terms)
        raise ValueError(f"Type must be 'column' or 'term', got {type}")

    def set_names(self, names, type: str = "column") -> None:
        """Set column (or term) names."""
        if isinstance(names, str):
            names = [names]
        if len(names) != self.shape[1]:
            raise ValueError(f"Length of names must be {self.shape[1]}")
        if type == "column":
            self._colnames = list(names)
        elif type == "term":
            self._terms = list(names)
        else:
            raise ValueError(f"Type must be 'column' or 'term', got {type}")

    @property
    def nbytes(self) -> int:
        """Bytes the block actually holds — ``4`` per row, whatever ``K`` is."""
        return int(self.code.nbytes)

    def __repr__(self) -> str:
        return (
            f"StepMatrix({self._name!r}, n={self.shape[0]}, knots={self.K}, "
            f"dtype={self.dtype})"
        )


# --------------------------------------------------------------------------
# glum shim
# --------------------------------------------------------------------------
_SHIM_INSTALLED = False


def install_glum_shim() -> None:
    """Teach glum's input validation to pass a :class:`StepMatrix` through.

    glum validates the design with ``glum._validation.check_array_tabmat_compliant``,
    which passes through the five tabmat classes it knows and sends everything
    else to ``sklearn.check_array`` — which would densify (or reject) our block.
    The function is private and takes no registry, so the only way in is to wrap
    it. The wrapper is **one branch**: a ``StepMatrix`` is already compliant, so
    return it (copying only when the caller asked for a copy); everything else
    goes to the original function unchanged. glum's own recursion into a
    ``SplitMatrix`` looks the name up in its module globals, so patching the
    module makes the blocks go through the wrapper too.

    This is deliberately the smallest possible change and is pinned to
    ``glum 3.4.*`` by ``pyproject.toml``. The upstream fix is a one-line
    ``isinstance(mat, tabmat.MatrixBase)`` pass-through in that function; an
    issue/PR against glum is tracked in ``docs/checks/g-scale.md``. If a future
    glum removes or renames the function this raises immediately rather than
    silently densifying a 5M-row design, and the fallback is
    ``DesignSpec.build(..., sparse=False)``.

    Idempotent and safe to call on every fit.
    """
    global _SHIM_INSTALLED
    if _SHIM_INSTALLED:
        return
    import glum
    import glum._glm as glm_module
    import glum._validation as validation_module

    version = getattr(glum, "__version__", "unknown")
    original = getattr(validation_module, "check_array_tabmat_compliant", None)
    if original is None:  # pragma: no cover - guards a future glum
        raise RuntimeError(
            f"glum {version} has no check_array_tabmat_compliant; easy_glm's "
            "sparse design cannot be validated by it. Fit with "
            "DesignSpec.build(..., sparse=False) or pin glum 3.4.*."
        )

    def patched(mat, drop_first: bool = False, **kwargs):
        if isinstance(mat, StepMatrix):
            return mat.copy() if kwargs.get("copy", False) else mat
        return original(mat, drop_first=drop_first, **kwargs)

    patched.__wrapped__ = original  # type: ignore[attr-defined]
    validation_module.check_array_tabmat_compliant = patched
    glm_module.check_array_tabmat_compliant = patched
    _SHIM_INSTALLED = True

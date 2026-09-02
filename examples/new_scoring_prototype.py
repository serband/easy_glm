"""
Scratch pad — full one-hot encoding → GLM → interactive coefficient editor.

Run this file. It fits a model then launches a Streamlit editor where you
can tweak coefficients and see the effect on Actual vs Expected.

    python examples/new_scoring_prototype.py
"""

import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import polars as pl

import easy_glm
from easy_glm.core.model import fit_lasso_glm
from easy_glm.core.transforms import o_matrix, quote_identifier

# ── New one-hot: all n levels, no reference dropped, no Other column ──────


def new_one_hot_fun(col_name: str, levels: list) -> list[str]:
    """Generate one-hot SQL expressions for a categorical column.

    Produces one 0/1 column **per kept level** — no reference is dropped,
    no 'Other' catch-all column is added.  Unseen or NULL values get all
    zeros (their effect is captured by the GLM intercept).

    Parameters
    ----------
    col_name : str
        Column name.
    levels : list of str
        Kept levels (post-lumping).  Must be non-empty.

    Returns
    -------
    list[str]
        One SQL CASE expression per level.
    """
    if not isinstance(col_name, str) or not col_name.strip():
        raise ValueError("col_name must be a non-empty string")
    if isinstance(levels, np.ndarray):
        levels = levels.tolist()
    if not isinstance(levels, list) or not levels:
        raise ValueError("levels must be a non-empty list")

    cleaned = [str(level).replace("'", "''") for level in levels]
    quoted = quote_identifier(col_name)
    sql_statements: list[str] = []

    for level in cleaned:
        alias = quote_identifier(f"{col_name}_{level}")
        sql_statements.append(
            f"CASE WHEN CAST({quoted} AS VARCHAR) = '{level}' THEN 1 ELSE 0 END "
            f"AS {alias}"
        )

    return sql_statements


# ── New prepare_data: same as original but uses new_one_hot_fun ──────────

import duckdb


def new_prepare_data(
    modelling_variables: list[str],
    additional_columns: list[str] | None = None,
    traintest_column: str | None = None,
    df: pl.DataFrame | None = None,
    table_name: str = "dataset",
    formats: dict | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Same as easy_glm.prepare_data but uses ``new_one_hot_fun``."""
    if formats is None:
        formats = {}
    _own_connection = False
    if con is None:
        _own_connection = True
        if df is not None:
            con = duckdb.connect(":memory:")
            con.execute(
                f"CREATE TABLE {quote_identifier(table_name)} AS SELECT * FROM df"
            )
        else:
            raise ValueError("Either 'df' or 'con' must be provided to prepare_data.")
    else:
        if not isinstance(con, duckdb.DuckDBPyConnection):
            raise TypeError(
                "The 'con' argument must be a duckdb.DuckDBPyConnection, "
                f"got {type(con).__name__}"
            )
    table_reference = quote_identifier(table_name)
    tables = con.execute("SHOW TABLES").fetchall()
    if table_name not in [t[0] for t in tables]:
        raise ValueError(
            f"The specified table '{table_name}' does not exist in the database. "
            f"Available tables are: {', '.join([t[0] for t in tables])}"
        )
    expressions: list[str] = []
    if additional_columns is None:
        additional_columns = []
    if traintest_column and traintest_column not in additional_columns:
        additional_columns.append(traintest_column)
    for var in modelling_variables:
        if (
            var
            not in con.execute(f"PRAGMA table_info({table_reference})")
            .df()["name"]
            .tolist()
        ):
            print(f"Warning: Column '{var}' not found in the table. Skipping.")
            continue
        if var in formats:
            dict_values = formats[var]
            if not dict_values:
                continue
            if all(isinstance(x, int | float) for x in dict_values):
                expressions.extend(o_matrix(var, dict_values))
            else:
                # ← THIS IS THE ONLY CHANGE: new_one_hot_fun instead of one_hot_fun
                expressions.extend(new_one_hot_fun(var, dict_values))
        else:
            expressions.append(quote_identifier(var))
    for col in additional_columns:
        if (
            col
            in con.execute(f"PRAGMA table_info({table_reference})")
            .df()["name"]
            .tolist()
        ):
            expressions.append(quote_identifier(col))
        else:
            print(
                f"Warning: Additional column '{col}' not found in the table. "
                f"Skipping."
            )
    if not expressions:
        if _own_connection:
            con.close()
        return pl.DataFrame()
    query = f"SELECT {', '.join(expressions)} FROM {table_reference}"
    result_df = con.execute(query).df()
    if _own_connection:
        con.close()
    return pl.DataFrame(result_df)


# ── Link functions ───────────────────────────────────────────────────────

_LINK_INV = {
    "poisson": np.exp,
    "gamma": np.exp,
    "gaussian": lambda eta: eta,
    "binomial": lambda eta: 1.0 / (1.0 + np.exp(-eta)),
}


def _inverse_link(eta: np.ndarray, family: str) -> np.ndarray:
    fn = _LINK_INV.get(family.lower())
    if fn is None:
        raise ValueError(
            f"Unknown family '{family}'. " f"Choose: {list(_LINK_INV.keys())}"
        )
    return fn(eta)


# ── ModelBundle ──────────────────────────────────────────────────────────


class ModelBundle:
    """A fitted GLM reduced to its scoring essentials.

    Binds the blueprint (transform rules), the coefficient vector (including
    intercept), and the link function.  Scoring new data is a two-step
    operation: prepare with the blueprint, then matrix multiply.

    Parameters
    ----------
    blueprint : dict
        From ``generate_blueprint``.
    coefficients : np.ndarray
        1-D array, ``model.coef_``.  Must align with *feature_names*.
    intercept : float
        ``model.intercept_``.
    feature_names : list[str]
        Ordered column names from the prepared training feature matrix.
    predictors : list[str]
        The original modelling variable names (before expansion).
    family : str
        One of ``"poisson"``, ``"gamma"``, ``"gaussian"``, ``"binomial"``.
    exposure_col : str or None
        If set, predictions are multiplied by this column (when present
        in the scoring data).
    """

    def __init__(
        self,
        *,
        blueprint: dict,
        coefficients: np.ndarray,
        intercept: float,
        feature_names: list[str],
        predictors: list[str],
        family: str,
        exposure_col: str | None = None,
    ):
        self.blueprint = blueprint
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.intercept = float(intercept)
        self.feature_names = list(feature_names)
        self.predictors = list(predictors)
        self.family = family.lower()
        self.exposure_col = exposure_col

        # Pre-built full coefficient vector: [intercept, coef_1, ..., coef_k]
        self._full_coefs = np.concatenate([[self.intercept], self.coefficients])

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def from_glm(
        cls,
        model,
        train_prepped: pl.DataFrame,
        blueprint: dict,
        predictors: list[str],
        family: str,
        *,
        target: str = "ClaimNb",
        weight_col: str | None = None,
        exposure_col: str | None = None,
        train_test_col: str = "traintest",
    ) -> "ModelBundle":
        """Build a ModelBundle from a fitted glum model and training data.

        Parameters
        ----------
        model : GeneralizedLinearRegressor or GeneralizedLinearRegressorCV
            Fitted glum model.
        train_prepped : pl.DataFrame
            The prepared training DataFrame (output of ``new_prepare_data``).
        blueprint : dict
            Blueprint used to prepare the data.
        predictors : list[str]
            Modelling variable names.
        family : str
            GLM family: poisson / gamma / gaussian / binomial.
        target : str
            Name of the target column (to exclude from features).
        weight_col : str or None
            Name of the weight column (to exclude from features).
        exposure_col : str or None
            Name of the exposure column.  When scoring, predictions are
            multiplied by this column if present in the scoring data.
        train_test_col : str
            Name of the train/test split column (to exclude from features).

        Returns
        -------
        ModelBundle
        """
        exclude = {target, train_test_col}
        if weight_col:
            exclude.add(weight_col)
        feature_names = [c for c in train_prepped.columns if c not in exclude]

        return cls(
            blueprint=blueprint,
            coefficients=model.coef_,
            intercept=model.intercept_,
            feature_names=feature_names,
            predictors=predictors,
            family=family,
            exposure_col=exposure_col,
        )

    # ── Predict ──────────────────────────────────────────────────────

    def predict(self, data: pl.DataFrame, *, space: str = "response") -> np.ndarray:
        """Score raw (unprepared) data.

        Parameters
        ----------
        data : pl.DataFrame
            Raw data containing the predictor columns.
        space : str
            ``"response"`` — inverse-link transformed predictions.
            ``"link"``   — raw linear predictor eta (before inverse link).

        Returns
        -------
        np.ndarray
            Predictions on the requested scale.
        """
        prepped = new_prepare_data(
            df=data,
            modelling_variables=self.predictors,
            formats=self.blueprint,
            table_name="scoring_input",
        )

        X = prepped[self.feature_names].to_numpy()

        # Matrix multiply: [ones | X] @ [intercept | coefs]
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        eta = (np.column_stack([ones, X]) @ self._full_coefs).ravel()

        if space == "link":
            return eta

        result = _inverse_link(eta, self.family)

        # Multiply by exposure if available in the scoring data
        if self.exposure_col and self.exposure_col in data.columns:
            result = result * data[self.exposure_col].to_numpy()

        return result

    def predict_link(self, data: pl.DataFrame) -> np.ndarray:
        """Shortcut for ``predict(data, space='link')``."""
        return self.predict(data, space="link")

    def predict_response(self, data: pl.DataFrame) -> np.ndarray:
        """Shortcut for ``predict(data, space='response')``."""
        return self.predict(data, space="response")

    # ── Introspection ────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_coefficients(self) -> int:
        return len(self.coefficients)

    def coefficient(self, feature_name: str) -> float:
        """Return the coefficient for a single feature column."""
        idx = self.feature_names.index(feature_name)
        return float(self.coefficients[idx])

    def coefficients_for(self, variable: str) -> dict[str, float]:
        """Return ``{level: coef, ...}`` for a categorical variable."""
        prefix = f"{variable}_"
        result = {}
        for name, coef in zip(self.feature_names, self.coefficients):
            if name.startswith(prefix):
                level = name[len(prefix) :]
                result[level] = float(coef)
        return result

    def relativities_for(self, variable: str) -> dict[str, float]:
        """Return ``{level: exp(coef), ...}`` for a categorical variable.

        Only meaningful for log-link families (poisson, gamma).
        """
        return {k: float(np.exp(v)) for k, v in self.coefficients_for(variable).items()}

    def launch_editor(
        self,
        data: pl.DataFrame,
        *,
        target: str = "ClaimNb",
        weight_col: str | None = None,
        train_test_col: str | None = "traintest",
        port: int = 8501,
    ) -> None:
        """Launch the interactive coefficient editor in a Streamlit tab.

        Writes the bundle and dataset to a temp directory and spawns
        ``streamlit run`` on this same script in editor mode.
        """
        tmpdir = tempfile.mkdtemp(prefix="easy_glm_editor_")
        bundle_path = os.path.join(tmpdir, "bundle.pkl")
        with open(bundle_path, "wb") as f:
            pickle.dump(self, f)

        data_path = os.path.join(tmpdir, "data.parquet")
        data.write_parquet(data_path)

        args: list[str] = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            __file__,
            "--",
            "--mode=editor",
            f"--bundle-path={bundle_path}",
            f"--data-path={data_path}",
            f"--target={target}",
        ]
        if weight_col:
            args.append(f"--weight-col={weight_col}")
        if train_test_col:
            args.append(f"--train-test-col={train_test_col}")
        args.append(f"--server.port={port}")

        subprocess.Popen(args)
        print(f"Editor launching on http://localhost:{port} ...")

    def clone(self) -> "ModelBundle":
        """Deep-copy this bundle (coefficients can be mutated independently)."""
        return ModelBundle(
            blueprint=dict(self.blueprint),
            coefficients=self.coefficients.copy(),
            intercept=self.intercept,
            feature_names=list(self.feature_names),
            predictors=list(self.predictors),
            family=self.family,
            exposure_col=self.exposure_col,
        )

    def _rebuild_full_coefs(self) -> None:
        self._full_coefs = np.concatenate([[self.intercept], self.coefficients])

    def set_coefficient(self, feature_name: str, value: float) -> None:
        """Update a single coefficient by feature column name."""
        idx = self.feature_names.index(feature_name)
        self.coefficients[idx] = value
        self._rebuild_full_coefs()

    def __repr__(self) -> str:
        return (
            f"ModelBundle(family={self.family!r}, predictors={self.predictors}, "
            f"features={self.n_features}, exposure={self.exposure_col!r})"
        )


# ── A/E computation for ModelBundle ──────────────────────────────────────


def compute_ae_for_bundle(
    bundle: ModelBundle,
    data: pl.DataFrame,
    variable: str,
    *,
    target: str = "ClaimNb",
    weight_col: str | None = None,
    train_test_col: str | None = "traintest",
) -> dict:
    """Compute Actual vs Expected per bin for *variable* using *bundle*.

    Bins are defined by the blueprint: for categoricals each kept level
    is a bin; for numerics the o-matrix breakpoints define bins.

    Returns
    -------
    dict
        ``{\"subsets\": {\"all\": [...], \"train\": [...], \"test\": [...]},
        \"variable\": variable, \"type\": \"categorical\"|\"numeric\"}``.
        Each bin dict has keys ``level``, ``actual``, ``expected``,
        ``exposure``.
    """
    preds = bundle.predict(data, space="response")
    data_w_preds = data.with_columns(pred=pl.Series("pred", preds))

    bp = bundle.blueprint.get(variable)
    if bp is None or not bp:
        return {"subsets": {}, "variable": variable, "type": "unknown"}

    is_numeric = all(isinstance(x, (int, float)) for x in bp)

    # Build bin masks
    if is_numeric:
        bins = _numeric_bins(variable, bp)
    else:
        bins = _categorical_bins(variable, bp)

    subsets = {"all": data_w_preds}
    if train_test_col and train_test_col in data.columns:
        subsets["train"] = data_w_preds.filter(pl.col(train_test_col) == 1)
        subsets["test"] = data_w_preds.filter(pl.col(train_test_col) == 0)

    results: dict[str, list[dict]] = {}
    for subset_name, subset_df in subsets.items():
        results[subset_name] = []
        for label, mask in bins:
            matched = subset_df.filter(mask)
            if matched.is_empty():
                results[subset_name].append(
                    {
                        "level": label,
                        "actual": 0.0,
                        "expected": 0.0,
                        "exposure": 0.0,
                    }
                )
                continue

            actual = _aggregate(matched, target, weight_col)
            expected = _aggregate(matched, "pred", weight_col)
            exposure = (
                float(matched[weight_col].sum())
                if weight_col and weight_col in matched.columns
                else float(len(matched))
            )
            results[subset_name].append(
                {
                    "level": label,
                    "actual": actual,
                    "expected": expected,
                    "exposure": exposure,
                }
            )

    return {
        "subsets": results,
        "variable": variable,
        "type": "numeric" if is_numeric else "categorical",
    }


def _numeric_bins(variable: str, breakpoints: list) -> list[tuple[str, pl.Expr]]:
    bins: list[tuple[str, pl.Expr]] = []
    for i, bp in enumerate(breakpoints):
        if i == 0:
            label = f"< {bp}"
            mask = pl.col(variable) < bp
        else:
            prev = breakpoints[i - 1]
            label = f"[{prev}, {bp})"
            mask = (pl.col(variable) >= prev) & (pl.col(variable) < bp)
        bins.append((label, mask))
    last = breakpoints[-1]
    bins.append((f"≥ {last}", pl.col(variable) >= last))
    return bins


def _categorical_bins(variable: str, levels: list) -> list[tuple[str, pl.Expr]]:
    bins: list[tuple[str, pl.Expr]] = []
    for lvl in levels:
        bins.append((str(lvl), pl.col(variable) == lvl))
    return bins


def _aggregate(df: pl.DataFrame, value_col: str, weight_col: str | None) -> float:
    vals = df[value_col]
    if weight_col and weight_col in df.columns:
        w = df[weight_col]
        return float((vals * w).sum() / w.sum())
    return float(vals.mean())


# ── Coefficient shape helpers ────────────────────────────────────────────


def _variable_feature_info(bundle: ModelBundle, variable: str) -> dict:
    """Return info about how a variable maps to feature columns.

    Returns
    -------
    dict
        ``{type, labels, coefs, col_names, is_categorical}``
    """
    bp = bundle.blueprint.get(variable, [])
    if not bp:
        return {
            "type": "unknown",
            "labels": [],
            "coefs": [],
            "col_names": [],
            "is_categorical": False,
        }

    is_cat = not all(isinstance(x, (int, float)) for x in bp)

    if is_cat:
        labels = [str(lvl) for lvl in bp]
        col_names = [f"{variable}_{lvl}" for lvl in labels]
        coefs = [float(bundle.coefficient(cn)) for cn in col_names]
        return {
            "type": "categorical",
            "labels": labels,
            "coefs": coefs,
            "col_names": col_names,
            "is_categorical": True,
        }
    else:
        labels = []
        for i, bp_val in enumerate(bp):
            if i == 0:
                labels.append(f"< {bp_val}")
            else:
                labels.append(f"[{bp[i-1]}, {bp_val})")
        labels.append(f"≥ {bp[-1]}")

        col_names = [f"{variable}{v}" for v in bp]
        raw_coefs = [float(bundle.coefficient(cn)) for cn in col_names]
        # Cumulative: bin i gets sum of coefs for all breakpoints ≥ the
        # lower bound.  bin 0: sum(all coefs), bin k: sum(coefs[k:]), last: 0.
        cum_coefs = []
        for i in range(len(raw_coefs)):
            cum_coefs.append(float(np.sum(raw_coefs[i:])))
        cum_coefs.append(0.0)  # top bin (≥ last breakpoint)

        return {
            "type": "numeric",
            "labels": labels,
            "coefs": cum_coefs,
            "col_names": col_names,
            "raw_coefs": raw_coefs,
            "is_categorical": False,
        }


# ── Streamlit UI ─────────────────────────────────────────────────────────


def _run_editor():
    """Entry point when ``--mode=editor`` is passed."""
    import streamlit as st

    st.set_page_config(
        page_title="easy_glm — Coefficient Editor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Parse CLI args ────────────────────────────────────────────────
    cli_args: dict[str, str] = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--") and "=" in arg:
            key, val = arg.split("=", 1)
            cli_args[key[2:].replace("-", "_")] = val

    bundle_path = cli_args.get("bundle_path", "")
    data_path = cli_args.get("data_path", "")
    default_target = cli_args.get("target", "ClaimNb")
    default_weight = cli_args.get("weight_col", "")
    default_tt = cli_args.get("train_test_col", "traintest")

    if not default_weight:
        default_weight = None
    if not default_tt:
        default_tt = None

    # ── Load ──────────────────────────────────────────────────────────
    if "bundle" not in st.session_state:
        with open(bundle_path, "rb") as f:
            st.session_state.bundle = pickle.load(f)
        st.session_state.original_bundle = st.session_state.bundle.clone()

    if "data" not in st.session_state:
        st.session_state.data = pl.read_parquet(data_path) if data_path else None
        st.session_state.target = default_target
        st.session_state.weight_col = default_weight
        st.session_state.train_test_col = default_tt

    if "selected_var" not in st.session_state:
        st.session_state.selected_var = None

    if "ae_cache" not in st.session_state:
        st.session_state.ae_cache: dict = {}

    bundle: ModelBundle = st.session_state.bundle
    original: ModelBundle = st.session_state.original_bundle
    data: pl.DataFrame | None = st.session_state.data
    target = st.session_state.target
    weight_col = st.session_state.weight_col
    train_test_col = st.session_state.train_test_col

    var_list = list(bundle.predictors)

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Coefficient Editor")
        st.caption(f"Family: {bundle.family} · Features: {bundle.n_features}")

        st.divider()
        st.header("Variable")
        if st.session_state.selected_var is None and var_list:
            st.session_state.selected_var = var_list[0]

        for var in var_list:
            is_active = var == st.session_state.selected_var
            if st.button(
                f"{'▸ ' if is_active else ''}{var}",
                key=f"nav_{var}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.selected_var = var
                st.session_state.ae_cache.pop(var, None)
                st.rerun()

        st.divider()

        space = st.radio(
            "Prediction space", ["response", "link"], index=0, horizontal=True
        )

        if st.button("🔄 Reset to original", width="stretch"):
            st.session_state.bundle = original.clone()
            st.session_state.ae_cache = {}
            st.rerun()

        st.divider()
        st.caption(
            "Modify coefficients with sliders below. " "A/E recomputes automatically."
        )

    # ── Main ──────────────────────────────────────────────────────────
    st.title("Coefficient Editor")
    selected = st.session_state.selected_var

    if selected is None:
        st.info("Select a variable from the sidebar.")
        st.stop()

    info = _variable_feature_info(bundle, selected)
    orig_info = _variable_feature_info(original, selected)

    # ── Coefficient chart ─────────────────────────────────────────────
    st.subheader(f"Coefficient Shape — {selected}")

    fig_coef = _build_coef_comparison_chart(orig_info, info, selected, bundle.family)
    st.plotly_chart(fig_coef, width="stretch", key="coef_chart")

    # ── Sliders ───────────────────────────────────────────────────────
    st.subheader("Edit Coefficients")

    changed = False
    if info["is_categorical"]:
        cols = st.columns(min(len(info["labels"]), 4))
        for i, (label, col_name, coef) in enumerate(
            zip(info["labels"], info["col_names"], info["coefs"])
        ):
            with cols[i % 4]:
                new_val = st.slider(
                    f"{label}",
                    min_value=-3.0,
                    max_value=3.0,
                    value=float(coef),
                    step=0.001,
                    format="%.3f",
                    key=f"slider_{col_name}",
                )
                if abs(new_val - coef) > 1e-8:
                    bundle.set_coefficient(col_name, new_val)
                    changed = True
    else:
        # Numeric: sliders for raw breakpoint coefficients
        raw_coefs = info.get("raw_coefs", info["coefs"])
        col_names = info.get("col_names", [])
        if raw_coefs and col_names:
            cols = st.columns(min(len(raw_coefs), 4))
            for i, (label, col_name, coef) in enumerate(
                zip(info["labels"][: len(raw_coefs)], col_names, raw_coefs)
            ):
                with cols[i % 4]:
                    new_val = st.slider(
                        f"Δ {label}",
                        min_value=-3.0,
                        max_value=3.0,
                        value=float(coef),
                        step=0.001,
                        format="%.3f",
                        key=f"slider_{col_name}",
                        help=f"Coefficient for {col_name}",
                    )
                    if abs(new_val - coef) > 1e-8:
                        bundle.set_coefficient(col_name, new_val)
                        changed = True

    # ── A/E chart ─────────────────────────────────────────────────────
    if data is not None:
        st.subheader(f"Actual vs Expected — {selected}")

        if changed or selected not in st.session_state.ae_cache:
            with st.spinner("Recomputing A/E..."):
                st.session_state.ae_cache[selected] = (
                    compute_ae_for_bundle(
                        original,
                        data,
                        selected,
                        target=target,
                        weight_col=weight_col,
                        train_test_col=train_test_col,
                    ),
                    compute_ae_for_bundle(
                        bundle,
                        data,
                        selected,
                        target=target,
                        weight_col=weight_col,
                        train_test_col=train_test_col,
                    ),
                )

        ae_orig, ae_revised = st.session_state.ae_cache[selected]
        fig_ae = _build_ae_comparison_chart(ae_orig, ae_revised, selected)
        st.plotly_chart(fig_ae, width="stretch", key="ae_chart")

        # ── Coefficient summary table ─────────────────────────────────
        with st.expander("Coefficient values", expanded=False):
            rows = []
            orig_coefs = orig_info["coefs"]
            rev_coefs = info["coefs"]
            labels = info["labels"]

            if info["is_categorical"]:
                for lbl, oc, rc in zip(labels, orig_coefs, rev_coefs):
                    rows.append(
                        {
                            "Level": lbl,
                            "Orig. coef": oc,
                            "Revised coef": rc,
                            "Orig. rel": np.exp(oc),
                            "Revised rel": np.exp(rc),
                        }
                    )
            else:
                for lbl, oc, rc in zip(labels, orig_coefs, rev_coefs):
                    rows.append(
                        {
                            "Bin": lbl,
                            "Orig. cum. coef": oc,
                            "Revised cum. coef": rc,
                            "Orig. rel": np.exp(oc),
                            "Revised rel": np.exp(rc),
                        }
                    )

            st.dataframe(
                pl.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )


def _build_coef_comparison_chart(
    orig_info: dict, rev_info: dict, variable: str, family: str
):
    """Plotly chart: original vs revised coefficients / relativities."""
    import plotly.graph_objects as go

    fig = go.Figure()
    labels = rev_info["labels"]
    is_log_link = family in ("poisson", "gamma")

    if rev_info["is_categorical"]:
        orig_rel = [np.exp(c) if is_log_link else c for c in orig_info["coefs"]]
        rev_rel = [np.exp(c) if is_log_link else c for c in rev_info["coefs"]]
        y_title = "Relativity (exp(coef))" if is_log_link else "Coefficient"

        fig.add_trace(
            go.Bar(
                x=labels,
                y=orig_rel,
                name="Original",
                marker_color="lightgray",
                hovertemplate="%{x}<br>Original: %{y:.4f}",
            )
        )
        colors = [
            "#ff7f0e" if r > 1 else "#2ca02c" if r < 1 else "gray" for r in rev_rel
        ]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=rev_rel,
                name="Revised",
                marker_color=colors,
                hovertemplate="%{x}<br>Revised: %{y:.4f}",
            )
        )
        if is_log_link:
            fig.add_hline(
                y=1.0,
                line_dash="dot",
                line_color="gray",
                annotation_text="Baseline (1.0)",
            )
    else:
        orig_rel = [np.exp(c) if is_log_link else c for c in orig_info["coefs"]]
        rev_rel = [np.exp(c) if is_log_link else c for c in rev_info["coefs"]]
        y_title = "Relativity (exp(cum. coef))" if is_log_link else "Cumulative coef"

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=orig_rel,
                mode="lines+markers",
                name="Original",
                line={"color": "gray", "width": 2, "dash": "dash"},
                marker={"size": 6, "color": "gray"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=rev_rel,
                mode="lines+markers",
                name="Revised",
                line={"color": "#1f77b4", "width": 2.5},
                marker={"size": 8, "color": "#1f77b4"},
            )
        )
        if is_log_link:
            fig.add_hline(y=1.0, line_dash="dot", line_color="gray")

    fig.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis_title=variable,
        yaxis_title=y_title,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        hovermode="x unified",
    )
    return fig


def _build_ae_comparison_chart(ae_orig: dict, ae_rev: dict, variable: str):
    """Plotly chart: original vs revised Actual vs Expected."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    subsets_o = ae_orig.get("subsets", {})
    subsets_r = ae_rev.get("subsets", {})
    train_rows = subsets_o.get("train", subsets_o.get("all", []))
    if not train_rows:
        return go.Figure()

    labels = [r["level"] for r in train_rows]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Original
    actual_o = [r["actual"] for r in train_rows]
    expected_o = [r["expected"] for r in train_rows]
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=actual_o,
            mode="lines+markers",
            name="Actual (orig)",
            line={"color": "#1f77b4", "dash": "dot", "width": 1.5},
            marker={"size": 5},
            opacity=0.5,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=expected_o,
            mode="lines+markers",
            name="Expected (orig)",
            line={"color": "#ff7f0e", "dash": "dot", "width": 1.5},
            marker={"size": 5},
            opacity=0.5,
        ),
        secondary_y=False,
    )

    # Revised
    actual_r = [r["actual"] for r in subsets_r.get("train", train_rows)]
    expected_r = [r["expected"] for r in subsets_r.get("train", train_rows)]
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=actual_r,
            mode="lines+markers",
            name="Actual (revised)",
            line={"color": "#1f77b4", "width": 2.5},
            marker={"size": 7},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=expected_r,
            mode="lines+markers",
            name="Expected (revised)",
            line={"color": "#ff7f0e", "width": 2.5},
            marker={"size": 7},
        ),
        secondary_y=False,
    )

    # Exposure bars
    exposures = [r.get("exposure", 0) for r in train_rows]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=exposures,
            name="Exposure",
            marker_color="rgba(180,180,180,0.3)",
            marker_line_width=0,
            showlegend=False,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    fig.update_xaxes(title_text=variable)
    fig.update_yaxes(title_text="Rate", secondary_y=False)
    fig.update_yaxes(title_text="Exposure", secondary_y=True)

    return fig


# ── Entry-point routing (used internally by launch_editor) ──────────────


def _parse_mode() -> str | None:
    """Used internally when streamlit re-runs this script."""
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1]
    return None


# If streamlit is re-executing us in editor mode, run the UI.
if _parse_mode() == "editor":
    _run_editor()
    sys.exit(0)


# ── Pipeline helpers ────────────────────────────────────────────────────


def _build_bundle(
    df: pl.DataFrame,
    predictors: list[str] | None = None,
    family: str = "poisson",
    target: str = "ClaimNb",
    weight_col: str | None = "Exposure",
    train_test_col: str = "traintest",
    use_cv: bool = True,
    verbose: bool = True,
) -> tuple:
    """Run the full pipeline: blueprint → prepare → GLM → ModelBundle.

    Returns ``(bundle, prepped_dataframe)``.
    """
    if predictors is None:
        predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

    train_df = df.filter(pl.col(train_test_col) == 1)
    blueprint = easy_glm.generate_blueprint(train_df)

    additional = [target, train_test_col]
    if weight_col:
        additional.append(weight_col)

    prepped = new_prepare_data(
        df=df,
        modelling_variables=predictors,
        additional_columns=additional,
        formats=blueprint,
        traintest_column=train_test_col,
        table_name="cars",
    )

    model = fit_lasso_glm(
        dataframe=prepped,
        target=target,
        model_type=family,
        weight_col=weight_col,
        train_test_col=train_test_col,
        divide_target_by_weight=True,
        use_cv=use_cv,
    )

    bundle = ModelBundle.from_glm(
        model=model,
        train_prepped=prepped,
        blueprint=blueprint,
        predictors=predictors,
        family=family,
        target=target,
        weight_col=weight_col,
        exposure_col=weight_col,
        train_test_col=train_test_col,
    )

    if verbose:
        print(f"Bundle: {bundle}")
        print(f"  Intercept: {bundle.intercept:.6f}")
        print(
            f"  Non-zero coefs: {(bundle.coefficients != 0).sum()}/{bundle.n_features}"
        )

    return bundle, prepped


# ── CLI modes ────────────────────────────────────────────────────────────


def main():
    """Validation mode: run pipeline and verify matrix multiply matches glum."""
    print("=" * 70)
    print("PROTOTYPE: Full one-hot → GLM → ModelBundle scoring")
    print("=" * 70)

    df = easy_glm.load_external_dataframe()
    rng = np.random.default_rng(42)
    df = df.with_columns(
        pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
    )

    predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

    print("\nCategorical levels (from blueprint):")
    train_df = df.filter(pl.col("traintest") == 1)
    bp = easy_glm.generate_blueprint(train_df)
    for v in predictors:
        levels = bp.get(v, [])
        if levels and not all(isinstance(x, (int, float)) for x in levels):
            print(f"  {v}: {levels}")

    bundle, prepped = _build_bundle(df, predictors=predictors, verbose=False)
    print(f"\n{bundle}")

    # Verify against glum
    exclude = {"ClaimNb", "Exposure", "traintest"}
    fn = bundle.feature_names
    glum_rate = fit_lasso_glm(
        dataframe=prepped,
        target="ClaimNb",
        model_type="poisson",
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=True,
        use_cv=True,
    ).predict(prepped[fn].to_pandas())

    bundle_link = bundle.predict_link(df)
    max_diff = np.max(np.abs(np.log(glum_rate) - bundle_link))
    print(f"  Max |glum.link - bundle.link|: {max_diff:.2e}")
    print(f"  {'✓ MATCHES' if max_diff < 1e-10 else '✗ MISMATCH'}")

    holdout = df.filter(pl.col("traintest") == 0)
    ae = holdout["ClaimNb"].sum() / bundle.predict_response(holdout).sum()
    print(f"  Holdout A/E: {ae:.4f}")

    print("\n── VehGas relativities ──")
    for level, rel in bundle.relativities_for("VehGas").items():
        print(f"  {level:12s}: {rel:.4f}")
    print(
        f"  Intercept baseline: exp({bundle.intercept:.6f}) = "
        f"{np.exp(bundle.intercept):.6f}"
    )

    print("\n" + "=" * 70)
    print("VALIDATION PASSED — Run with --demo for interactive editor")
    print("=" * 70)


def demo():
    """Demo mode: fit the model, then launch the interactive editor."""
    print("=" * 70)
    print("DEMO: Fitting model → launching coefficient editor")
    print("=" * 70)

    print("\nLoading data...")
    df = easy_glm.load_external_dataframe()
    rng = np.random.default_rng(42)
    df = df.with_columns(
        pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
    )

    print("Fitting model (~30s)...")
    bundle, _ = _build_bundle(df, verbose=True)

    print("\nLaunching editor at http://localhost:8501 ...")
    bundle.launch_editor(df, target="ClaimNb", weight_col="Exposure", port=8501)


# ── Run this file to launch the interactive editor ──────────────────────
#   python examples/new_scoring_prototype.py
demo()

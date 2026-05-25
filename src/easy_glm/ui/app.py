"""easy_glm Relativity Editor — Streamlit application.

Launch via :meth:`RateModel.launch_editor` or::

    python -m easy_glm.ui --model-path=model.easyglm
"""

from __future__ import annotations

import os
import tempfile

import polars as pl
import streamlit as st

from easy_glm.engine import RateModel
from easy_glm.engine.models import FromToRow
from easy_glm.ui import _parse_args
from easy_glm.ui.charts import (
    build_actual_vs_expected,
    build_ae_comparison,
    build_histogram,
    build_relativity_chart,
    build_relativity_comparison,
)
from easy_glm.ui.metrics import FORMULAS, compute_actual_expected

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="easy_glm — Relativity Editor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

args = _parse_args()
_MODEL_PATH = args.get("model_path", "model.easyglm")
_DATA_PATH = args.get("data_path")
_TEST_PATH = args.get("test_path")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "baseline_rm" not in st.session_state:
    if os.path.exists(_MODEL_PATH):
        st.session_state.baseline_rm = RateModel.from_json(_MODEL_PATH)
    else:
        st.session_state.baseline_rm = None

if "working_rm" not in st.session_state and st.session_state.baseline_rm is not None:
    st.session_state.working_rm = st.session_state.baseline_rm.clone()

if "saved_models" not in st.session_state:
    st.session_state.saved_models: dict[str, RateModel] = {}

if "data" not in st.session_state:
    if _DATA_PATH and os.path.exists(_DATA_PATH):
        st.session_state.data = pl.read_parquet(_DATA_PATH)
    elif _TEST_PATH and os.path.exists(_TEST_PATH):
        st.session_state.data = pl.read_parquet(_TEST_PATH)
    else:
        st.session_state.data = None

if "test_data" not in st.session_state:
    if _TEST_PATH and os.path.exists(_TEST_PATH):
        st.session_state.test_data = pl.read_parquet(_TEST_PATH)
    else:
        st.session_state.test_data = None

if "selected_var" not in st.session_state:
    st.session_state.selected_var = None

if "ae_baseline" not in st.session_state:
    st.session_state.ae_baseline: dict = {}

if "ae_working" not in st.session_state:
    st.session_state.ae_working: dict = {}

if "auto_recompute" not in st.session_state:
    st.session_state.auto_recompute = True

if "dirty" not in st.session_state:
    st.session_state.dirty = False

if "actual_formula" not in st.session_state:
    st.session_state.actual_formula = "sum_weighted"

# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

baseline = st.session_state.baseline_rm
working = st.session_state.working_rm
data = st.session_state.data

# Merge test data into main data if both exist
if st.session_state.test_data is not None and data is not None:
    data = pl.concat([data, st.session_state.test_data])
    st.session_state.data = data
    st.session_state.test_data = None


def _apply_mapping(dataframe: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    rename: dict[str, str] = {}
    for model_col, dataset_col in mapping.items():
        if dataset_col in dataframe.columns and dataset_col != model_col:
            rename[dataset_col] = model_col
    if rename:
        return dataframe.rename(rename)
    return dataframe


def _row_label(row: FromToRow) -> str:
    if row.from_ is None and row.to_ is None:
        return "Other / Unknown"
    if row.from_ is None:
        return f"< {row.to_}"
    if row.to_ is None:
        return f"≥ {row.from_}"
    if row.from_ == row.to_:
        return str(row.from_)
    return f"[{row.from_}, {row.to_})"


def _compute_ae_for_model(
    rm: RateModel,
    mapped_data: pl.DataFrame,
    variable: str,
    formula: str,
) -> dict | None:
    """Compute A/E for a single variable on a given model."""
    try:
        return compute_actual_expected(rm, mapped_data, variable, formula=formula)
    except Exception:
        return None


def _compute_all_ae() -> None:
    """Recompute A/E for the selected variable on both baseline and working."""
    if st.session_state.data is None:
        return
    variable = st.session_state.selected_var
    if variable is None:
        return
    formula = st.session_state.actual_formula
    mapped = _apply_mapping(
        st.session_state.data, st.session_state.baseline_rm.column_mapping
    )

    if variable not in st.session_state.ae_baseline:
        result = _compute_ae_for_model(
            st.session_state.baseline_rm, mapped, variable, formula
        )
        if result is not None:
            st.session_state.ae_baseline[variable] = result

    if variable not in st.session_state.ae_working:
        result = _compute_ae_for_model(
            st.session_state.working_rm, mapped, variable, formula
        )
        if result is not None:
            st.session_state.ae_working[variable] = result


def _compute_working_ae_for_var(variable: str) -> None:
    """Recompute A/E for *working copy only* on the given variable."""
    if st.session_state.data is None:
        return
    formula = st.session_state.actual_formula
    mapped = _apply_mapping(
        st.session_state.data, st.session_state.working_rm.column_mapping
    )
    result = _compute_ae_for_model(
        st.session_state.working_rm, mapped, variable, formula
    )
    if result is not None:
        st.session_state.ae_working[variable] = result


# ---------------------------------------------------------------------------
# Landing screen (no model loaded)
# ---------------------------------------------------------------------------

if baseline is None:
    st.title("easy_glm — Relativity Editor")
    st.markdown("Upload a model and a dataset to get started.")

    col_a, col_b = st.columns(2)
    with col_a:
        model_file = st.file_uploader("Upload .easyglm model", type=["easyglm", "json"])
        if model_file is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".easyglm", mode="w"
            ) as tf:
                tf.write(model_file.getvalue().decode())
                tf.flush()
                st.session_state.baseline_rm = RateModel.from_json(tf.name)
                st.session_state.working_rm = st.session_state.baseline_rm.clone()
            st.rerun()
    with col_b:
        dataset_file = st.file_uploader(
            "Upload dataset (optional)", type=["parquet", "csv"]
        )
        if dataset_file is not None:
            if dataset_file.name.endswith(".csv"):
                st.session_state.data = pl.read_csv(dataset_file)
            else:
                st.session_state.data = pl.read_parquet(dataset_file)
            st.rerun()

    st.info(
        "You can also launch with a model pre-loaded:\n\n"
        "```python\n"
        "from easy_glm.engine import RateModel\n\n"
        "rm = RateModel.from_json('my_model.easyglm')\n"
        "rm.launch_editor(data=my_dataset)\n"
        "```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Post-load initialisation
# ---------------------------------------------------------------------------

meta = baseline.metadata
non_const = baseline.non_constant_variables
var_names = sorted(non_const.keys())

if st.session_state.selected_var is None and var_names:
    st.session_state.selected_var = var_names[0]

if data is not None:
    dataset_columns = data.columns
else:
    dataset_columns = []

# Ensure column mapping defaults exist
for var in meta.predictor_variables:
    if var not in baseline.column_mapping:
        baseline.column_mapping[var] = var
    if var not in working.column_mapping:
        working.column_mapping[var] = baseline.column_mapping.get(var, var)

# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.title("easy_glm")
    st.caption("Relativity Editor")

    # ── Dataset upload ──────────────────────────────────────────────────────
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload dataset",
        type=["parquet", "csv"],
        key="upload_dataset",
    )
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                st.session_state.data = pl.read_csv(uploaded)
            else:
                st.session_state.data = pl.read_parquet(uploaded)
            data = st.session_state.data
            dataset_columns = data.columns
            st.session_state.ae_baseline = {}
            st.session_state.ae_working = {}
            st.success(f"Loaded: {len(data):,} rows × {len(dataset_columns)} cols")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to load: {exc}")

    if data is not None:
        st.success(f"{len(data):,} rows × {len(dataset_columns)} columns")
    else:
        st.info("Upload a dataset to see A/E")

    st.divider()

    # ── Variable overview ───────────────────────────────────────────────────
    st.header("Variables")
    if not non_const:
        st.info("All variables have constant relativities.")
    else:
        for name in var_names:
            config = non_const[name]
            n_bins = len(config.table)
            is_active = name == st.session_state.selected_var
            label = f"{'▸ ' if is_active else ''}{name} ({n_bins} bins)"
            if st.button(
                label,
                key=f"nav_{name}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.selected_var = name
                # Clear cached A/E when switching variables
                if name not in st.session_state.ae_baseline:
                    st.session_state.ae_baseline.pop(name, None)
                if name not in st.session_state.ae_working:
                    st.session_state.ae_working.pop(name, None)

    st.divider()

    # ── Column Mapping ──────────────────────────────────────────────────────
    with st.expander("Column Mapping", expanded=False):
        st.caption("Map model columns → dataset columns")
        model_cols: list[tuple[str, str]] = []

        if meta.target:
            model_cols.append((meta.target, "Target"))
        if meta.weight_col:
            model_cols.append((meta.weight_col, "Weight / Exposure"))
        if meta.train_test_col:
            model_cols.append((meta.train_test_col, "Train / Test"))

        for var in meta.predictor_variables:
            model_cols.append((var, f"Predictor: {var}"))

        for model_col, label in model_cols:
            default = baseline.column_mapping.get(model_col, model_col)
            options = (
                ["(not mapped)"] + list(dataset_columns)
                if dataset_columns
                else ["(not mapped)"]
            )
            try:
                idx = options.index(default)
            except ValueError:
                idx = 0
            chosen = st.selectbox(
                label,
                options,
                index=idx,
                key=f"map_{model_col}",
            )
            if chosen != "(not mapped)":
                if chosen != baseline.column_mapping.get(model_col):
                    baseline.column_mapping[model_col] = chosen
                if chosen != working.column_mapping.get(model_col):
                    working.column_mapping[model_col] = chosen

    st.divider()

    # ── A/E controls ────────────────────────────────────────────────────────
    st.header("Actual vs Expected")

    formula_key = st.selectbox(
        "Formula",
        list(FORMULAS.keys()),
        format_func=lambda k: FORMULAS[k],
        index=list(FORMULAS.keys()).index(st.session_state.actual_formula),
    )
    st.session_state.actual_formula = formula_key

    st.session_state.auto_recompute = st.checkbox(
        "Auto-recompute on edit", value=st.session_state.auto_recompute
    )

    if not st.session_state.auto_recompute:
        if st.button("🔄 Recompute A/E", type="primary", width="stretch"):
            _compute_all_ae()
    else:
        st.caption("A/E recomputes automatically")

    st.divider()

    # ── Save / Reset ────────────────────────────────────────────────────────
    st.header("Save & Reset")

    save_name = st.text_input("Model name", placeholder="e.g. my_revision_v1")
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        if st.button(
            "💾 Save Working Copy",
            type="primary",
            width="stretch",
            disabled=not save_name.strip(),
        ):
            name = save_name.strip()
            saved = working.clone()
            saved.create_snapshot(f"Saved as '{name}'")
            st.session_state.saved_models[name] = saved
            st.session_state.dirty = False
            st.success(f"Saved as '{name}'")
            st.rerun()
    with col_s2:
        if st.button("📥 Download", width="stretch", disabled=not save_name.strip()):
            name = save_name.strip()
            saved = st.session_state.saved_models.get(name)
            if saved is None:
                saved = working.clone()
            download_path = os.path.join(tempfile.mkdtemp(), f"{name}.easyglm")
            saved.to_json(download_path)
            with open(download_path) as fh:
                st.download_button(
                    label="⬇ .easyglm",
                    data=fh,
                    file_name=f"{name}.easyglm",
                    mime="application/json",
                    width="stretch",
                    key="dl_model",
                )

    if st.button(
        "🔄 Reset Working Copy",
        width="stretch",
        type="secondary",
    ):
        # Pick the reference model
        ref_name = st.session_state.get("compare_ref", None)
        if ref_name and ref_name in st.session_state.saved_models:
            ref = st.session_state.saved_models[ref_name]
        else:
            ref = baseline
        st.session_state.working_rm = ref.clone()
        st.session_state.dirty = False
        st.session_state.ae_working = {}
        st.rerun()

    st.divider()

    # ── Saved models ────────────────────────────────────────────────────────
    if st.session_state.saved_models:
        st.header("Saved Models")
        for name in list(st.session_state.saved_models.keys()):
            col_m1, col_m2 = st.columns([3, 1])
            with col_m1:
                st.caption(f"📦 {name}")
            with col_m2:
                if st.button("🗑", key=f"del_{name}", help=f"Delete '{name}'"):
                    del st.session_state.saved_models[name]
                    st.rerun()

    st.divider()

    # ── Version history from baseline ───────────────────────────────────────
    with st.expander("Version History", expanded=False):
        for s in baseline.snapshots:
            active = "▶ " if s.version == baseline.current_version else ""
            st.caption(
                f"{active}v{s.version}: {s.description[:50]} — {s.timestamp[:19]}"
            )


# ===========================================================================
# MAIN PANEL
# ===========================================================================

st.title("Relativity Editor")
st.caption(
    f"Model: {meta.model_type or 'unknown'} · "
    f"Base rate: {baseline.base_rate:.6f}"
    + (f" · Unsaved changes" if st.session_state.dirty else "")
)

if not var_names:
    st.info("No variables with non-constant relativities to edit.")
    st.stop()

# ── Variable selector ─────────────────────────────────────────────────────
selected = st.selectbox(
    "Variable",
    var_names,
    index=(
        var_names.index(st.session_state.selected_var)
        if st.session_state.selected_var in var_names
        else 0
    ),
    key="var_selector",
)
st.session_state.selected_var = selected

# Ensure A/E is computed if in auto mode
if st.session_state.auto_recompute and data is not None:
    _compute_all_ae()

config_b = baseline.variables[selected]
config_w = working.variables[selected]

# ── Overlaid relativity chart ────────────────────────────────────────────
st.subheader("Relativities")
fig_rel = build_relativity_comparison(config_b, config_w, selected, height=400)
st.plotly_chart(fig_rel, width="stretch", key="rel_overlay")

# ── Overlaid A/E chart ───────────────────────────────────────────────────
if data is not None and selected in st.session_state.ae_baseline:
    st.subheader("Actual vs Expected")
    ae_work = st.session_state.ae_working.get(
        selected, st.session_state.ae_baseline[selected]
    )
    fig_ae = build_ae_comparison(
        st.session_state.ae_baseline[selected],
        ae_work,
        selected,
        height=400,
    )
    st.plotly_chart(fig_ae, width="stretch", key="ae_overlay")
else:
    st.info("Upload a dataset to see actual vs expected.")

# ── Histogram ─────────────────────────────────────────────────────────────
if data is not None:
    mapped = _apply_mapping(data, baseline.column_mapping)
    if selected in mapped.columns:
        with st.expander("Distribution", expanded=False):
            hist_fig = build_histogram(mapped, selected, meta.weight_col, height=250)
            st.plotly_chart(hist_fig, width="stretch")

# ── Editable Table (original + new columns) ──────────────────────────────
st.subheader("Edit Relativities")

rows_data = []
for i, row in enumerate(config_b.table):
    rows_data.append(
        {
            "id": i,
            "label": _row_label(row),
            "original": row.relativity,
            "revised": config_w.table[i].relativity,
        }
    )

edited_df = st.data_editor(
    pl.DataFrame(rows_data),
    column_config={
        "id": st.column_config.NumberColumn("#", disabled=True, width="small"),
        "label": st.column_config.TextColumn("Bin", disabled=True, width="medium"),
        "original": st.column_config.NumberColumn(
            "Original",
            disabled=True,
            format="%.4f",
        ),
        "revised": st.column_config.NumberColumn(
            "Revised",
            min_value=0.0,
            max_value=10.0,
            step=0.01,
            format="%.4f",
        ),
    },
    hide_index=True,
    width="stretch",
    key="data_editor",
    num_rows="fixed",
    disabled=["id", "label", "original"],
)

if edited_df is not None:
    for row in edited_df.iter_rows(named=True):
        idx = row["id"]
        new_rel = row["revised"]
        current_rel = config_w.table[idx].relativity
        if abs(new_rel - current_rel) > 1e-8:
            working.update_relativity(
                selected,
                from_=config_w.table[idx].from_,
                to_=config_w.table[idx].to_,
                new_value=new_rel,
            )
            st.session_state.dirty = True
            st.session_state.ae_working.pop(selected, None)
            if st.session_state.auto_recompute and data is not None:
                _compute_working_ae_for_var(selected)
            st.rerun()

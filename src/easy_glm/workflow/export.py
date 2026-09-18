"""Render a :class:`Project` (and a fitted run) as a standalone Python script."""

from __future__ import annotations

import json
import pprint
from typing import Any

from easy_glm.core.design import (
    CategoricalEncoder,
    DesignSpec,
    InteractionEncoder,
    LinearEncoder,
    StepEncoder,
)
from easy_glm.core.fit import TwoStageFit

from .project import ModelConfig, Project, premium_offset_column
from .run import ModelRun, exposure_for, stage2_alpha


def to_scoring_script(run: ModelRun, *, output_prefix: str | None = None) -> str:
    """Export the exact deployed tables as a standalone, CatBoost-free scorer.

    This is a frozen scoring artefact. :func:`to_script` instead re-runs the
    training workflow, which may select different parameters on changed data.
    """
    payload = json.dumps(run.rate_model.to_dict(), ensure_ascii=False, default=str)
    prefix = output_prefix or run.name
    return "\n".join(
        [
            f'"""Frozen table-only scorer for {run.name!r}.',
            "",
            "This script contains the exact deployed GLM and ordered pair tables.",
            "It scores prepared data and does not train or import CatBoost.",
            '"""',
            "",
            "import json",
            "from easy_glm.engine import RateModel",
            "",
            f"rate_model = RateModel.from_dict(json.loads({payload!r}))",
            "",
            "_MODEL_EXPOSURE = object()",
            "",
            "def predict(data, *, exposure_col=_MODEL_EXPOSURE, column_map=None):",
            '    """Score prepared rows through the frozen, ordered rate tables.',
            "",
            "    Omit exposure_col to use the model's saved exposure setting; pass None",
            '    for unit predictions, as the workbench does for its diagnostics."""',
            "    if exposure_col is _MODEL_EXPOSURE:",
            "        return rate_model.predict(data, column_map=column_map)",
            "    return rate_model.predict(data, exposure_col=exposure_col, column_map=column_map)",
            "",
            "if __name__ == '__main__':",
            f"    rate_model.to_json({prefix + '.easyglm'!r})",
            f"    rate_model.to_excel({prefix + '_rate_tables.xlsx'!r})",
            "",
        ]
    )


def _pair_training_script(project: Project, model: str, prefix: str) -> str:
    """Replay preparation, optional screening, and the complete staged fit."""
    snapshot = pprint.pformat(project.to_dict(), width=88, sort_dicts=False)
    uses_sas = project.data.source.type.lower() == "sas7bdat"
    lines = [
        f'"""{project.name} — model {model!r}: training workflow replay.',
        "",
        "Running this script refits the main GLM and every CatBoost pair stage in",
        "order, then exports their deployed tables. Use the frozen scoring export",
        "when byte-for-byte identical saved tables are required.",
        '"""',
        "",
        "import json",
        "import polars as pl",
    ]
    if uses_sas:
        lines.append("import pandas as pd")
    lines.extend(
        [
            "from easy_glm.workflow import Project, prepare, run_model, totals, train_holdout",
            *(
                ["from easy_glm.workflow.feature_selection import select_variables"]
                if isinstance(project.exploration.get("feature_selection"), dict)
                and isinstance(
                    project.exploration["feature_selection"].get("project"), dict
                )
                else []
            ),
            "",
            "# Saved preparation, Variables bin settings, main fit and ordered pair stages.",
            f"project = Project.from_dict({snapshot})",
            *_load_code(project),
            *_feature_selection_code(project, prefix),
            "prepared = prepare(project, df)",
            f"run = run_model(project, prepared, {model!r}, replay_pair_adjustments=True)",
            "rate_model = run.rate_model",
            f"rate_model.to_json({prefix + '.easyglm'!r})",
            f"rate_model.to_excel({prefix + '_rate_tables.xlsx'!r})",
            "train, holdout = train_holdout(prepared, project.data.split)",
            "if not holdout.is_empty():",
            "    actual, expected, _ = totals(",
            "        holdout, run.config, run.predict(holdout)",
            "    )",
            "    holdout_expected = float(expected.sum())",
            "    holdout_ae = (float(actual.sum()) / holdout_expected",
            "                  if holdout_expected > 0 else float('nan'))",
            "    print('holdout A/E:', holdout_ae)",
            "print('pair stages:', len(run.pair_stages))",
            "",
        ]
    )
    return "\n".join(lines)


def _lit(value: Any) -> str:
    """Python literal for knots / levels / scalars."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value)) if abs(value) < 1e15 else repr(value)
    return repr(value)


def _list(values: list[Any], indent: int = 8, width: int = 88) -> str:
    items = [_lit(v) for v in values]
    one_line = "[" + ", ".join(items) + "]"
    if len(one_line) + indent <= width:
        return one_line
    pad = " " * indent
    lines, cur = [], pad
    for it in items:
        if len(cur) + len(it) + 2 > width:
            lines.append(cur.rstrip())
            cur = pad
        cur += it + ", "
    lines.append(cur.rstrip())
    return "[\n" + "\n".join(lines) + "\n" + " " * (indent - 4) + "]"


def _penalty_arg(enc) -> str:
    """``, penalty_weight=...`` when the variable is not penalised like the rest
    of the design (0 = unpenalised), else nothing."""
    weight = float(getattr(enc, "penalty_weight", 1.0))
    return "" if weight == 1.0 else f", penalty_weight={weight!r}"


def _spec_code(spec: DesignSpec) -> str:
    parts = []
    inter: list[InteractionEncoder] = []
    for var, enc in spec.encoders.items():
        if isinstance(enc, StepEncoder):
            parts.append(
                f"    {var!r}: StepEncoder({var!r}, {_list(enc.knots)}, "
                f"null_indicator={enc.null_indicator}{_penalty_arg(enc)}),"
            )
        elif isinstance(enc, CategoricalEncoder):
            parts.append(
                f"    {var!r}: CategoricalEncoder({var!r}, {_list(enc.levels)}"
                f"{_penalty_arg(enc)}),"
            )
        elif isinstance(enc, LinearEncoder):
            parts.append(
                f"    {var!r}: LinearEncoder({var!r}, {_list(enc.knots)}, "
                f"clamp=({_lit(enc.lo)}, {_lit(enc.hi)}), "
                f"null_indicator={enc.null_indicator}{_penalty_arg(enc)}),"
            )
        elif isinstance(enc, InteractionEncoder):
            inter.append(enc)
        else:
            raise NotImplementedError(f"No script rule for {type(enc).__name__}")
    code = "spec = DesignSpec({\n" + "\n".join(parts) + "\n})"
    for enc in inter:
        cells = ", ".join(f"({i}, {j})" for i, j in enc.cells)
        exposure = ",\n        ".join(_list(row, indent=8) for row in enc.exposure)
        code += (
            f"\n# {enc.variable}: {len(enc.cells)} kept cells of "
            f"{enc.a.n_rows}×{enc.b.n_rows}; the kept cells and training exposure are "
            "written out so the design does not depend on the data\n"
            f"spec.add_interaction(InteractionEncoder(\n"
            f"    spec[{enc.a.variable!r}], spec[{enc.b.variable!r}],\n"
            f"    cells=[{cells}],\n"
            f"    exposure=[\n        {exposure},\n    ],\n"
            f"    min_cell_exposure={enc.min_cell_exposure!r}, "
            f"penalty_weight={enc.penalty_weight!r},\n))"
        )
    return code


def _load_code(project: Project) -> list[str]:
    src = project.data.source
    opts = ", ".join(f"{k}={v!r}" for k, v in src.options.items())
    opts = (", " + opts) if opts else ""
    kind = src.type.lower()
    if kind == "parquet":
        return [f"df = pl.read_parquet({src.path!r}{opts})"]
    if kind == "csv":
        return [f"df = pl.read_csv({src.path!r}{opts})"]
    if kind in ("ipc", "arrow", "feather"):
        return [f"df = pl.read_ipc({src.path!r}{opts})"]
    if kind in ("xlsx", "excel"):
        return [f"df = pl.read_excel({src.path!r}{opts})"]
    if kind == "sas7bdat":
        enc = src.options.get("encoding", "latin-1")
        return [f"df = pl.from_pandas(pd.read_sas({src.path!r}, encoding={enc!r}))"]
    raise ValueError(f"Unsupported source type {src.type!r}")


def _feature_selection_code(project: Project, prefix: str) -> list[str]:
    """Render an optional, editable replay of the saved screening recipe."""
    recipe = project.exploration.get("feature_selection")
    if recipe is None:
        return []
    if not isinstance(recipe, dict) or recipe.get("version") != 1:
        raise ValueError("Saved feature-selection recipe is invalid or unsupported")
    if not isinstance(recipe.get("project"), dict) or not isinstance(
        recipe.get("options"), dict
    ):
        raise ValueError(
            "Saved feature-selection recipe needs project and options objects"
        )
    snapshot = pprint.pformat(recipe["project"], width=88, sort_dicts=False)
    options = recipe.get("options", {})
    options_text = pprint.pformat(options, width=88, sort_dicts=False)
    result_path = f"{prefix}_feature_selection.json"
    return [
        "",
        "# --------------------------------------------------- optional feature screen",
        "# This saved, training-only recipe runs on the raw source, before renames,",
        "# filters and the final model workflow. Changed source data can change it.",
        "# The final reviewed predictors below remain explicit; this screen never",
        "# silently removes a final-model variable.",
        "RUN_FEATURE_SELECTION = True  # set False to skip this when re-running",
        f"selection_options = {options_text}",
        f"screening_project = Project.from_dict({snapshot})",
        "if RUN_FEATURE_SELECTION:",
        "    feature_selection = select_variables(screening_project, df, **selection_options)",
        f"    with open({result_path!r}, 'w', encoding='utf-8') as selection_file:",
        "        json.dump(feature_selection, selection_file, indent=2)",
        "    print('feature selection:', feature_selection['tested_count'], 'tested')",
        "    for item in sorted(feature_selection['rows'],",
        "                       key=lambda row: row['importance'] if row['importance'] is not None else float('-inf'),",
        "                       reverse=True)[:10]:",
        "        print('  ', item['variable'], item['status'], item['importance'])",
        f"    print('feature-selection report:', {result_path!r})",
    ]


def _fit_code(
    cfg: ModelConfig,
    alpha: float | None,
    monotone: dict[str, str],
    chose_by_cv: bool,
    *,
    spec_expr: str = "spec",
    var: str = "fit",
    extra: tuple[str, ...] = (),
    use_offset_col: bool = True,
    call: str = "fit_glm",
    cv_seed: int = 42,
    data_expr: str = "train",
) -> str:
    """One ``fit_glm(...)`` call. ``alpha=None`` re-runs the cross-validation
    the workbench ran; ``extra`` carries the second stage's arguments."""
    args = [
        data_expr,
        spec_expr,
        repr(cfg.target),
        f"family={cfg.family!r}",
    ]
    if cfg.family == "tweedie":
        args.append(f"tweedie_power={float(cfg.tweedie_power)!r}")
    if cfg.link:
        args.append(f"link={cfg.link!r}")
    if cfg.weight:
        args.append(f"weight_col={cfg.weight!r}")
    if cfg.offset and use_offset_col:
        args.append(f"offset_col={cfg.offset!r}")
    if cfg.divide_target_by_weight:
        args.append("divide_target_by_weight=True")
    if alpha is None:
        args.append(
            f"cv={cfg.penalty.cv}, n_alphas={cfg.penalty.n_alphas}, "
            f"cv_seed={cv_seed}"
        )
    else:
        args.append(f"alpha={alpha!r}")
    if cfg.penalty.l1_ratio != 1.0:
        args.append(f"l1_ratio={cfg.penalty.l1_ratio!r}")
    if monotone:
        args.append(f"monotone={monotone!r}")
    args.extend(extra)
    comment = ""
    if chose_by_cv:
        comment = (
            f"# alpha was chosen by {cfg.penalty.cv}-fold CV over {cfg.penalty.n_alphas} "
            "alphas in the workbench; it is written out so this script is deterministic.\n"
        )
    return comment + f"{var} = {call}(\n    " + ",\n    ".join(args) + ",\n)"


def _two_stage_code(
    cfg: ModelConfig,
    alpha: float | None,
    alpha2: float | None,
    monotone: dict[str, str],
    chose_by_cv: bool,
    stage2_chose_by_cv: bool,
    cv_seed: int,
) -> list[str]:
    """The two stages of a model with interactions, written out in full."""
    lines = [
        "# Stage 1 — the main effects on their own. This is exactly the fit this",
        "# model would get with no interaction at all, and the rate tables and base",
        "# rate below are read off it, so adding an interaction never moves them.",
        _fit_code(
            cfg,
            alpha,
            monotone,
            chose_by_cv,
            spec_expr="spec.main_effects_spec()",
            var="stage1",
            cv_seed=cv_seed,
        ),
        "",
        "# Stage 2 — the interaction cells as pure adjustments on top of stage 1:",
        "# no intercept, and a stage-1 linear predictor as the offset. A cell",
        "# coefficient of 0 (relativity 1.00) means 'no adjustment'.",
    ]
    if stage2_chose_by_cv:
        lines += [
            "# CV selected the cells' alpha using out-of-fold main predictions.",
            "# Fit the final cells against the final frozen main effects.",
        ]
    eta1 = "eta1 = stage1.linear_predictor(train)"
    if cfg.offset:
        # Match fit_glm's cast, and include the external offset exactly once.
        eta1 += f" + train[{cfg.offset!r}].cast(pl.Float64).to_numpy()"
    lines.append(eta1)
    lines += [
        _fit_code(
            cfg,
            alpha2,
            {},
            stage2_chose_by_cv,
            spec_expr="spec.interactions_spec()",
            var="stage2",
            extra=(
                "offset=eta1",
                "fit_intercept=False",
                # glum cannot standardise without an intercept; the cell penalty
                # rule is the same either way (see core.fit.penalty_weights)
                "scale_predictors=False",
            ),
            use_offset_col=False,
            cv_seed=cv_seed,
        ),
        "fit = TwoStageFit(stage1, stage2)",
    ]
    return lines


def to_script(
    project: Project,
    model: str | None = None,
    *,
    run: ModelRun | None = None,
    output_prefix: str | None = None,
) -> str:
    """Python source reproducing ``project.models[model]`` with the public API.

    With a fitted ``run`` the design is written out explicitly (every knot and
    level) and the resolved alpha is used, so the script is self-contained and
    deterministic. Without a run the design is derived from the data at run
    time and CV (if configured) is re-run.
    """
    model = model or project.champion or next(iter(project.models), None)
    if model is None or model not in project.models:
        raise ValueError("No model to export")
    cfg = project.models[model]
    d = project.data
    if not d.source.path.strip():
        raise ValueError(
            "Save the source data to a file and select it before exporting a Python script."
        )
    prefix = output_prefix or model
    if cfg.pair_stages:
        return _pair_training_script(project, model, prefix)
    uses_sas = d.source.type.lower() == "sas7bdat"
    # Whether there really were two stages is a property of the *fit*, not of the
    # encoders: an interaction whose every cell is below the exposure floor has an
    # encoder but no columns, so ``fit_two_stage`` returns a plain ``GLMFit`` and
    # a stage-2 block would be a fit on a zero-column design. Without a run the
    # answer is not knowable at export time (the cells are decided from the data
    # when the script runs), so that branch calls ``fit_two_stage``, which makes
    # the same decision at run time.
    two_stage = isinstance(run.fit, TwoStageFit) if run is not None else False
    derive_stages = run is None and bool(cfg.interactions)

    lines: list[str] = [
        f'"""{project.name} — model {model!r}: generated by easy_glm workbench.',
        "",
        "Re-run this file to rebuild the model, its rate tables and the .easyglm scorer.",
        '"""',
        "",
        "import numpy as np",
        "import polars as pl",
        "import json",
    ]
    if uses_sas:
        lines.append("import pandas as pd")
    lines += [
        "",
        "from easy_glm import (",
        "    CategoricalEncoder,",
        "    DesignSpec,",
        "    InteractionEncoder,",
        "    LinearEncoder,",
        "    StepEncoder,",
        *(["    TwoStageFit,"] if two_stage else []),
        "    fit_glm,",
        *(["    fit_two_stage,"] if derive_stages else []),
        "    to_rate_model,",
        ")",
        "from easy_glm.workflow import build_design",
        "from easy_glm.workflow.project import Interaction, Project",
        *(
            ["from easy_glm.workflow.feature_selection import select_variables"]
            if isinstance(project.exploration.get("feature_selection"), dict)
            and isinstance(
                project.exploration["feature_selection"].get("project"), dict
            )
            else []
        ),
        "",
        "# ---------------------------------------------------------------- 1. data",
        *_load_code(project),
        *_feature_selection_code(project, prefix),
    ]
    if d.renames:
        lines.append(f"df = df.rename({d.renames!r})")
    for col, rc in d.recodes.items():
        default = (
            f"pl.col({col!r}).cast(pl.Utf8)" if rc.default is None else repr(rc.default)
        )
        if rc.mapping:
            lines += [
                "df = df.with_columns(",
                f"    pl.col({col!r}).cast(pl.Utf8).replace_strict(",
                f"        {rc.mapping!r},",
                f"        default={default}, return_dtype=pl.Utf8,",
                f"    ).alias({col!r})",
                ")",
            ]
        else:
            lines.append(f"df = df.with_columns(pl.col({col!r}).cast(pl.Utf8))")
    for col, kind in d.types.items():
        cast = "pl.Utf8" if kind == "categorical" else "pl.Float64, strict=False"
        lines.append(f"df = df.with_columns(pl.col({col!r}).cast({cast}))")
    for der in d.derived:
        lines.append(f"df = df.with_columns(({der.expr}).alias({der.name!r}))")
    for f in d.filters:
        lines.append(f"df = df.filter({f})")
    if (premium := project.current_premium) is not None:
        # exactly what workflow.prep.add_premium_offset does, written out so the
        # rate-change setup is visible rather than implied by a role
        lines += [
            f"# {premium} is the premium charged today; its log is the model's",
            "# offset, so the base rate is the overall rate change and every",
            "# relativity is a multiplier on the current premium",
            f"df = df.with_columns(pl.col({premium!r}).cast(pl.Float64).log()"
            f".alias({premium_offset_column(premium)!r}))",
        ]

    split = d.split
    lines += [
        "",
        "# --------------------------------------------------------------- 2. split",
    ]
    if split.mode == "random":
        lines += [
            f"is_train = np.random.default_rng({split.seed}).random(df.height) < {split.fraction}",
            f"df = df.with_columns(pl.Series({split.column!r}, is_train.astype(np.int64)))",
        ]
    elif split.holdout_value is not None:
        lines += [
            "from easy_glm.workflow import Split, add_split_column",
            f"split = Split(mode='column', column={split.column!r}, "
            f"train_value={split.train_value!r}, holdout_value={split.holdout_value!r})",
            "df = add_split_column(df, split)",
        ]
    else:
        lines.append(
            f"df = df.with_columns((pl.col({split.column!r}) == {split.train_value!r}).cast(pl.Int64).alias({split.column!r}))"
        )
    lines += [
        f"train = df.filter(pl.col({split.column!r}) == 1)",
        f"holdout = df.filter(pl.col({split.column!r}) == 0)",
        "",
        "# -------------------------------------------------------------- 3. design",
    ]
    ignored = project.exploration.get("leakage", {}).get("ignored", [])
    if ignored:
        lines.append(f"# excluded after the leakage review: {', '.join(ignored)}")
    alpha: float | None
    if run is not None:
        lines.append(
            f"# final reviewed predictors: {cfg.predictors!r}; retained explicitly"
        )
        lines.append(_spec_code(run.spec))
        alpha = run.fit.alpha
        alpha2 = run.alpha_stage2
        chose_by_cv = cfg.penalty.alpha is None
        stage2_chose_by_cv = chose_by_cv and stage2_alpha(cfg) is None
        monotone = dict(run.fit.monotone)
    else:
        # build_design is the workflow's one source of truth for per-variable
        # n_bins, integer cuts, levels, null handling, clamps and interactions.
        design_snapshot = project.to_dict()
        design_snapshot["models"] = {}
        design_snapshot["champion"] = None
        design_snapshot["exploration"] = {}
        design_text = pprint.pformat(design_snapshot, width=88, sort_dicts=False)
        lines += [
            "# Rebuild the final design from its saved per-variable settings.",
            f"final_design_project = Project.from_dict({design_text})",
            f"predictors = {cfg.predictors!r}  # final reviewed selection; retained explicitly",
            "spec = build_design(",
            "    final_design_project, train, predictors,",
            f"    weight_col={cfg.weight!r}, interactions={cfg.interactions!r},",
            ")",
        ]
        alpha = cfg.penalty.alpha
        # stage 2 follows stage 1 unless an interaction asked for its own alpha
        # None here means "follow the mains", which is fit_two_stage's own default
        alpha2 = stage2_alpha(cfg) if derive_stages else None
        chose_by_cv = False
        stage2_chose_by_cv = False
        monotone = {
            **{
                v: vd.monotone
                for v, vd in project.design.variables.items()
                if vd.monotone
            },
            **cfg.monotone,
        }
    lines += [
        "",
        "# ----------------------------------------------------------------- 4. fit",
    ]
    alpha = None if alpha is None else float(alpha)
    if two_stage:
        lines += _two_stage_code(
            cfg,
            alpha,
            alpha2,
            monotone,
            chose_by_cv,
            stage2_chose_by_cv,
            split.seed,
        )
    elif derive_stages:
        lines += [
            "# The mains are fitted first and frozen; the interaction cells are then",
            "# fitted on top of them (no intercept, stage 1's linear predictor as the",
            "# offset), so adding an interaction never moves a main-effect table.",
            "# fit_two_stage decides here, from this data, whether any cell has enough",
            "# exposure to be rated on its own; if none has, there is no second stage.",
            "# With CV, its second stage independently evaluates the same folds,",
            "# shuffled with the project seed, and alpha-path length on the cells (it never",
            "# reuses the mains alpha). Its offset uses out-of-fold stage-1 predictions.",
            _fit_code(
                cfg,
                alpha,
                monotone,
                chose_by_cv,
                call="fit_two_stage",
                extra=() if alpha2 is None else (f"stage2_alpha={alpha2!r}",),
                cv_seed=split.seed,
            ),
        ]
    else:
        lines.append(_fit_code(cfg, alpha, monotone, chose_by_cv, cv_seed=split.seed))
    lines += ["print(fit)", "print(fit.coef_table(drop_zero=True))"]

    exposure = exposure_for(project, cfg)
    lines += [
        "",
        "# --------------------------------------------------- 5. rate tables & scorer",
        "rm = to_rate_model(",
        "    fit,",
        f"    base={cfg.base!r},",
        f"    base_rate_override={cfg.base_rate_override!r},",
        f"    exposure_col={exposure!r},",
        f"    train_test_col={split.column!r},",
        f"    model_type={cfg.family!r},",
        *(
            ["    offset_is_premium=True,  # tables are multipliers on the premium"]
            if project.current_premium
            and cfg.offset == premium_offset_column(project.current_premium)
            else []
        ),
        ")",
    ]
    if cfg.adjustments:
        lines.append("# manual adjustments made in the relativity editor")
        for adj in cfg.adjustments:
            if adj.cell:
                lines.append(
                    f"rm.update_relativity({adj.variable!r}, {adj.from_!r}, {adj.to_!r}, "
                    f"{float(adj.relativity)!r}, from_b={adj.from_b!r}, to_b={adj.to_b!r})"
                )
            else:
                lines.append(
                    f"rm.update_relativity({adj.variable!r}, {adj.from_!r}, {adj.to_!r}, {float(adj.relativity)!r})"
                )
        lines.append(
            f'rm.create_snapshot("{len(cfg.adjustments)} manual adjustment(s)")'
        )
    lines += [
        "",
        f"rm.to_json({prefix + '.easyglm'!r})",
        f"rm.to_excel({prefix + '_rate_tables.xlsx'!r})  # adjusted tables, as scored",
        "",
        "# Compare actual and expected totals on the model's weighting basis.",
        "holdout_pred = rm.predict(holdout, exposure_col=None)",
        f"holdout_actual = holdout[{cfg.target!r}].cast(pl.Float64).to_numpy()",
    ]
    if cfg.weight:
        lines.append(
            f"holdout_weight = holdout[{cfg.weight!r}].cast(pl.Float64).to_numpy()"
        )
        if not cfg.divide_target_by_weight:
            lines.append("holdout_actual = holdout_actual * holdout_weight")
        lines.append("holdout_pred = holdout_pred * holdout_weight")
    lines += [
        "holdout_expected = float(holdout_pred.sum())",
        "holdout_ae = (float(holdout_actual.sum()) / holdout_expected",
        "              if holdout_expected > 0 else float('nan'))",
        "print('holdout A/E:', holdout_ae)",
        "",
    ]
    return "\n".join(lines)

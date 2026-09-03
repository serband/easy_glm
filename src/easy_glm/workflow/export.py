"""Render a :class:`Project` (and a fitted run) as a standalone Python script."""

from __future__ import annotations

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
) -> str:
    """One ``fit_glm(...)`` call. ``alpha=None`` re-runs the cross-validation
    the workbench ran; ``extra`` carries the second stage's arguments."""
    args = [
        "train",
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
        args.append(f"cv={cfg.penalty.cv}, n_alphas={cfg.penalty.n_alphas}")
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
) -> list[str]:
    """The two stages of a model with interactions, written out in full."""
    eta1 = "eta1 = stage1.linear_predictor(train)"
    if cfg.offset:
        # the same cast fit_glm applies, so an Int64 or Float32 offset column
        # takes the same path here as it does in the workbench
        eta1 += f" + train[{cfg.offset!r}].cast(pl.Float64).to_numpy()"
    return [
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
        ),
        "",
        "# Stage 2 — the interaction cells as pure adjustments on top of stage 1:",
        "# no intercept, and stage 1's linear predictor as the offset. A cell",
        "# coefficient of 0 (relativity 1.00) means 'no adjustment'.",
        eta1,
        _fit_code(
            cfg,
            alpha2,
            {},
            False,
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
        ),
        "fit = TwoStageFit(stage1, stage2)",
    ]


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
    prefix = output_prefix or model
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
        "",
        "# ---------------------------------------------------------------- 1. data",
        *_load_code(project),
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
        lines.append(_spec_code(run.spec))
        alpha = run.fit.alpha
        alpha2 = run.alpha_stage2
        chose_by_cv = cfg.penalty.alpha is None
        monotone = dict(run.fit.monotone)
    else:
        dd = project.design.defaults
        lines += [
            "# (fit in the workbench to have every knot and level written out explicitly)",
            "spec = DesignSpec.from_data(",
            f"    train, {cfg.predictors!r},",
            f"    n_bins={dd.n_bins}, min_level_share={dd.min_level_share}, null_indicator={dd.null_indicator},",
            f"    weight_col={cfg.weight!r},",
            *(
                # "continuous" is a linear term with no interior knots
                [f"    linear={linear_vars!r},"]
                if (
                    linear_vars := [
                        v
                        for v, vd in project.design.variables.items()
                        if vd.kind in ("linear", "continuous") and v in cfg.predictors
                    ]
                )
                else []
            ),
            *(
                [f"    knots={explicit_knots!r},"]
                if (
                    explicit_knots := {
                        v: (
                            []
                            if vd.kind == "continuous"
                            else [float(k) for k in vd.knots]
                        )
                        for v, vd in project.design.variables.items()
                        if (vd.kind == "continuous" or isinstance(vd.knots, list))
                        and v in cfg.predictors
                    }
                )
                else []
            ),
            *(
                [f"    penalty_weight={pweights!r},"]
                if (
                    pweights := {
                        v: float(vd.penalty_weight)
                        for v, vd in project.design.variables.items()
                        if float(vd.penalty_weight) != 1.0 and v in cfg.predictors
                    }
                )
                else []
            ),
            *(
                [f"    clamp={clamps!r},"]
                if (
                    clamps := {
                        v: (float(vd.clamp[0]), float(vd.clamp[1]))
                        for v, vd in project.design.variables.items()
                        if vd.kind in ("linear", "continuous")
                        and vd.clamp
                        and v in cfg.predictors
                    }
                )
                else []
            ),
            ")",
        ]
        # the interactions are added one by one, not through from_data's single
        # `interactions=`, so each keeps its own cell floor and penalty weight
        weights = f"train[{cfg.weight!r}]" if cfg.weight else "None"
        for it in cfg.interactions:
            lines += [
                "spec.add_interaction(InteractionEncoder.from_data(",
                f"    spec[{it.a!r}], spec[{it.b!r}], train, weights={weights},",
                f"    min_cell_exposure={it.min_cell_exposure!r}, "
                f"penalty_weight={it.penalty_weight!r},",
                "))",
            ]
        alpha = cfg.penalty.alpha
        # stage 2 follows stage 1 unless an interaction asked for its own alpha
        # None here means "follow the mains", which is fit_two_stage's own default
        alpha2 = stage2_alpha(cfg) if derive_stages else None
        chose_by_cv = False
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
        lines += _two_stage_code(cfg, alpha, alpha2, monotone, chose_by_cv)
    elif derive_stages:
        lines += [
            "# The mains are fitted first and frozen; the interaction cells are then",
            "# fitted on top of them (no intercept, stage 1's linear predictor as the",
            "# offset), so adding an interaction never moves a main-effect table.",
            "# fit_two_stage decides here, from this data, whether any cell has enough",
            "# exposure to be rated on its own; if none has, there is no second stage.",
            _fit_code(
                cfg,
                alpha,
                monotone,
                chose_by_cv,
                call="fit_two_stage",
                extra=() if alpha2 is None else (f"stage2_alpha={alpha2!r}",),
            ),
        ]
    else:
        lines.append(_fit_code(cfg, alpha, monotone, chose_by_cv))
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
        "holdout_pred = rm.predict(holdout)",
        f"print('holdout A/E:', holdout[{cfg.target!r}].sum() / holdout_pred.sum())",
        "",
    ]
    return "\n".join(lines)

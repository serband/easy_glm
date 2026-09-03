#!/usr/bin/env python
"""Scale spike for easy_glm 0.4, workstream G: how to build/fit 1M-5M-row designs.

Re-runnable. Every candidate runs in a fresh subprocess so peak RSS
(``ru_maxrss``) is attributable. A watchdog kills a child whose RSS exceeds
60% of physical RAM.

    python bench.py                       # default plan (200k, 1M, 5M + French motor)
    python bench.py --sizes 200000        # quick
    python bench.py --sizes 1000000 --candidates baseline split32 agg
    python bench.py --run split32 --n 1000000   # (child mode, prints RESULT_JSON:{...})

Results accumulate in results.json (one record per (candidate, n)); the table
is printed at the end and can be re-printed with ``--table-only``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

SPIKE = Path(__file__).resolve().parent
OUT = SPIKE / "out"
PY = "/Users/serban/Documents/Projects/easy_glm/.venv/bin/python"
RESULTS = SPIKE / "results.json"
FRENCH = Path.home() / ".cache" / "easy_glm"

ALPHA = 3e-4
N_BINS = 30
MEM_TOTAL = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode())
MEM_CAP = 0.60 * MEM_TOTAL

ALL_CANDIDATES = [
    "baseline",  # current easy_glm path: spec.build() dense float64 numpy + glum
    "baseline_tight",  # same, gradient_tol=1e-8 (solver noise floor)
    "dense64c",  # dense float64 written straight from bin codes (lean builder)
    "dense32",  # dense float32 numpy
    "split64",  # tabmat SplitMatrix: CategoricalMatrix blocks + dense f64 step block
    "split32",  # same, float32
    "agg",  # rows aggregated by identical design row, dense f64
    "agg_split32",  # aggregated + SplitMatrix f32
    "stepmat64",  # StepMatrix prototype blocks (cumsum trick) + Categorical, f64
    "stepmat32",  # same, float32
    "baseline_cv",  # 3-fold CV on dense f64
    "split32_cv",  # 3-fold CV on SplitMatrix f32
    "split64_cv",  # 3-fold CV on SplitMatrix f64
    "stepmat64_cv",  # 3-fold CV on StepMatrix f64
    "dense32_tight",  # float32 variants at gradient_tol=1e-8, compared with baseline_tight
    "split32_tight",
    "stepmat32_tight",
]

DEFAULT_PLAN = {
    200_000: [c for c in ALL_CANDIDATES if not c.endswith("_cv")],
    1_000_000: [c for c in ALL_CANDIDATES if c != "split32_cv"],  # split32_cv: 374 s, non-convergent (see report)
    5_000_000: ["baseline", "dense32", "split64", "split32", "agg_split32", "stepmat64", "stepmat32"],
}


# ==========================================================================
# child side
# ==========================================================================
def _rss() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def _start_watchdog(result: dict) -> None:
    def run():
        peak = 0
        while True:
            r = _rss()
            peak = max(peak, r)
            result["rss_watchdog_peak"] = peak
            if r > MEM_CAP:
                result["status"] = "killed_mem_cap"
                result["error"] = f"RSS {r / 2**30:.2f} GB > cap {MEM_CAP / 2**30:.2f} GB"
                _emit(result)
                os._exit(3)
            time.sleep(0.2)

    threading.Thread(target=run, daemon=True).start()


def _emit(result: dict) -> None:
    result["ru_maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    print("RESULT_JSON:" + json.dumps(result, default=str), flush=True)


def _data(n: int):
    import polars as pl

    import spike_lib as L

    OUT.mkdir(exist_ok=True)
    path = OUT / f"data_{n}.parquet"
    if not path.exists():
        L.make_data(n).write_parquet(path)
    return pl.read_parquet(path)


def _spec(n: int, df):
    from easy_glm.core.design import DesignSpec

    import spike_lib as L

    path = OUT / f"spec_{n}.json"
    if path.exists():
        return DesignSpec.from_json(path)
    spec = DesignSpec.from_data(df, L.PREDICTORS, n_bins=N_BINS, weight_col="exposure")
    spec.to_json(path)
    return spec


def _fit(X, y, w, *, cv: bool = False, **kw):
    from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV

    common = dict(family="poisson", link="log", l1_ratio=1.0, fit_intercept=True, scale_predictors=True)
    common.update(kw)
    if cv:
        m = GeneralizedLinearRegressorCV(cv=3, n_alphas=10, min_alpha_ratio=1e-3, **common)
    else:
        m = GeneralizedLinearRegressor(alpha=ALPHA, **common)
    m.fit(X, y, sample_weight=w)
    return m


def _nbytes(X) -> int:
    import numpy as np
    import tabmat as tm

    import spike_lib as L

    if isinstance(X, np.ndarray):
        return X.nbytes
    if isinstance(X, tm.DenseMatrix):
        return X._array.nbytes
    if isinstance(X, tm.CategoricalMatrix):
        return X.indices.nbytes
    if isinstance(X, L.StepMatrix):
        return X.code.nbytes
    if isinstance(X, tm.SplitMatrix):
        return sum(_nbytes(m) for m in X.matrices)
    return -1


def run_candidate(cand: str, n: int) -> dict:
    import numpy as np

    import spike_lib as L

    res: dict = {"candidate": cand, "n": n, "status": "ok", "notes": [], "alpha": ALPHA}
    _start_watchdog(res)
    t0 = time.time()
    df = _data(n)
    spec = _spec(n, df)
    res["p"] = spec.n_features
    y = (df["claims"] / df["exposure"]).to_numpy()
    w = df["exposure"].to_numpy()
    res["t_load"] = time.time() - t0
    res["rss_after_load"] = _rss()

    cv = cand.endswith("_cv")
    base = cand[:-3] if cv else cand
    fit_kw = {}
    tight = base.endswith("_tight") and base != "baseline_tight"
    if tight:
        base = base[: -len("_tight")]
        fit_kw["gradient_tol"] = 1e-8
        res["notes"].append("gradient_tol=1e-8; compared with baseline_tight")
    codes = None
    group_of_row = None
    n_fit_rows = n
    tb = time.time()

    # ---- build ------------------------------------------------------------
    if base == "baseline" or base == "baseline_tight":
        X = spec.build(df)  # the current easy_glm path
        if base == "baseline_tight":
            fit_kw["gradient_tol"] = 1e-8
    else:
        codes = L.var_codes(spec, df)
        if base == "dense64c":
            X = L.build_dense_from_codes(spec, codes, np.float64)
        elif base == "dense32":
            X = L.build_dense_from_codes(spec, codes, np.float32)
        elif base == "split64":
            X = L.build_split(spec, codes, np.float64)
        elif base == "split32":
            X = L.build_split(spec, codes, np.float32)
        elif base in ("agg", "agg_split32"):
            codes_agg, ybar, W, group_of_row = L.aggregate(spec, codes, y, w)
            n_fit_rows = len(W)
            res["n_groups"] = n_fit_rows
            res["compression_ratio"] = n / n_fit_rows
            y_fit, w_fit = ybar, W
            if base == "agg":
                X = L.build_dense_from_codes(spec, codes_agg, np.float64)
            else:
                X = L.build_split(spec, codes_agg, np.float32)
        elif base in ("stepmat64", "stepmat32"):
            L.patch_glum_validation()
            X = L.build_split_stepmatrix(spec, codes, np.float64 if base == "stepmat64" else np.float32)
        else:
            raise ValueError(cand)
    if group_of_row is None:
        y_fit, w_fit = y, w
    if np.dtype(X.dtype) == np.float32:
        # glum 3.4.1 casts y to X.dtype but leaves float64 weights as they are
        # (sklearn check_array with dtype=[f64, f32]); tabmat's Cython kernels
        # then get mixed dtypes: CategoricalMatrix raises, DenseMatrix segfaults.
        y_fit = y_fit.astype(np.float32)
        w_fit = w_fit.astype(np.float32)
        res["notes"].append("y and sample_weight cast to float32 before glum (required, see probe)")
    res["t_build"] = time.time() - tb
    res["design_bytes"] = _nbytes(X)
    res["design_dtype"] = str(X.dtype)
    res["rss_after_build"] = _rss()

    # ---- fit --------------------------------------------------------------
    tf = time.time()
    try:
        model = _fit(X, y_fit, w_fit, cv=cv, **fit_kw)
    except Exception as exc:  # noqa: BLE001
        res["status"] = "fit_failed"
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-1500:]
        res["t_fit"] = time.time() - tf
        res["t_total"] = time.time() - t0
        return res
    res["t_fit"] = time.time() - tf
    res["rss_after_fit"] = _rss()
    coef = np.asarray(model.coef_)
    res["coef_dtype"] = str(model.coef_.dtype)
    res["intercept_dtype"] = str(np.asarray(model.intercept_).dtype)
    res["n_iter"] = int(getattr(model, "n_iter_", -1))
    res["nnz"] = int((coef != 0).sum())
    if cv:
        res["alpha"] = float(model.alpha_)
        res["notes"].append("CV: alpha chosen by CV, predictions not comparable to fixed-alpha baseline")

    # ---- exactness check: glum's own predict on this X vs float64 recomposition --
    tp = time.time()
    p_glum = np.asarray(model.predict(X), dtype=np.float64)
    if codes is None:
        codes = L.var_codes(spec, df)
    if group_of_row is not None:
        p_glum = p_glum[group_of_row]
    pred = L.predict_from_codes(spec, codes, coef.astype(np.float64), float(model.intercept_))
    res["t_predict"] = time.time() - tp
    res["glum_pred_vs_f64_recomposed_max_rel"] = float(np.max(np.abs(p_glum - pred) / pred))
    del X

    # ---- compare with baseline ------------------------------------------------
    np.save(OUT / f"pred_{cand}_{n}.npy", pred)
    np.save(OUT / f"coef_{cand}_{n}.npy", np.concatenate([[model.intercept_], coef]).astype(np.float64))
    ref = "baseline_tight" if tight else "baseline"
    bpath = OUT / f"pred_{ref}_{n}.npy"
    bcoef = OUT / f"coef_{ref}_{n}.npy"
    if cand != "baseline" and bpath.exists():
        p0 = np.load(bpath)
        rel = np.abs(pred - p0) / p0
        res["max_rel_pred_diff"] = float(rel.max())
        res["p99_rel_pred_diff"] = float(np.quantile(rel, 0.99))
        c0 = np.load(bcoef)
        res["max_abs_coef_diff"] = float(np.abs(c0[1:] - coef).max())
        res["nnz_set_diff"] = int(((c0[1:] != 0) != (coef != 0)).sum())
        res["nnz_baseline"] = int((c0[1:] != 0).sum())

    # ---- dtype side-tests (cheap, small n only) ----------------------------
    if base in ("dense32", "split32") and n <= 200_000:
        probe = subprocess.run(
            [PY, str(SPIKE / "bench.py"), "--run", f"probe_f64_weights_{base}", "--n", str(n)],
            capture_output=True, text=True, timeout=600, env=dict(os.environ, PYTHONPATH=str(SPIKE)),
        )
        tail = (probe.stdout + probe.stderr).strip().splitlines()[-1:] 
        res["notes"].append(
            f"probe: float32 X with float64 sample_weight -> returncode {probe.returncode}"
            f"{' (SIGSEGV)' if probe.returncode in (139, -11) else ''}: {tail[0][:160] if tail else ''}"
        )
    if base == "dense32" and n <= 200_000:
        Xs = L.build_dense_from_codes(spec, codes, np.float32)
        for label, kw in [
            ("P1_float64", dict(P1=np.ones(spec.n_features))),
            ("offset_float64", dict()),
            ("lower_bounds_float64", dict(lower_bounds=np.full(spec.n_features, -np.inf))),
        ]:
            try:
                from glum import GeneralizedLinearRegressor

                m = GeneralizedLinearRegressor(family="poisson", alpha=ALPHA, l1_ratio=1.0, max_iter=2, **kw)
                if label == "offset_float64":
                    m.fit(Xs, y.astype(np.float32), sample_weight=w.astype(np.float32), offset=np.zeros(n))
                else:
                    m.fit(Xs, y.astype(np.float32), sample_weight=w.astype(np.float32))
                res["notes"].append(f"{label} with float32 X: OK (coef {m.coef_.dtype})")
            except Exception as exc:  # noqa: BLE001
                res["notes"].append(f"{label} with float32 X: {type(exc).__name__}: {str(exc)[:160]}")
        del Xs
    if base == "split32" and n <= 200_000:
        import tabmat as tm

        try:
            dense, idx = L.step_block_dense(spec, codes, np.float32)
            blocks = L.cat_blocks(spec, codes, np.float64)
            tm.SplitMatrix([tm.DenseMatrix(dense), blocks[0][0]], [np.asarray(idx), np.asarray(blocks[0][1])])
            res["notes"].append("mixed f32 dense + f64 categorical SplitMatrix: constructed without error")
        except Exception as exc:  # noqa: BLE001
            res["notes"].append(f"mixed f32 dense + f64 categorical SplitMatrix: {type(exc).__name__}: {str(exc)[:160]}")

    res["t_total"] = time.time() - t0
    return res


def run_french(n_bins: int = 20) -> dict:
    """Row-level vs aggregated fit on the cached French motor set; compression ratio."""
    import numpy as np
    import polars as pl
    from easy_glm.core.design import DesignSpec

    import spike_lib as L

    res: dict = {"candidate": "french_agg", "status": "ok", "notes": [], "alpha": ALPHA}
    _start_watchdog(res)
    t0 = time.time()
    files = sorted(FRENCH.glob("*.parquet"))
    if not files:
        res["status"] = "skipped"
        res["error"] = "no cached French motor parquet"
        return res
    df = pl.read_parquet(files[0])
    res["n"] = df.height
    spec = DesignSpec.from_data(df, L.FRENCH_PREDICTORS, n_bins=n_bins, weight_col="Exposure")
    res["p"] = spec.n_features
    res["spec"] = repr(spec)
    y = (df["ClaimNb"] / df["Exposure"]).to_numpy()
    w = df["Exposure"].to_numpy()
    codes = L.var_codes(spec, df)
    # compression at several design granularities
    ratios = {}
    for nb in (10, 20, 30):
        sp = DesignSpec.from_data(df, L.FRENCH_PREDICTORS, n_bins=nb, weight_col="Exposure")
        cd = L.var_codes(sp, df)
        ratios[f"n_bins={nb} (p={sp.n_features})"] = df.height / L.n_distinct_rows(sp, cd)
    # and dropping the finest variables
    for drop in (["Density"], ["Density", "BonusMalus"], ["Density", "BonusMalus", "Region"]):
        preds = [v for v in L.FRENCH_PREDICTORS if v not in drop]
        sp = DesignSpec.from_data(df, preds, n_bins=n_bins, weight_col="Exposure")
        cd = L.var_codes(sp, df)
        ratios[f"without {'+'.join(drop)} (p={sp.n_features})"] = df.height / L.n_distinct_rows(sp, cd)
    res["compression_ratios"] = ratios

    tb = time.time()
    X = spec.build(df)
    m0 = _fit(X, y, w)
    res["t_rowlevel"] = time.time() - tb
    p0 = L.predict_from_codes(spec, codes, m0.coef_, m0.intercept_)
    ta = time.time()
    codes_agg, ybar, W, g = L.aggregate(spec, codes, y, w)
    Xa = L.build_dense_from_codes(spec, codes_agg, np.float64)
    m1 = _fit(Xa, ybar, W)
    res["t_aggregated"] = time.time() - ta
    res["n_groups"] = len(W)
    res["compression_ratio"] = df.height / len(W)
    p1 = L.predict_from_codes(spec, codes, m1.coef_, m1.intercept_)
    rel = np.abs(p1 - p0) / p0
    res["max_rel_pred_diff"] = float(rel.max())
    res["max_abs_coef_diff"] = float(np.abs(m1.coef_ - m0.coef_).max())
    res["nnz"] = int((m0.coef_ != 0).sum())
    res["nnz_agg"] = int((m1.coef_ != 0).sum())
    res["nnz_set_diff"] = int(((m0.coef_ != 0) != (m1.coef_ != 0)).sum())
    res["n_iter"] = int(m0.n_iter_)
    res["n_iter_agg"] = int(m1.n_iter_)
    # tighter tolerance: does the gap close?
    m0t = _fit(X, y, w, gradient_tol=1e-8)
    m1t = _fit(Xa, ybar, W, gradient_tol=1e-8)
    p0t = L.predict_from_codes(spec, codes, m0t.coef_, m0t.intercept_)
    p1t = L.predict_from_codes(spec, codes, m1t.coef_, m1t.intercept_)
    res["max_rel_pred_diff_tight_tol"] = float(np.max(np.abs(p1t - p0t) / p0t))
    res["max_rel_pred_diff_default_vs_tight_rowlevel"] = float(np.max(np.abs(p0t - p0) / p0))
    # scale_predictors=False: aggregation still exact?
    m0u = _fit(X, y, w, scale_predictors=False)
    m1u = _fit(Xa, ybar, W, scale_predictors=False)
    p0u = L.predict_from_codes(spec, codes, m0u.coef_, m0u.intercept_)
    p1u = L.predict_from_codes(spec, codes, m1u.coef_, m1u.intercept_)
    res["max_rel_pred_diff_unscaled"] = float(np.max(np.abs(p1u - p0u) / p0u))
    res["nnz_unscaled"] = int((m0u.coef_ != 0).sum())
    # Gamma severity-style check (aggregation exactness for another family)
    from glum import GeneralizedLinearRegressor

    sev = df.filter(pl.col("ClaimNb") > 0)
    if sev.height > 1000:
        rng = np.random.default_rng(0)
        cnt = sev["ClaimNb"].to_numpy()
        amt = rng.gamma(2.0, 1000.0, sev.height) * cnt  # synthetic total cost per policy
        ysev = amt / cnt
        sp_s = DesignSpec.from_data(sev, L.FRENCH_PREDICTORS, n_bins=10, weight_col="ClaimNb")
        cds = L.var_codes(sp_s, sev)
        Xs = L.build_dense_from_codes(sp_s, cds)
        g0 = GeneralizedLinearRegressor(family="gamma", link="log", alpha=ALPHA, l1_ratio=1.0).fit(Xs, ysev, sample_weight=cnt)
        ca, yb, Wg, gg = L.aggregate(sp_s, cds, ysev, cnt.astype(float))
        g1 = GeneralizedLinearRegressor(family="gamma", link="log", alpha=ALPHA, l1_ratio=1.0).fit(
            L.build_dense_from_codes(sp_s, ca), yb, sample_weight=Wg
        )
        q0 = L.predict_from_codes(sp_s, cds, g0.coef_, g0.intercept_)
        q1 = L.predict_from_codes(sp_s, cds, g1.coef_, g1.intercept_)
        res["gamma_max_rel_pred_diff"] = float(np.max(np.abs(q1 - q0) / q0))
        res["gamma_compression_ratio"] = sev.height / len(Wg)
    res["t_total"] = time.time() - t0
    return res


# ==========================================================================
# parent side
# ==========================================================================
def _child(args: list[str], timeout: int = 3600) -> dict:
    env = dict(os.environ, PYTHONPATH=str(SPIKE))
    t = time.time()
    try:
        proc = subprocess.run(
            [PY, str(SPIKE / "bench.py"), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(SPIKE),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"> {timeout}s", "t_total": time.time() - t}
    out = proc.stdout
    line = next((ln for ln in out.splitlines()[::-1] if ln.startswith("RESULT_JSON:")), None)
    if line is None:
        return {
            "status": "crashed",
            "error": (proc.stderr or out)[-2000:],
            "returncode": proc.returncode,
            "t_total": time.time() - t,
        }
    rec = json.loads(line[len("RESULT_JSON:") :])
    rec["wall_subprocess"] = time.time() - t
    if proc.returncode not in (0, 3):
        rec.setdefault("error", proc.stderr[-1000:])
    return rec


def _load_results() -> list[dict]:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return []


def _save_results(rows: list[dict]) -> None:
    RESULTS.write_text(json.dumps(rows, indent=1, default=str))


def _upsert(rows: list[dict], rec: dict) -> None:
    key = (rec.get("candidate"), rec.get("n"))
    rows[:] = [r for r in rows if (r.get("candidate"), r.get("n")) != key]
    rows.append(rec)


def _gb(b) -> str:
    return "" if b in (None, "", -1) else f"{b / 2**30:.2f}"


def _fmt(v, spec=".1e") -> str:
    return "" if v in (None, "") else format(v, spec)


def print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'candidate':14s} {'n':>9s} {'p':>4s} {'status':10s} {'build_s':>7s} {'fit_s':>7s} "
        f"{'total_s':>7s} {'peakGB':>6s} {'desGB':>6s} {'iter':>4s} {'nnz':>4s} "
        f"{'maxRelPred':>10s} {'p99RelPred':>10s} {'nnzΔ':>4s} {'glumVsF64':>9s} {'ratio':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r.get("n") or 0, ALL_CANDIDATES.index(r["candidate"]) if r["candidate"] in ALL_CANDIDATES else 99)):
        if r["candidate"] == "french_agg":
            continue
        print(
            f"{r['candidate']:14s} {r.get('n', ''):>9} {r.get('p', ''):>4} {r.get('status', ''):10s} "
            f"{_fmt(r.get('t_build'), '7.1f'):>7s} {_fmt(r.get('t_fit'), '7.1f'):>7s} {_fmt(r.get('t_total'), '7.1f'):>7s} "
            f"{_gb(r.get('ru_maxrss')):>6s} {_gb(r.get('design_bytes')):>6s} {r.get('n_iter', ''):>4} {r.get('nnz', ''):>4} "
            f"{_fmt(r.get('max_rel_pred_diff')):>10s} {_fmt(r.get('p99_rel_pred_diff')):>10s} {r.get('nnz_set_diff', ''):>4} "
            f"{_fmt(r.get('glum_pred_vs_f64_recomposed_max_rel')):>9s} {_fmt(r.get('compression_ratio'), '6.2f'):>6s}"
        )
    for r in rows:
        if r["candidate"] == "french_agg":
            print("\nFrench motor aggregation:")
            for k, v in r.items():
                if k not in ("candidate", "rss_watchdog_peak"):
                    print(f"  {k}: {v}")
    for r in rows:
        if r.get("error") or r.get("notes"):
            print(f"\n[{r['candidate']} n={r.get('n')}] status={r.get('status')}")
            if r.get("error"):
                print("  error:", str(r["error"])[:600])
            for note in r.get("notes", []):
                print("  note:", note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="child mode: candidate name or 'french_agg'")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--sizes", type=int, nargs="*", default=list(DEFAULT_PLAN))
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--skip-french", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if args.run and args.run.startswith("probe_f64_weights_"):
        sys.path.insert(0, str(SPIKE))
        import numpy as np

        import spike_lib as L

        df = _data(args.n).head(50_000)
        spec = _spec(args.n, df)
        codes = L.var_codes(spec, df)
        X = (
            L.build_dense_from_codes(spec, codes, np.float32)
            if args.run.endswith("dense32")
            else L.build_split(spec, codes, np.float32)
        )
        y = (df["claims"] / df["exposure"]).to_numpy()
        w = df["exposure"].to_numpy()
        try:
            _fit(X, y, w)  # float64 y / w on purpose
            print("OK: no error with float64 weights")
        except Exception as exc:  # noqa: BLE001
            print(f"{type(exc).__name__}: {exc}")
        return

    if args.run:
        sys.path.insert(0, str(SPIKE))
        if args.run == "french_agg":
            rec = run_french()
        else:
            rec = run_candidate(args.run, args.n)
        _emit(rec)
        return

    rows = _load_results()
    if args.table_only:
        print_table(rows)
        return

    print(f"machine: {MEM_TOTAL / 2**30:.0f} GB RAM, cap {MEM_CAP / 2**30:.1f} GB; alpha={ALPHA}, n_bins={N_BINS}")
    for n in args.sizes:
        cands = args.candidates or DEFAULT_PLAN.get(n, ["baseline", "split32"])
        # baseline first so others can compare against it
        cands = sorted(cands, key=lambda c: (c != "baseline", ALL_CANDIDATES.index(c) if c in ALL_CANDIDATES else 99))
        for cand in cands:
            print(f"--- {cand} n={n} ...", flush=True)
            rec = _child(["--run", cand, "--n", str(n)], timeout=args.timeout)
            rec.setdefault("candidate", cand)
            rec.setdefault("n", n)
            _upsert(rows, rec)
            _save_results(rows)
            print(
                f"    status={rec.get('status')} total={rec.get('t_total', 0):.1f}s "
                f"peak={_gb(rec.get('ru_maxrss'))}GB maxRelPred={_fmt(rec.get('max_rel_pred_diff'))} "
                f"{rec.get('error', '')[:200] if rec.get('error') else ''}",
                flush=True,
            )
    if not args.skip_french and not args.candidates:
        print("--- french_agg ...", flush=True)
        rec = _child(["--run", "french_agg"], timeout=args.timeout)
        rec.setdefault("candidate", "french_agg")
        _upsert(rows, rec)
        _save_results(rows)
    print()
    print_table(rows)


if __name__ == "__main__":
    main()

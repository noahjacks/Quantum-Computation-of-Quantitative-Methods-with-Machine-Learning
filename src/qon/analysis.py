"""
analysis.py  --  TURN SAVED PREDICTIONS INTO TABLES AND CHARTS
==============================================================

The long runs (scripts 04 and 05) only save raw predictions: for every test
request, every model's p(fill).  Everything a reader sees - every table and
chart in the notebooks - is computed from those files by the functions here,
so the numbers in the write-up can always be regenerated and checked.

Reading the charts
------------------
Most charts are "dot and whisker" plots.  The dot is the mean daily AUC; the
whisker is the 95% confidence interval (see stats.py).  Two dots are only
"really different" if the interval of their DIFFERENCE excludes zero, which
is why every headline chart has a companion chart of paired differences.

Colours always mean the same thing:
    blue    classical (no quantum computer involved)
    orange  quantum (the circuit, perfect or noisy)
    green   controls (classical imitations built to explain a quantum result)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .stats import bootstrap_mean, brier_score, daily_auc, summarize

# ---- look of the charts (colour-blind-safe palette, checked with a validator)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
FAMILY_COLOUR = {"classical": BLUE, "quantum": ORANGE, "control": GREEN}

SET_INFO = {   # name -> (label shown to readers, family)
    "A": ("A  classical features", "classical"),
    "B": ("B  quantum, perfect chip", "quantum"),
    "C": ("C  quantum, noisy chip", "quantum"),
    "D": ("D  quantum, real IBM chip", "quantum"),
    "E": ("E  B + classical noise", "control"),
    "F": ("F  random classical features", "control"),
    "B_shots": ("B + shot noise only", "quantum"),
    "C_drift_chrono": ("drifting chip, events in date order", "quantum"),
    "C_drift_random": ("drifting chip, events in random order", "quantum"),
}
METHOD_FAMILY = {"LR": "classical", "RF": "classical", "XGB": "classical", "NN": "classical",
                 "SVM-linear": "classical", "SVM-RBF": "classical",
                 "SVM-PQK": "quantum", "SVM-PQK-noisy": "quantum",
                 "SVM-FQK": "quantum", "SVM-FQK-shots": "quantum"}
METHOD_LABEL = {"LR": "Logistic regression", "RF": "Random forest", "XGB": "Gradient boosting",
                "NN": "Neural network", "SVM-linear": "SVM, linear kernel",
                "SVM-RBF": "SVM, classical RBF kernel",
                "SVM-PQK": "SVM, projected quantum kernel",
                "SVM-PQK-noisy": "SVM, projected quantum kernel (noisy chip)",
                "SVM-FQK": "SVM, fidelity quantum kernel",
                "SVM-FQK-shots": "SVM, fidelity quantum kernel (4096 shots)"}
for _name in list(METHOD_FAMILY):
    if _name.startswith("SVM"):
        METHOD_FAMILY[_name.replace("SVM", "KRR")] = METHOD_FAMILY[_name]
        METHOD_LABEL[_name.replace("SVM", "KRR")] = METHOD_LABEL[_name].replace("SVM", "Kernel ridge")
MODELS = ("LR", "RF", "XGB", "NN")


def _style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "font.family": "sans-serif", "legend.frameon": False, "figure.dpi": 110,
        "xtick.major.size": 0, "ytick.major.size": 0, "axes.axisbelow": True})
    return plt


def _save(fig, name: str):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(C.TABLES / f"{name}.parquet")


def auc_by_day(pred: pd.DataFrame, key: str = "set") -> pd.DataFrame:
    """One row per test day; columns (key, model).  Adds the pseudo-model
    "All" = the average of the four models' AUCs on that day."""
    cols = {}
    for (k, model), g in pred.groupby([key, "model"]):
        cols[(k, model)] = daily_auc(g["filled"].to_numpy(), g["p"].to_numpy(), g["day"].to_numpy())
    out = pd.DataFrame(cols)
    if key == "set":
        for k in out.columns.get_level_values(0).unique():
            members = [m for m in MODELS if (k, m) in out.columns]
            if members:
                out[(k, "All")] = out[[(k, m) for m in members]].mean(axis=1)
    return out.sort_index(axis=1)


def part1_table(pred: pd.DataFrame, blinding: int, model: str = "All",
                sets=("A", "B", "C", "E", "F"), baseline: str = "A") -> pd.DataFrame:
    """Mean daily AUC of each feature set and the paired difference to set A."""
    daily = auc_by_day(pred[pred["blinding"] == blinding])
    table = pd.DataFrame({s: daily[(s, model)] for s in sets if (s, model) in daily.columns})
    return summarize(table, baseline)


def part1_matrix(pred: pd.DataFrame, blinding: int, sets=("A", "B", "C", "E", "F")) -> pd.DataFrame:
    """Compact overview: feature sets in rows, models in columns; each cell
    is 'mean daily AUC (difference to A)', with * if the difference is
    statistically significant at the 95% level."""
    rows = {}
    for model in ("All", *MODELS):
        t = part1_table(pred, blinding, model, sets)
        rows[model] = [f"{r.mean_auc:.3f}" if s == "A" else
                       f"{r.mean_auc:.3f} ({r.diff_vs_A:+.3f}{'*' if r.significant else ''})"
                       for s, r in t.iterrows()]
    out = pd.DataFrame(rows, index=[SET_INFO[s][0] for s in sets if s in pred["set"].unique()])
    out.index.name = f"blinding = {blinding} day(s)"
    return out


def paired_difference(pred: pd.DataFrame, a: str, b: str, blinding: int, model: str = "All") -> dict:
    """Mean of (AUC of set a - AUC of set b) over test days, with a 95% interval."""
    daily = auc_by_day(pred[pred["blinding"] == blinding])
    d = (daily[(a, model)] - daily[(b, model)]).dropna().to_numpy()
    m, lo, hi = bootstrap_mean(d)
    return {"a": a, "b": b, "blinding": blinding, "model": model, "mean_diff": m,
            "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)}


def part2_table(pred: pd.DataFrame, blinding: int, baseline: str = "XGB") -> pd.DataFrame:
    """Part 2: one row per learning algorithm."""
    sub = pred[pred["blinding"] == blinding]
    daily = pd.DataFrame({m: daily_auc(g["filled"].to_numpy(), g["p"].to_numpy(), g["day"].to_numpy())
                          for m, g in sub.groupby("model")})
    best_classical = max((m for m in daily.columns if METHOD_FAMILY[m] == "classical"),
                         key=lambda m: daily[m].mean())
    out = summarize(daily, best_classical)
    out["brier"] = [brier_score(g["filled"], g["p"]) for m, g in
                    ((m, sub[sub["model"] == m]) for m in out.index)]
    out["family"] = [METHOD_FAMILY[m] for m in out.index]
    out.attrs["baseline"] = best_classical
    return out.sort_values("mean_auc", ascending=False)


def clock_leak(drifting: np.ndarray, steady: np.ndarray, day: np.ndarray, is_cal: np.ndarray,
               seed: int = 0) -> float:
    """Can the chip's DRIFT tell early events from late ones?

    `drifting` are the features from the drifting simulated chip and `steady`
    the features of the same events from the same chip without drift.  Their
    difference contains only drift (plus measurement randomness).  A simple
    classifier is asked to say, from that difference alone, whether an event
    comes from the first or the second half of the test window.

        AUC ~ 0.5  the drift knows nothing about the date (safe)
        AUC ~ 1.0  the drift is a hidden clock: the model inputs now contain
                   the date even though nobody put it there"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    diff = (drifting - steady)[~is_cal]
    late = (day[~is_cal] > np.median(day[~is_cal])).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    p = cross_val_predict(model, diff, late, cv=KFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(late, p))


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def _dot_whisker(ax, labels, means, lows, highs, colours, ref=None, ref_label=None, fmt="{:.3f}"):
    y = np.arange(len(labels))[::-1]
    for yi, m, lo, hi, col in zip(y, means, lows, highs, colours):
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([m], [yi], "o", color=col, ms=9, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(fmt.format(m), (hi, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.grid(axis="y", visible=False)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, zorder=1)
        if ref_label:
            ax.annotate(ref_label, (ref, len(labels) - 0.45), xytext=(4, 0),
                        textcoords="offset points", fontsize=8.5, color=INK2, va="top")


def fig_part1_headline(pred: pd.DataFrame, blinding: int = 0, sets=("A", "B", "C", "E", "F"),
                       name: str = "part1_headline"):
    plt = _style()
    t = part1_table(pred, blinding, "All", sets)
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(t) + 1.5))
    _dot_whisker(ax, [SET_INFO[s][0] for s in t.index], t["mean_auc"], t["ci_low"], t["ci_high"],
                 [FAMILY_COLOUR[SET_INFO[s][1]] for s in t.index],
                 ref=t.loc["A", "mean_auc"], ref_label="classical baseline (A)")
    ax.set_xlabel("mean daily AUC, averaged over the four models (dot) with 95% interval (line)")
    ax.set_title(f"Part 1: no feature set beats the classical baseline  (blinding {blinding} d)"
                 if not (t["significant"] & (t["diff_vs_A"] > 0)).any()
                 else f"Part 1: feature sets compared  (blinding {blinding} d)")
    pad = 0.35 * (t["ci_high"].max() - t["ci_low"].min())
    ax.set_xlim(min(t["ci_low"].min() - pad, 0.5), t["ci_high"].max() + 2.2 * pad)
    if ax.get_xlim()[0] <= 0.5:
        ax.axvline(0.5, color=GRID, lw=1)
        ax.annotate("guessing", (0.5, -0.55), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="bottom")
    return _save(fig, f"{name}_blind{blinding}")


def fig_part1_differences(pred: pd.DataFrame, sets=("B", "C", "E", "F"),
                          name: str = "part1_differences"):
    """Paired difference to set A for every model and blinding."""
    plt = _style()
    blindings = sorted(pred["blinding"].unique())
    fig, axes = plt.subplots(1, len(blindings), figsize=(4.1 * len(blindings), 0.3 * len(sets) * 5 + 1.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, blindings):
        labels, means, lows, highs, cols = [], [], [], [], []
        for s in sets:
            for model in ("All", *MODELS):
                r = part1_table(pred, b, model, ("A", s)).loc[s]
                labels.append(f"{s}  {model}")
                means.append(r["diff_vs_A"]); lows.append(r["diff_ci_low"]); highs.append(r["diff_ci_high"])
                cols.append(FAMILY_COLOUR[SET_INFO[s][1]])
        y = np.arange(len(labels))[::-1]
        for yi, m, lo, hi, col, lab in zip(y, means, lows, highs, cols, labels):
            bold = lab.endswith("All")
            ax.plot([lo, hi], [yi, yi], color=col, lw=2.2 if bold else 1.4, zorder=2,
                    alpha=1 if bold else 0.75)
            ax.plot([m], [yi], "o", color=col, ms=8 if bold else 5.5, mec=SURFACE, mew=1.5, zorder=3)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"blinding {b} d", fontweight="normal", fontsize=10.5)
        ax.set_xlabel("AUC minus AUC of set A")
    fig.suptitle("Difference to the classical baseline, per model (0 = no difference)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, name)


def fig_hardware(table: pd.DataFrame, name: str = "partD_hardware", rehearsal: bool = False,
                 hsbc_range: tuple[float, float] = (0.75, 0.97)):
    """The real-hardware comparison: one pooled AUC per feature set, on the
    same events, with the scores HSBC report for hardware shown for scale.

    table : the output of hardware_analysis.analyse (rows set/model/auc/ci)."""
    plt = _style()
    t = table[table["model"] == "All"].set_index("set").loc[["A", "B", "C", "D", "E", "F"]]
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(t) + 1.7))
    ax.axvspan(*hsbc_range, color=GRID, alpha=0.75, lw=0, zorder=0)
    ax.annotate("what HSBC report\nfor real hardware", (sum(hsbc_range) / 2, len(t) - 0.5),
                ha="center", va="top", fontsize=8.5, color=INK2)
    _dot_whisker(ax, [SET_INFO[s][0] for s in t.index], t["auc"], t["ci_low"], t["ci_high"],
                 [FAMILY_COLOUR[SET_INFO[s][1]] for s in t.index], ref=0.5, ref_label="guessing",
                 fmt="{:.2f}")
    ax.set_xlim(0.33, 1.0)
    ax.set_xlabel("AUC over all test requests, average of the four models (dot) with 95% interval (line)")
    what = "REHEARSAL on simulated data, not hardware" if rehearsal else "real IBM chip, 400 quote requests"
    ax.set_title(f"Set D: every input scores at guessing level  ({what})")
    return _save(fig, name)


def fig_drift(rows: pd.DataFrame, name: str = "part1_drift"):
    """rows: output of several paired_difference() calls plus a 'label' column."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.6, 0.6 * len(rows) + 1.5))
    _dot_whisker(ax, rows["label"], rows["mean_diff"], rows["ci_low"], rows["ci_high"],
                 [ORANGE] * len(rows), ref=0.0, ref_label="no effect", fmt="{:+.3f}")
    ax.set_xlabel("AUC with events sent in date order  minus  AUC with events sent in random order")
    ax.set_title("Drift check: does sending events to the chip in date order help?")
    span = max(abs(rows["ci_low"].min()), abs(rows["ci_high"].max())) * 1.6
    ax.set_xlim(-span, span)
    return _save(fig, name)


def fig_part2(pred: pd.DataFrame, blinding: int = 0, name: str = "part2_methods",
              title: str = "Part 2: learning algorithms compared on the same inputs"):
    plt = _style()
    from matplotlib.lines import Line2D
    t = part2_table(pred, blinding)
    fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(t) + 1.7))
    _dot_whisker(ax, [METHOD_LABEL[m] for m in t.index], t["mean_auc"], t["ci_low"], t["ci_high"],
                 [FAMILY_COLOUR[f] for f in t["family"]],
                 ref=t.loc[t.attrs["baseline"], "mean_auc"], ref_label="best classical method")
    ax.set_xlabel("mean daily AUC (dot) with 95% interval (line)")
    ax.set_title(f"{title}  (blinding {blinding} d)")
    pad = 0.3 * (t["ci_high"].max() - t["ci_low"].min())
    ax.set_xlim(min(t["ci_low"].min() - pad, 0.495), t["ci_high"].max() + 1.6 * pad)
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (1, 2)), zorder=1)
    ax.annotate("guessing", (0.5, -0.55), xytext=(4, 0), textcoords="offset points",
                fontsize=8.5, color=INK2, va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="classical"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label="quantum kernel")],
              loc="lower right", fontsize=9)
    return _save(fig, f"{name}_blind{blinding}")


def fig_fqk_screen(screen: pd.DataFrame, name: str = "part2_fqk_bandwidth"):
    """Two separate panels (never two scales on one axis)."""
    plt = _style()
    s = screen.sort_values("alpha")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.3))
    a1.plot(s["alpha"], s["median_similarity"].clip(lower=1e-300), "-o", color=ORANGE, lw=2, ms=8,
            mec=SURFACE, mew=2)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("rotation scale alpha (the kernel's bandwidth)")
    a1.set_ylabel("typical similarity of two different requests")
    a1.set_title("Similarity collapses as alpha grows")
    a2.plot(s["alpha"], s["validation_auc"], "-o", color=ORANGE, lw=2, ms=8, mec=SURFACE, mew=2)
    a2.axhline(0.5, color=INK2, lw=1)
    a2.annotate("guessing", (s["alpha"].min(), 0.5), xytext=(0, 4), textcoords="offset points",
                fontsize=8.5, color=INK2)
    a2.set_xscale("log")
    a2.set_xlabel("rotation scale alpha (the kernel's bandwidth)")
    a2.set_ylabel("AUC on held-out calibration days")
    a2.set_title("SVM score for each bandwidth")
    fig.tight_layout()
    return _save(fig, name)


def fig_fill_rate(events: pd.DataFrame, name: str = "data_fill_rate"):
    plt = _style()
    main = events[events["period"] == "main"]
    daily = main.groupby(["date", "side"])["filled"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for side, col, lab in ((1, BLUE, "customer buys"), (-1, ORANGE, "customer sells")):
        ax.plot(daily.index, daily[side], color=col, lw=2, label=lab)
    ax.axhline(main["filled"].mean(), color=INK2, lw=1)
    ax.annotate(f"average {main['filled'].mean():.0%}", (daily.index[-1], main["filled"].mean()),
                xytext=(8, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="center")
    ax.set_ylabel("share of our quotes filled")
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Fill rate of the replayed quotes, day by day")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.autofmt_xdate()
    return _save(fig, name)


def fig_noise_effect(sets: dict, column: int = 0, n: int = 2500, name: str = "circuit_noise_effect"):
    """What noise does to one of the 327 quantum features."""
    plt = _style()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(sets["B"]), n, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True, sharey=True)
    for ax, key, col, title in ((axes[0], "C", ORANGE, "C: noisy simulated chip"),
                                (axes[1], "E", GREEN, "E: classical imitation of that noise")):
        ax.scatter(sets["B"][idx, column], sets[key][idx, column], s=6, color=col, alpha=0.35, lw=0)
        lim = [sets["B"][idx, column].min(), sets["B"][idx, column].max()]
        ax.plot(lim, lim, color=INK2, lw=1)
        ax.set_title(title, fontweight="normal", fontsize=10.5)
        ax.set_xlabel("value on a perfect chip (set B)")
    axes[0].set_ylabel("value after noise")
    axes[0].text(0.04, 0.95, "grey line: where the points\nwould sit without noise",
                 transform=axes[0].transAxes, fontsize=8.5, color=INK2, va="top")
    fig.suptitle("Noise shrinks each quantum feature and adds scatter", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, name)

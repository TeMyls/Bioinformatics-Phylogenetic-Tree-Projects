"""Benchmark harness (Project A): run methods over a size ladder x replicates, score, summarise, plot."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics as M
from .metrics import signflip_wilcoxon as M_signflip
from .infer import available_methods, infer_tree
from .plotting import PALETTE, method_colors, plt
from .seqio import Alignment
from .tree import Tree

# A "dataset factory" returns (alignment, reference tree or None, {tip: lineage} or None)
DatasetFactory = Callable[[int, int], tuple[Alignment, Tree | None, dict | None]]


def sim_factory(length: int = 29_903, rate: float = 1e-3, span_years: float = 2.0) -> DatasetFactory:
    from .simulate import simulate_dataset

    def make(n: int, seed: int):
        sim = simulate_dataset(n, length=length, rate=rate, span_years=span_years, seed=seed)
        return sim.aln, sim.tree_subs, sim.lineages

    return make


def real_factory(aln: Alignment, meta: pd.DataFrame, reference: Tree | None = None) -> DatasetFactory:
    """Sample n tips stratified by Pango lineage from a pre-aligned, pre-masked, de-duplicated alignment."""
    from .qc import subsample_stratified

    meta = meta[meta["name"].isin(set(aln.names))]
    lab = dict(zip(meta["name"], meta["lineage"]))

    def make(n: int, seed: int):
        sub = subsample_stratified(meta, n, seed=seed)
        a = aln.subset_names(list(sub["name"]))
        ref = reference.pruned(set(a.names) & set(reference.leaf_names)) if reference is not None else None
        return a, ref, {k: lab[k] for k in a.names}

    return make


def run_benchmark(factory: DatasetFactory, sizes: list[int], reps: int, methods: list[str] | None = None,
                  workdir: str | Path | None = None, seed0: int = 0, threads: int = 1,
                  method_params: dict | None = None, collapse_len: int = 29_903, verbose: bool = True,
                  warmup: bool = True) -> pd.DataFrame:
    """One row per (size, replicate, method). Every method sees the *same* dataset within a replicate."""
    methods = methods or available_methods()
    method_params = method_params or {}
    workdir = Path(workdir or tempfile.mkdtemp(prefix="covidphylo_bench_"))
    thr = 0.5 / collapse_len                      # branches with <0.5 expected mutations are unresolvable
    rows = []
    if warmup:   # first call of each method pays import / allocation costs; run once and discard
        w_aln, _, _ = factory(min(sizes), seed0 + 999_999)
        for m in methods:
            try:
                infer_tree(m, w_aln, workdir / f"warmup_{m}", seed=seed0, threads=threads, params=method_params.get(m))
            except Exception:
                pass
    for n in sizes:
        for r in range(reps):
            seed = seed0 + 1000 * n + r
            aln, ref, labels = factory(n, seed)
            for m in methods:
                try:
                    res = infer_tree(m, aln, workdir / f"{m}_{n}_{r}", seed=seed + 1, threads=threads,
                                     params=method_params.get(m))
                except Exception as e:                     # keep going; record failure
                    if verbose:
                        print(f"[skip] {m} n={n} rep={r}: {str(e)[:120]}")
                    continue
                row = dict(method=m, n_tips=aln.n, rep=r, seed=seed, wall_s=res.wall_s,
                           peak_mb=res.peak_mb, mem_source=res.mem_source)
                if ref is not None:
                    raw = M.rf_metrics(ref, res.tree)
                    col = M.rf_metrics(ref, res.tree, collapse_below=thr)
                    row.update(nrf=raw["nrf"], nrf_collapsed=col["nrf"],
                               patristic_r=M.patristic_correlation(ref, res.tree))
                if labels:
                    row["lineage_auc"] = M.lineage_auc(res.tree, labels, seed=seed)
                    row["lineage_monophyly"] = M.strict_monophyly(res.tree, labels)["fraction"]
                rows.append(row)
                if verbose:
                    print(f"{m:9s} n={aln.n:5d} rep={r} {res.wall_s:8.2f}s "
                          + (f"nRF*={row.get('nrf_collapsed', float('nan')):.3f}" if ref is not None else ""))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

_LOG_COLS = ("wall_s", "peak_mb")                 # positive, right-skewed -> geometric mean, log-scale CI
_BOUNDS = {"nrf": (0, 1), "nrf_collapsed": (0, 1), "lineage_auc": (0, 1),
           "lineage_monophyly": (0, 1), "patristic_r": (-1, 1)}


def summarise(df: pd.DataFrame, cols=("wall_s", "peak_mb", "nrf", "nrf_collapsed",
                                      "patristic_r", "lineage_auc", "lineage_monophyly")) -> pd.DataFrame:
    """Mean and 95% CI per (method, n). Time/memory: geometric mean with a log-scale CI;
    bounded metrics: CI clipped to the valid range. (F1 on collapsed trees equals 1 - nRF, so it is not reported.)"""
    out = []
    for (m, n), g in df.groupby(["method", "n_tips"]):
        row = dict(method=m, n_tips=n, reps=len(g))
        for c in cols:
            if c in g and g[c].notna().any():
                mean, lo, hi = M.mean_ci(g[c].to_numpy(float), log=c in _LOG_COLS, bounds=_BOUNDS.get(c))
                row[c], row[c + "_lo"], row[c + "_hi"] = mean, lo, hi
        out.append(row)
    return pd.DataFrame(out).sort_values(["n_tips", "method"]).reset_index(drop=True)


def paired_tests(df: pd.DataFrame, metric: str = "nrf_collapsed", at_n: int | None = None) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test between every method pair, Holm-adjusted.

    Pairs are (size, replicate) datasets that every method saw. at_n=None pools all sizes
    (more pairs -> more power); at_n=N restricts to one size. Time/memory are compared on the
    log scale (``fold_change`` = geometric-mean ratio a/b). ``min_possible_p`` is the smallest
    p-value attainable with that many non-zero pairs: if it is above your alpha the test
    cannot be significant, whatever the data - run more replicates.
    """
    d = df if at_n is None else df[df["n_tips"] == at_n]
    wide = d.pivot_table(index=["n_tips", "rep"], columns="method", values=metric)
    islog = metric in _LOG_COLS
    if islog:
        wide = np.log(wide)
    ms, rows = list(wide.columns), []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            x = wide[[ms[i], ms[j]]].dropna()
            diff = (x.iloc[:, 0] - x.iloc[:, 1]).to_numpy()
            p, k = M_signflip(diff)
            row = dict(metric=metric, at_n="all" if at_n is None else at_n, a=ms[i], b=ms[j],
                       n_pairs=len(x), n_nonzero=k,
                       min_possible_p=(2.0 ** (1 - k) if k else np.nan), p=p)
            if islog:
                row["fold_change_a_over_b"] = float(np.exp(diff.mean())) if len(diff) else np.nan
            else:
                row["mean_diff_a_minus_b"] = float(diff.mean()) if len(diff) else np.nan
            row["a_lower_in"] = int((diff < -1e-12).sum())
            row["b_lower_in"] = int((diff > 1e-12).sum())
            rows.append(row)
    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out["p_holm"] = np.nan
    if ok.any():
        out.loc[ok, "p_holm"] = M.holm(out.loc[ok, "p"].tolist())
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _errplot(ax, S, col, ylabel, colors, logy=False):
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    methods = sorted(S["method"].unique())
    sizes = sorted(S["n_tips"].unique())
    for k, (m, g) in enumerate(S.groupby("method")):
        if col not in g:
            continue
        x = g["n_tips"] * (1 + 0.025 * (k - (len(methods) - 1) / 2))      # dodge so overlapping series stay visible
        ax.errorbar(x, g[col], yerr=[g[col] - g[col + "_lo"], g[col + "_hi"] - g[col]],
                    marker="o", ms=4, lw=1.2, capsize=2, label=m, color=colors[m])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Number of tips")
    ax.set_ylabel(ylabel)


def plot_benchmark(df: pd.DataFrame, outdir: str | Path) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    S = summarise(df)
    colors = method_colors(sorted(df["method"].unique()))
    files = []

    fig, ax = plt.subplots(1, 2 if "peak_mb" in S else 1, figsize=(7.2, 3.1), squeeze=False)
    _errplot(ax[0, 0], S, "wall_s", "Wall time (s, 1 thread)", colors, logy=True)
    if "peak_mb" in S:
        _errplot(ax[0, 1], S, "peak_mb", "Peak memory (MB)", colors, logy=True)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "fig_runtime_memory.png"
    fig.savefig(p)
    plt.close(fig)
    files.append(p)

    acc = [c for c in ("nrf_collapsed", "patristic_r", "lineage_auc") if c in S]
    if acc:
        labels = {"nrf_collapsed": "Normalised RF (collapsed)\n(lower = better)",
                  "patristic_r": "Patristic-distance correlation\n(higher = better)",
                  "lineage_auc": "Lineage-recovery AUC\n(higher = better)"}
        fig, ax = plt.subplots(1, len(acc), figsize=(2.9 * len(acc) + 0.5, 3.1), squeeze=False)
        for k, c in enumerate(acc):
            _errplot(ax[0, k], S, c, labels[c], colors)
        ax[0, 0].legend(fontsize=7)
        fig.tight_layout()
        p = outdir / "fig_accuracy.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p)

    if "nrf_collapsed" in S:
        big = S[S["n_tips"] == S["n_tips"].max()]
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        for _, r in big.iterrows():
            ax.errorbar(r["wall_s"], r["nrf_collapsed"],
                        xerr=[[r["wall_s"] - r["wall_s_lo"]], [r["wall_s_hi"] - r["wall_s"]]],
                        yerr=[[r["nrf_collapsed"] - r["nrf_collapsed_lo"]], [r["nrf_collapsed_hi"] - r["nrf_collapsed"]]],
                        fmt="o", color=colors[r["method"]], capsize=2, label=r["method"])
        ax.set_xscale("log")
        ax.set_xlabel(f"Wall time (s), n = {int(big['n_tips'].iloc[0])}")
        ax.set_ylabel("Normalised RF (collapsed)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = outdir / "fig_speed_accuracy.png"
        fig.savefig(p)
        plt.close(fig)
        files.append(p)
    return files

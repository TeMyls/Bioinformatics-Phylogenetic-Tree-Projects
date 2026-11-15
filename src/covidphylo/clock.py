"""Root-to-tip regression, best-fitting root, temporal-signal test and simple node dating.

This is a lightweight TempEst/LSD-style analysis. For publication-grade dating use
TreeTime or BEAST (see README) -- this module is for clock diagnostics and quick estimates.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .tree import Tree


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

def date_to_decimal_year(s) -> float:
    """'2021-03-15' -> 2021.2 ; incomplete dates ('2021-03-XX', '2021') -> NaN (excluded)."""
    try:
        d = _dt.date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return float("nan")
    start = _dt.date(d.year, 1, 1)
    end = _dt.date(d.year + 1, 1, 1)
    return d.year + (d - start).days / (end - start).days


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------

@dataclass
class ClockFit:
    slope: float            # substitutions / site / year
    intercept: float
    r2: float
    pearson_r: float
    p_value: float
    tmrca: float            # x-intercept (decimal year)
    n: int
    rss: float
    root_edge: tuple | None = None


def regress(rtt: np.ndarray, t: np.ndarray) -> ClockFit:
    ok = np.isfinite(rtt) & np.isfinite(t)
    res = stats.linregress(t[ok], rtt[ok])
    resid = rtt[ok] - (res.intercept + res.slope * t[ok])
    tm = -res.intercept / res.slope if res.slope != 0 else float("nan")
    return ClockFit(res.slope, res.intercept, res.rvalue**2, res.rvalue, res.pvalue, tm,
                    int(ok.sum()), float((resid**2).sum()))


def _tip_dates_in_order(view, dates: dict[str, float]) -> np.ndarray:
    return np.array([dates.get(nm, np.nan) for nm in view.leaf_names], dtype=float)


def root_to_tip(tree: Tree, dates: dict[str, float]):
    """Return (leaf_names, root-to-tip distances, dates) for the tree's stored root."""
    v = tree.rooted()
    rtt = np.array([v.depth[l] for l in v.leaves])
    return v.leaf_names, rtt, _tip_dates_in_order(v, dates)


# --------------------------------------------------------------------------
# Best-fitting root (TempEst-style, residual-sum-of-squares criterion)
# --------------------------------------------------------------------------

def find_best_root(tree: Tree, dates: dict[str, float], require_positive_slope: bool = True):
    """Search every branch (and the position along it) for the root minimising regression residuals.

    For a root at distance y from child c on edge (c, p), tip i has
        d_i(y) = base_i + s_i * y ,  s_i = +1 (tip below c) / -1 (otherwise)
    which is linear in y, so (intercept, slope, y) solve a 3-parameter least-squares problem per edge.
    Returns (rooted Tree copy, ClockFit).  O(edges x tips).
    """
    v = tree.rooted(root_edge=None)                       # arbitrary starting root (node)
    rows = v.node_leaf_distances()
    t = _tip_dates_in_order(v, dates)
    ok = np.isfinite(t)
    if ok.sum() < 4:
        raise ValueError("need >=4 tips with complete dates")
    tt = t[ok]
    n = len(t)
    best = None
    ones = np.ones(ok.sum())
    for c in v.order:
        if c == v.root:
            continue
        p, l = v.parent[c], v.plen[c]
        s = -np.ones(n)
        s[v.lo[c]: v.hi[c]] = 1.0
        base = np.where(s > 0, rows[c], l + rows[p])      # d_i at y=0
        so, bo = s[ok], base[ok]
        A = np.column_stack([ones, tt, -so])              # base ~ a + b t - s y
        sol, *_ = np.linalg.lstsq(A, bo, rcond=None)
        y = float(np.clip(sol[2], 0.0, l))
        d = bo + so * y
        slope, icpt = np.polyfit(tt, d, 1)
        if require_positive_slope and slope <= 0:
            continue
        rss = float(((d - (icpt + slope * tt)) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, c, p, y, l)
    if best is None:
        raise ValueError("no root gives a positive clock rate (no temporal signal?)")
    _, c, p, y, l = best
    # express root as (a, b, x) with x = distance from a ; a=c so x=y
    rooted = tree.with_root(c, p, y)
    rv = rooted.rooted()
    rtt = np.array([rv.depth[x] for x in rv.leaves])
    fit = regress(rtt, _tip_dates_in_order(rv, dates))
    fit.root_edge = (c, p, y)
    return rooted, fit


def permutation_test(tree: Tree, dates: dict[str, float], n_perm: int = 1000, seed: int = 0) -> dict:
    """Date-randomisation test at a FIXED root: is R^2 larger than when dates are shuffled?

    (A full test re-searches the root for each permutation; this variant is faster and slightly
    conservative towards declaring signal -- report it as such.)
    """
    names, rtt, t = root_to_tip(tree, dates)
    obs = regress(rtt, t)
    rng = np.random.default_rng(seed)
    ok = np.isfinite(t)
    tv = t[ok]
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = regress(rtt[ok], rng.permutation(tv)).r2
    p = (1 + (null >= obs.r2).sum()) / (n_perm + 1)
    return dict(observed_r2=obs.r2, p_permutation=float(p), null_mean=float(null.mean()),
                null_q95=float(np.quantile(null, 0.95)), n_perm=n_perm)


# --------------------------------------------------------------------------
# Node dating under a strict clock (rate and root position from the regression)
# --------------------------------------------------------------------------

def date_nodes(rooted: Tree, fit: ClockFit) -> dict[int, float]:
    """Estimated date (decimal year) for every node incl. the virtual root (-1): t = (depth - intercept)/slope."""
    v = rooted.rooted()
    return {node: (d - fit.intercept) / fit.slope for node, d in v.depth.items()}


def bootstrap_clock(tree: Tree, dates: dict[str, float], n_boot: int = 500, seed: int = 0) -> dict:
    """Bootstrap tips (fixed root) -> 95% intervals for rate and tMRCA."""
    names, rtt, t = root_to_tip(tree, dates)
    ok = np.flatnonzero(np.isfinite(t))
    rng = np.random.default_rng(seed)
    rate, tm = [], []
    for _ in range(n_boot):
        s = rng.choice(ok, size=len(ok), replace=True)
        f = regress(rtt[s], t[s])
        rate.append(f.slope)
        tm.append(f.tmrca)
    q = lambda x: tuple(np.nanquantile(x, [0.025, 0.975]))
    return dict(rate_ci=q(rate), tmrca_ci=q(tm), n_boot=n_boot)

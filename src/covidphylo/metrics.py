"""Tree-comparison and lineage-recovery metrics (+ small stats helpers)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .tree import Tree


# --------------------------------------------------------------------------
# Topology: Robinson-Foulds family
# --------------------------------------------------------------------------

def _common(a: Tree, b: Tree) -> tuple[Tree, Tree, list[str]]:
    shared = sorted(set(a.leaf_names) & set(b.leaf_names))
    if len(shared) < 4:
        raise ValueError("trees share fewer than 4 leaves")
    if len(shared) != a.n_leaves:
        a = a.pruned(shared)
    if len(shared) != b.n_leaves:
        b = b.pruned(shared)
    return a, b, shared


def rf_metrics(true: Tree, est: Tree, collapse_below: float | None = None) -> dict:
    """Robinson-Foulds comparison of unrooted topologies on the shared leaves.

    collapse_below: contract internal branches <= this length in BOTH trees first.
    For SARS-CoV-2 set it to ~0.5 / genome_length: branches with <1 expected mutation
    are statistically unresolvable, so penalising their arbitrary resolution is noise.
    """
    a, b, shared = _common(true, est)
    if collapse_below is not None:
        a, b = a.collapse_short_branches(collapse_below), b.collapse_short_branches(collapse_below)
    idx = {nm: i for i, nm in enumerate(shared)}
    A, B = a.split_masks(idx), b.split_masks(idx)
    inter = len(A & B)
    rf = len(A) + len(B) - 2 * inter
    denom = len(A) + len(B)
    prec = inter / len(B) if B else 1.0       # fraction of inferred splits that are true
    rec = inter / len(A) if A else 1.0        # fraction of true splits recovered
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(rf=rf, nrf=rf / denom if denom else 0.0, precision=prec, recall=rec, f1=f1,
                n_true_splits=len(A), n_est_splits=len(B), n_shared_leaves=len(shared))


def patristic_correlation(a: Tree, b: Tree) -> float:
    """Pearson correlation of leaf-to-leaf patristic distances (branch-length fidelity)."""
    a, b, shared = _common(a, b)
    na, Da = a.patristic()
    nb, Db = b.patristic()
    iu = np.triu_indices(len(shared), 1)
    return float(np.corrcoef(Da[iu], Db[iu])[0, 1])


# --------------------------------------------------------------------------
# Lineage recovery
# --------------------------------------------------------------------------

def lineage_auc(tree: Tree, labels: dict[str, str], max_pairs: int = 300_000, seed: int = 0) -> float:
    """P(patristic distance of a same-lineage pair < that of a different-lineage pair).

    0.5 = no relation between tree and labels, 1.0 = perfect. Nesting-agnostic, so it is safe
    for hierarchical Pango names. Computed by Mann-Whitney on sampled tip pairs.
    """
    names = [nm for nm in tree.leaf_names if nm in labels]
    sub = tree if len(names) == tree.n_leaves else tree.pruned(names)
    nm, D = sub.patristic()
    lab = np.array([labels[x] for x in nm])
    n = len(nm)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d = D[i, j]
    same = lab[i] == lab[j]
    if same.sum() == 0 or (~same).sum() == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([d[same], d[~same]]))
    n1, n2 = int(same.sum()), int((~same).sum())
    u = r[:n1].sum() - n1 * (n1 + 1) / 2          # large U = same-pairs have larger distances
    return float(1.0 - u / (n1 * n2))


def strict_monophyly(tree: Tree, labels: dict[str, str], min_size: int = 2) -> dict:
    """Fraction of label groups (>=min_size tips) that form exactly one clade (any rooting)."""
    names = [nm for nm in tree.leaf_names if nm in labels]
    sub = tree if len(names) == tree.n_leaves else tree.pruned(names)
    names = sub.leaf_names
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    full = (1 << n) - 1
    splits = sub.split_masks(idx)
    groups: dict[str, int] = {}
    for nm in names:
        groups[labels[nm]] = groups.get(labels[nm], 0) | (1 << idx[nm])
    tested, mono = 0, 0
    for m in groups.values():
        k = m.bit_count()
        if k < min_size:
            continue
        tested += 1
        if k >= n - 1 or k == n:          # trivially a clade
            mono += 1
            continue
        mm = m ^ full if m & 1 else m
        mono += mm in splits
    return dict(n_groups=tested, n_monophyletic=mono, fraction=mono / tested if tested else float("nan"))


def truncate_lineage(label: str, depth: int) -> str:
    """'B.1.617.2' -> depth 3 -> 'B.1.617'.  Crude parent-lineage grouping (aliases not resolved)."""
    return ".".join(label.split(".")[:depth])


# --------------------------------------------------------------------------
# Stats helpers used by the benchmark report
# --------------------------------------------------------------------------

def mean_ci(x, conf: float = 0.95, bounds: tuple[float, float] | None = None,
            log: bool = False) -> tuple[float, float, float]:
    """(centre, lower, upper) t-interval; degenerate for n<2.

    bounds: clip the interval to the metric's valid range (e.g. (0, 1) for a fraction).
    log:    interval on the log scale, returned as geometric mean / back-transformed limits.
            Use for strictly positive, right-skewed quantities (time, memory) so the lower
            limit can never go negative.
    """
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if log:
        x = np.log(np.clip(x, 1e-12, None))
    m = x.mean()
    if len(x) < 2:
        lo = hi = m
    else:
        h = stats.sem(x) * stats.t.ppf((1 + conf) / 2, len(x) - 1)
        lo, hi = m - h, m + h
    if log:
        m, lo, hi = np.exp(m), np.exp(lo), np.exp(hi)
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return (float(m), float(lo), float(hi))


def signflip_wilcoxon(d, max_exact: int = 20) -> tuple[float, int]:
    """Two-sided paired Wilcoxon signed-rank test on differences ``d``; returns (p, n_nonzero).

    Zero differences are dropped (Wilcoxon's convention). With <= ``max_exact`` non-zero
    differences the p-value is computed EXACTLY by enumerating all sign assignments
    (valid with ties, unlike scipy's exact mode, which silently falls back to a normal
    approximation when zeros are present - unreliable for n < ~10). Otherwise scipy's
    normal approximation is used, where n is large enough for it to be fine.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.abs(d) > 1e-12]
    k = len(d)
    if k == 0:
        return (np.nan, 0)
    if k > max_exact:
        return (float(stats.wilcoxon(d, method="approx").pvalue), k)
    ranks = stats.rankdata(np.abs(d))
    signs = (np.arange(2 ** k)[:, None] >> np.arange(k)) & 1          # all 2^k sign patterns
    t_all = (signs * ranks).sum(axis=1)
    t_obs = ranks[d > 0].sum()
    centre = ranks.sum() / 2
    p = float(np.mean(np.abs(t_all - centre) >= abs(t_obs - centre) - 1e-9))
    return (p, k)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, k in enumerate(order):
        running = max(running, (m - rank) * p[k])
        adj[k] = min(running, 1.0)
    return adj.tolist()

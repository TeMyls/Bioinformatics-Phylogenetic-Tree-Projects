"""Outbreak simulator: serially-sampled birth-death tree -> K80+Gamma sequence evolution.

Gives (a) a *known true tree* for benchmarking accuracy (Project A) and (b) sampling dates
with a known clock rate for testing root-to-tip / dating code (Project B).
Defaults mimic SARS-CoV-2: 29.9 kb genome, ~1e-3 subs/site/year, sampled over ~2 years.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .seqio import Alignment
from .tree import Tree

_TRANSITION = np.array([2, 3, 0, 1], dtype=np.uint8)           # A<->G, C<->T
_TRANSVERSIONS = np.array([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=np.uint8)
_BASE_FREQ = np.array([0.30, 0.18, 0.20, 0.32])                 # roughly SARS-CoV-2 composition


@dataclass
class SimResult:
    aln: Alignment
    tree_time: Tree                  # branch lengths in years
    tree_subs: Tree                  # branch lengths in expected subs/site (clock rate x time x multiplier)
    tip_dates: dict[str, float]      # decimal years
    lineages: dict[str, str]         # pseudo-Pango labels
    rate: float
    branch_mutations: dict[int, int]
    root_date: float = float('nan')   # true tMRCA of the sample (decimal year)


# --------------------------------------------------------------------------
# Tree simulation
# --------------------------------------------------------------------------

def _birth_death_sampling(n_tips, birth, death, sampling, rng, max_events=5_000_000):
    parent, end, sampled = [-1], [0.0], [False]
    active = [0]
    t, nsamp, events = 0.0, 0, 0
    tot_rate = birth + death + sampling
    while nsamp < n_tips and active and events < max_events:
        events += 1
        k = len(active)
        t += rng.exponential(1.0 / (k * tot_rate))
        j = int(rng.integers(k))
        node = active[j]
        u = rng.random() * tot_rate
        if u < birth:
            for _ in range(2):
                parent.append(node); end.append(0.0); sampled.append(False)
            end[node] = t
            active[j] = len(parent) - 2
            active.append(len(parent) - 1)
        elif u < birth + death:
            active[j] = active[-1]
            active.pop()
        else:
            end[node] = t
            sampled[node] = True
            nsamp += 1
            active[j] = active[-1]
            active.pop()
    return parent, end, sampled, nsamp


def simulate_tree(n_tips: int, birth: float = 2.0, death: float = 0.3, sampling: float = 0.4,
                  span_years: float = 2.0, seed: int = 0, max_attempts: int = 100):
    """Return (Tree with lengths in years, {tip: years since first sample start}, rng)."""
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        parent, end, sampled, nsamp = _birth_death_sampling(n_tips, birth, death, sampling, rng)
        if nsamp == n_tips:
            break
    else:
        raise RuntimeError("birth-death simulation kept going extinct; raise birth or lower death")
    N = len(parent)
    has = list(sampled)
    for v in range(N - 1, 0, -1):
        if has[v]:
            has[parent[v]] = True
    kids: dict[int, list[int]] = {}
    for v in range(1, N):
        if has[v]:
            kids.setdefault(parent[v], []).append(v)
    kept = [v for v in range(N) if has[v] and (sampled[v] or len(kids.get(v, [])) >= 2)]
    kept_set = set(kept)
    anc: dict[int, int | None] = {}
    for v in range(N):
        if not has[v]:
            continue
        p = parent[v]
        anc[v] = None if p < 0 else (p if p in kept_set else anc[p])
    scale = span_years / max(end[v] for v in kept if sampled[v])
    t = Tree()
    tid: dict[int, int] = {}
    tip_times: dict[str, float] = {}
    for v in kept:
        if sampled[v]:
            nm = f"seq{len(tip_times):05d}"
            tid[v] = t.new_node(nm)
            tip_times[nm] = end[v] * scale
        else:
            tid[v] = t.new_node()
    roots = [v for v in kept if anc[v] is None]
    assert len(roots) == 1, roots
    for v in kept:
        a = anc[v]
        if a is not None:
            t.add_edge(tid[a], tid[v], (end[v] - end[a]) * scale)
    t._suppress_root_node(tid[roots[0]])
    # shift so the earliest tip has date offset 0 is NOT done: root is at time end[root]*scale
    root_time = end[roots[0]] * scale
    return t, tip_times, root_time, rng


# --------------------------------------------------------------------------
# Sequence evolution
# --------------------------------------------------------------------------

def simulate_alignment(tree_time: Tree, length: int = 29_903, rate: float = 1e-3, kappa: float = 4.0,
                       gamma_shape: float = 0.5, relaxed_sd: float = 0.0, seed: int = 0):
    """Evolve sequences down ``tree_time`` (lengths in years). Returns (Alignment, tree_subs, mut_counts)."""
    rng = np.random.default_rng(seed)
    w = rng.gamma(gamma_shape, 1.0, size=length) if gamma_shape else np.ones(length)
    cumw = np.cumsum(w / w.sum())
    p_ts = kappa / (kappa + 2.0)
    root_seq = rng.choice(4, size=length, p=_BASE_FREQ).astype(np.uint8)

    view = tree_time.rooted()
    tree_subs = tree_time.copy()
    seqs: dict[int, np.ndarray] = {view.root: root_seq}
    remaining = {v: len(view.children[v]) for v in view.order}
    muts: dict[int, int] = {}
    leaf_rows: dict[int, np.ndarray] = {}
    esubs: dict[int, float] = {}
    for v in view.order:
        if v == view.root:
            continue
        p = view.parent[v]
        mult = rng.lognormal(-relaxed_sd**2 / 2, relaxed_sd) if relaxed_sd > 0 else 1.0
        exp_subs = view.plen[v] * rate * mult
        k = rng.poisson(exp_subs * length)
        muts[v] = int(k)
        s = seqs[p].copy()
        if k:
            sites = np.minimum(np.searchsorted(cumw, rng.random(k)), length - 1)
            cur = s[sites]
            is_ts = rng.random(k) < p_ts
            new = np.where(is_ts, _TRANSITION[cur],
                           _TRANSVERSIONS[cur, rng.integers(0, 2, size=k)])
            s[sites] = new
        seqs[v] = s
        remaining[p] -= 1
        if remaining[p] == 0 and p != view.root:
            del seqs[p]
        if v in tree_time.name:
            leaf_rows[v] = s
            del seqs[v]
        esubs[v] = exp_subs
        if p != -1:
            tree_subs.adj[v][p] = tree_subs.adj[p][v] = exp_subs
    if tree_time.root_edge is not None:               # root edge = two halves with their own multipliers
        a, b, _ = tree_time.root_edge
        tree_subs.adj[a][b] = tree_subs.adj[b][a] = esubs[a] + esubs[b]
        tree_subs.root_edge = (a, b, esubs[a])
    ids = sorted(leaf_rows)
    names = [tree_time.name[i] for i in ids]
    aln = Alignment(names, np.vstack([leaf_rows[i] for i in ids]))
    return aln, tree_subs, muts


# --------------------------------------------------------------------------
# Pseudo-Pango lineage labels
# --------------------------------------------------------------------------

def assign_pseudo_lineages(tree: Tree, muts: dict[int, int], min_size: int = 5, min_muts: int = 2) -> dict[str, str]:
    """Hierarchical lineage names (A, A.1, A.1.2 ...) on the true rooted tree.

    A node founds a new lineage if its branch carries >= min_muts mutations and it has
    >= min_size tips -- a rough analogue of Pango's designation rules.
    """
    view = tree.rooted()
    label: dict[int, str] = {view.root: "A"}
    counter: dict[str, int] = {}
    out: dict[str, str] = {}
    for v in view.order:
        if v == view.root:
            continue
        p = view.parent[v]
        lab = label[p]
        n_tips = view.hi[v] - view.lo[v]
        if v not in tree.name and muts.get(v, 0) >= min_muts and n_tips >= min_size:
            counter[lab] = counter.get(lab, 0) + 1
            lab = f"{lab}.{counter[lab]}"
        label[v] = lab
        if v in tree.name:
            out[tree.name[v]] = lab
    return out


# --------------------------------------------------------------------------
# One-call convenience
# --------------------------------------------------------------------------

def simulate_dataset(n_tips: int, length: int = 29_903, rate: float = 1e-3, span_years: float = 2.0,
                     start_year: float = 2020.0, relaxed_sd: float = 0.0, seed: int = 0,
                     min_lineage_size: int = 5) -> SimResult:
    tree_t, tip_times, root_time, _ = simulate_tree(n_tips, span_years=span_years, seed=seed)
    aln, tree_s, muts = simulate_alignment(tree_t, length=length, rate=rate, relaxed_sd=relaxed_sd,
                                           seed=seed + 10_000)
    lineages = assign_pseudo_lineages(tree_t, muts, min_size=min_lineage_size)
    dates = {nm: start_year + t for nm, t in tip_times.items()}
    return SimResult(aln, tree_t, tree_s, dates, lineages, rate, muts, start_year + root_time)

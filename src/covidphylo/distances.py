"""Pairwise genetic distances from an Alignment (vectorised, handles ambiguity)."""
from __future__ import annotations

import numpy as np

from .seqio import Alignment, AMBIG


def mismatch_counts(aln: Alignment):
    """Return (mismatches, comparable_sites), both (n x n) float64.

    Pairwise-deletion: a site only counts for pair (i, j) if both are A/C/G/T.
    Cost is dominated by *variable* columns, so it scales well to 30 kb genomes.
    """
    X = aln.X
    n, L = X.shape
    valid = X != AMBIG

    # comparable sites = L - inv_i - inv_j + both_invalid_ij   (only columns with any invalid)
    inv_cols = np.flatnonzero(~valid.all(axis=0))
    inv = (~valid[:, inv_cols]).astype(np.float32)
    inv_count = inv.sum(axis=1).astype(np.float64)
    both_inv = (inv @ inv.T).astype(np.float64)
    comparable = L - inv_count[:, None] - inv_count[None, :] + both_inv

    # mismatches only arise in variable columns
    var_cols = aln.variable_columns()
    Xv = X[:, var_cols]
    Vv = (Xv != AMBIG).astype(np.float32)
    pair_valid = (Vv @ Vv.T).astype(np.float64)
    matches = np.zeros((n, n), dtype=np.float64)
    for b in range(4):
        Ib = (Xv == b).astype(np.float32)
        matches += (Ib @ Ib.T)
    mism = pair_valid - matches
    np.fill_diagonal(mism, 0.0)
    return mism, comparable


def distance_matrix(aln: Alignment, model: str = "jc") -> np.ndarray:
    """p-distance ('p') or Jukes-Cantor-corrected distance ('jc') in subs/site."""
    mism, comp = mismatch_counts(aln)
    p = np.divide(mism, comp, out=np.zeros_like(mism), where=comp > 0)
    if model == "p":
        return p
    if model == "jc":
        arg = 1.0 - (4.0 / 3.0) * p
        arg = np.clip(arg, 1e-12, None)          # saturated pairs -> large finite distance
        d = -0.75 * np.log(arg)
        np.fill_diagonal(d, 0.0)
        return d
    raise ValueError(f"unknown model {model!r} (use 'p' or 'jc')")

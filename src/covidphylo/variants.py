"""Variant discovery from a reference-coordinate alignment (SNPs vs reference, lineage-defining sites).

Insertions/deletions are NOT called here: gaps are treated as missing (code 4). For indel calling use
the aligner's output (e.g. nextclade) or a variant caller on the raw reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .seqio import AMBIG, Alignment, encode, read_fasta

BASES = np.array(list("ACGT"))


def consensus(aln: Alignment) -> np.ndarray:
    """Majority base per column (ambiguous ignored). Used as the reference when none is supplied."""
    counts = np.stack([(aln.X == b).sum(axis=0) for b in range(4)])
    return counts.argmax(axis=0).astype(np.uint8)


def load_reference(path, length: int | None = None) -> np.ndarray:
    """First record of a FASTA file as a uint8 code vector (must match the alignment length)."""
    name, seq = next(iter(read_fasta(path).items()))
    ref = encode(seq)
    if length is not None and len(ref) != length:
        raise ValueError(f"reference {name} has length {len(ref)} but the alignment has {length}")
    return ref


def site_table(aln: Alignment, ref: np.ndarray) -> pd.DataFrame:
    """One row per *variable* position: reference base, count of each base, alt allele and its frequency."""
    cols = aln.variable_columns()
    X = aln.X[:, cols]
    counts = np.stack([(X == b).sum(axis=0) for b in range(4)], axis=1)       # (sites x 4)
    valid = counts.sum(axis=1)
    refc = ref[cols]
    alt_counts = counts.copy()
    ok = refc < 4
    alt_counts[np.arange(len(cols))[ok], refc[ok]] = 0
    alt = alt_counts.argmax(axis=1)
    return pd.DataFrame({
        "pos": cols + 1, "ref": np.where(ok, BASES[np.minimum(refc, 3)], "N"),
        "alt": BASES[alt], "n_valid": valid, "n_alt": alt_counts.max(axis=1),
        "alt_freq": alt_counts.max(axis=1) / np.maximum(valid, 1),
        **{f"n_{b}": counts[:, i] for i, b in enumerate("ACGT")},
    })


def call_snps(aln: Alignment, ref: np.ndarray) -> pd.DataFrame:
    """Long-format SNP calls: one row per (sample, position) where the sample differs from the reference."""
    cols = aln.variable_columns()
    X = aln.X[:, cols]
    diff = (X != ref[cols][None, :]) & (X != AMBIG) & (ref[cols][None, :] < 4)
    i, j = np.nonzero(diff)
    return pd.DataFrame({"sample": np.asarray(aln.names)[i], "pos": cols[j] + 1,
                         "ref": BASES[ref[cols][j]], "alt": BASES[X[i, j]]})


def snps_per_sample(calls: pd.DataFrame, aln: Alignment) -> pd.Series:
    s = calls.groupby("sample").size().reindex(aln.names).fillna(0).astype(int)
    s.name = "n_snps_vs_reference"
    return s


def lineage_defining_sites(aln: Alignment, ref: np.ndarray, labels: dict[str, str], min_in: float = 0.9,
                           max_out: float = 0.05, min_n: int = 5, hierarchical: bool = True) -> pd.DataFrame:
    """Positions where a lineage carries a non-reference allele at >= min_in frequency but the rest at <= max_out.

    hierarchical=True treats sub-lineages (label starts with 'L.') as *inside* the lineage, because they inherit its
    mutations. Pango aliases (e.g. AY.4 = B.1.617.2.4) are not resolved, so real-data results for aliased
    lineages are conservative.
    """
    names = [n for n in aln.names if n in labels]
    idx = {n: i for i, n in enumerate(aln.names)}
    rows_i = np.array([idx[n] for n in names])
    lab = np.array([labels[n] for n in names])
    cols = aln.variable_columns()
    X = aln.X[np.ix_(rows_i, cols)]
    refc = ref[cols]
    nonref = (X != refc[None, :]) & (X != AMBIG) & (refc[None, :] < 4)
    valid = (X != AMBIG)
    out = []
    for L in sorted(set(lab)):
        inside = (lab == L) | np.char.startswith(lab.astype(str), L + ".") if hierarchical else (lab == L)
        if inside.sum() < min_n or (~inside).sum() < min_n:
            continue
        f_in = nonref[inside].sum(0) / np.maximum(valid[inside].sum(0), 1)
        f_out = nonref[~inside].sum(0) / np.maximum(valid[~inside].sum(0), 1)
        hit = np.flatnonzero((f_in >= min_in) & (f_out <= max_out))
        for h in hit:
            col = X[inside, h]
            alt = int(np.bincount(col[(col != refc[h]) & (col != AMBIG)], minlength=4).argmax())
            out.append(dict(lineage=L, pos=int(cols[h]) + 1, ref=BASES[refc[h]], alt=BASES[alt],
                            freq_in=float(f_in[h]), freq_out=float(f_out[h]), n_in=int(inside.sum())))
    return pd.DataFrame(out, columns=["lineage", "pos", "ref", "alt", "freq_in", "freq_out", "n_in"])

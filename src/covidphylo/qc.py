"""Quality control, site masking and subsampling (shared by both projects)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .seqio import Alignment

# --------------------------------------------------------------------------
# Raw-sequence QC (before alignment)
# --------------------------------------------------------------------------

def qc_raw_sequences(
    seqs: dict[str, str],
    min_length: int = 29_000,
    max_length: int = 30_500,
    max_ambig_frac: float = 0.01,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Drop sequences that are too short/long or have too many non-ACGT characters.

    Returns (kept sequences, per-sequence report incl. the reason for removal).
    """
    keep, rows = {}, []
    for n, s in seqs.items():
        L = len(s)
        u = s.upper()
        n_acgt = u.count("A") + u.count("C") + u.count("G") + u.count("T")
        frac = 1 - n_acgt / L if L else 1.0
        reason = ""
        if L < min_length:
            reason = "too_short"
        elif L > max_length:
            reason = "too_long"
        elif frac > max_ambig_frac:
            reason = "too_many_ambiguous"
        rows.append(dict(name=n, length=L, ambig_frac=frac, removed=reason))
        if not reason:
            keep[n] = s
    return keep, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------

def read_problematic_sites_vcf(path, filters=("mask",)) -> list[int]:
    """Parse the De Maio et al. 'problematic sites' VCF -> sorted 0-based positions.

    Only rows whose FILTER column is in ``filters`` are returned
    (use filters=("mask","caution") for a more aggressive mask).
    """
    sites = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) >= 7 and f[6] in filters:
                sites.add(int(f[1]) - 1)
    return sorted(sites)


def terminal_sites(length: int, five_prime: int = 55, three_prime: int = 100) -> list[int]:
    """0-based positions of the 5' and 3' ends (low coverage / UTR artefacts)."""
    return list(range(five_prime)) + list(range(max(length - three_prime, 0), length))


def build_mask(length: int, vcf_sites: list[int] | None = None,
               five_prime: int = 55, three_prime: int = 100) -> list[int]:
    return sorted(set(terminal_sites(length, five_prime, three_prime)) | set(vcf_sites or []))


# --------------------------------------------------------------------------
# Aligned-sequence QC
# --------------------------------------------------------------------------

def drop_ambiguous(aln: Alignment, max_ambig_frac: float = 0.05) -> Alignment:
    """Remove sequences with too many ambiguous/gap characters *after* masking."""
    ok = np.flatnonzero(aln.ambiguous_fraction() <= max_ambig_frac)
    return aln.subset(ok)


def deduplicate(aln: Alignment, prefer_order: list[int] | None = None):
    """Collapse identical sequences (identical on unmasked sites).

    ``prefer_order`` ranks rows (e.g. by sampling date); the first-ranked member
    is kept as representative. Returns (alignment, {representative: [members]}).
    """
    order = list(range(aln.n)) if prefer_order is None else list(prefer_order)
    seen: dict[bytes, int] = {}
    groups: dict[int, list[int]] = {}
    for i in order:
        key = aln.X[i].tobytes()
        if key in seen:
            groups[seen[key]].append(i)
        else:
            seen[key] = i
            groups[i] = [i]
    reps = sorted(groups)
    out = aln.subset(reps)
    return out, {aln.names[r]: [aln.names[m] for m in groups[r]] for r in reps}


# --------------------------------------------------------------------------
# Metadata + subsampling
# --------------------------------------------------------------------------

def read_metadata(path, name_col="strain", date_col="date", lineage_col="pango_lineage") -> pd.DataFrame:
    """Read a TSV/CSV metadata file; returns a frame with columns name/date/lineage (+ the rest)."""
    sep = "\t" if str(path).endswith((".tsv", ".tsv.gz", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    miss = [c for c in (name_col, date_col, lineage_col) if c not in df.columns]
    if miss:
        raise KeyError(f"Metadata is missing column(s) {miss}; available: {list(df.columns)[:15]} ...")
    df = df.rename(columns={name_col: "name", date_col: "date", lineage_col: "lineage"})
    return df


def subsample_stratified(meta: pd.DataFrame, n: int, seed: int = 0,
                         max_per_lineage: int | None = None) -> pd.DataFrame:
    """Draw ~n rows spread as evenly as possible over lineages (rare lineages kept whole)."""
    rng = np.random.default_rng(seed)
    meta = meta.dropna(subset=["lineage"])
    meta = meta[~meta["lineage"].isin(["None", "Unassigned", ""])]
    groups = {k: g.index.to_numpy() for k, g in meta.groupby("lineage")}
    cap = max_per_lineage or max(1, int(np.ceil(n / max(len(groups), 1))))
    # water-filling: raise the per-lineage cap until we have n rows
    sizes = np.array([len(v) for v in groups.values()])
    lo, hi = 1, int(sizes.max()) if len(sizes) else 1
    while lo < hi:
        mid = (lo + hi) // 2
        if np.minimum(sizes, mid).sum() >= n:
            hi = mid
        else:
            lo = mid + 1
    cap = min(cap, lo) if max_per_lineage else lo
    picked = []
    for idx in groups.values():
        k = min(len(idx), cap)
        picked.extend(rng.choice(idx, size=k, replace=False))
    picked = np.array(picked)
    if len(picked) > n:
        picked = rng.choice(picked, size=n, replace=False)
    return meta.loc[np.sort(picked)]

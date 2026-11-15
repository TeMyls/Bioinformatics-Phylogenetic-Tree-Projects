"""Exploratory data analysis + QC summaries for a SARS-CoV-2 alignment (shared by both projects).

Everything here is drawn with matplotlib from the data - suitable for the 'exploratory analysis' and
'quality-control' slides/figures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .clock import date_to_decimal_year
from .distances import distance_matrix
from .plotting import PALETTE, plt
from .seqio import AMBIG, Alignment


def qc_table(aln: Alignment, mask_sites: list[int] | None = None, max_ambig: float = 0.05) -> pd.DataFrame:
    """Per-sequence QC: ambiguous fraction before/after masking and pass/fail."""
    before = aln.ambiguous_fraction()
    masked = aln.mask_sites(mask_sites) if mask_sites else aln
    after = masked.ambiguous_fraction()
    return pd.DataFrame({"name": aln.names, "ambig_before_mask": before, "ambig_after_mask": after,
                         "pass": after <= max_ambig})


def eda_summary(aln: Alignment, meta: pd.DataFrame | None, mask_sites: list[int] | None = None) -> dict:
    var = aln.variable_columns()
    s = {"n_sequences": aln.n, "alignment_length": aln.length, "n_variable_sites": int(len(var)),
         "variable_site_fraction": float(len(var) / aln.length),
         "median_ambig_fraction": float(np.median(aln.ambiguous_fraction())),
         "n_masked_sites": len(set(mask_sites or []))}
    if meta is not None:
        m = meta[meta["name"].isin(set(aln.names))]
        d = m["date"].map(date_to_decimal_year)
        s.update(n_lineages=int(m["lineage"].nunique()), n_complete_dates=int(d.notna().sum()),
                 date_min=float(d.min()), date_max=float(d.max()))
    return s


def eda_panel(aln: Alignment, meta: pd.DataFrame | None, outdir: str | Path,
              mask_sites: list[int] | None = None, max_pairs_seqs: int = 300, seed: int = 0) -> Path:
    """2x3 EDA figure: lineage sizes, sampling dates, ambiguity, SNP density, pairwise distances, QC pass."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.0))

    # (a) lineage sizes
    a = ax[0, 0]
    if meta is not None:
        m = meta[meta["name"].isin(set(aln.names))]
        vc = m["lineage"].value_counts()
        top = vc.head(15)
        a.barh(range(len(top)), top.values, color=PALETTE[0])
        a.set_yticks(range(len(top)))
        a.set_yticklabels(top.index, fontsize=6)
        a.invert_yaxis()
        a.set_xlabel("Sequences")
        a.set_title(f"Top 15 of {len(vc)} Pango lineages")
    else:
        a.axis("off")

    # (b) sampling dates
    b = ax[0, 1]
    if meta is not None:
        d = m["date"].map(date_to_decimal_year).dropna()
        b.hist(d, bins=40, color=PALETTE[2])
        b.xaxis.set_major_locator(MaxNLocator(5))
        b.set_xlabel("Collection date (year)")
        b.set_ylabel("Sequences")
        b.set_title("Sampling dates")
    else:
        b.axis("off")

    # (c) ambiguous fraction, before vs after masking
    c = ax[0, 2]
    qc = qc_table(aln, mask_sites)
    bins = np.linspace(0, max(0.05, qc["ambig_before_mask"].max()), 40)
    c.hist(qc["ambig_before_mask"], bins=bins, alpha=0.7, color=PALETTE[1], label="before masking")
    c.hist(qc["ambig_after_mask"], bins=bins, alpha=0.7, color=PALETTE[0], label="after masking")
    c.set_yscale("log")
    c.set_xlabel("Ambiguous-base fraction per sequence")
    c.set_ylabel("Sequences (log)")
    c.legend(fontsize=7)
    c.set_title("Sequence quality")

    # (d) SNP density along the genome
    d_ = ax[1, 0]
    counts = np.stack([(aln.X == k).sum(axis=0) for k in range(4)])
    minor = counts.sum(axis=0) - counts.max(axis=0)
    binsize = 500
    nb = aln.length // binsize + 1
    dens = np.bincount(np.arange(aln.length) // binsize, weights=(minor > 0).astype(float), minlength=nb)
    d_.step(np.arange(nb) * binsize / 1000, dens, where="post", color=PALETTE[0], lw=0.9)
    if mask_sites:
        ms = np.array(sorted(set(mask_sites)))
        d_.plot(ms / 1000, np.full(len(ms), -0.04 * dens.max()), "|", color=PALETTE[1], ms=3, label="masked")
        d_.legend(fontsize=7)
    d_.set_xlabel("Genome position (kb)")
    d_.set_ylabel(f"Variable sites per {binsize} bp")
    d_.set_title("Variation along the genome")

    # (e) pairwise distance distribution
    e = ax[1, 1]
    sub = aln.subset(rng.choice(aln.n, size=min(aln.n, max_pairs_seqs), replace=False))
    D = distance_matrix(sub.mask_sites(mask_sites) if mask_sites else sub, "p")
    iu = np.triu_indices(sub.n, 1)
    e.hist(D[iu] * aln.length, bins=50, color=PALETTE[3])
    e.set_xlabel("Pairwise differences (SNPs)")
    e.set_ylabel("Sequence pairs")
    e.set_title(f"Pairwise distance (n={sub.n} sampled)")

    # (f) QC pass/fail
    f = ax[1, 2]
    n_pass = int(qc["pass"].sum())
    f.bar(["pass", "fail"], [n_pass, len(qc) - n_pass], color=[PALETTE[2], PALETTE[1]])
    f.set_ylabel("Sequences")
    f.set_title("QC outcome (<=5% N after mask)")

    for panel, letter in zip(ax.ravel(), "abcdef"):
        panel.text(-0.14, 1.06, letter, transform=panel.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout()
    p = out / "fig_eda.png"
    fig.savefig(p)
    plt.close(fig)
    return p

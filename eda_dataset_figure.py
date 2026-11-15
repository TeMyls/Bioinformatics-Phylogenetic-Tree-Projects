#!/usr/bin/env python3
"""Exploratory figure for the real dataset: genomes per variant and collection-date span per variant.

Two inputs (use either):
  --metadata data/raw/genome_metadata.tsv   (from dataset_summary.py) -> a dot for EVERY genome's collection
                                             date, so outliers are visible (best for the report)
  --summary  data/raw/dataset_summary.md    -> date-range bars only (what you get if you only have the summary)

Output: PNG (matplotlib). The dashed vertical line marks the first reports of Omicron (Nov 2021);
Omicron genomes dated before it point to a metadata error or a mislabelled lineage.
Requires matplotlib (and numpy).

Usage:
    python scripts/eda_dataset_figure.py --metadata data/raw/genome_metadata.tsv --out results/fig_eda_dataset.png
"""
import argparse
import csv
import datetime as dt
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["Alpha", "Beta", "Delta", "Omicron", "Other"]
TEAL, INK, WARM, GREY = "#1F6F7C", "#12263A", "#B0432C", "#6B7683"
OMICRON_FIRST = 2021 + (dt.date(2021, 11, 24).timetuple().tm_yday - 1) / 365.25   # first public report
DATE = r"\d{4}(?:-\d{2}(?:-\d{2})?)?"


def dec_year(s, end=False):
    """'YYYY[-MM[-DD]]' -> decimal year; for a partial date, start of the period (end=False) or its end."""
    p = [int(x) for x in s.split("-")]
    y, m, d = p[0], (p[1] if len(p) > 1 else 1), (p[2] if len(p) > 2 else 1)
    base = dt.date(y, m, d)
    val = y + (base.timetuple().tm_yday - 1) / 365.25
    if end and len(p) < 3:
        nxt = dt.date(y + 1, 1, 1) if len(p) == 1 else (dt.date(y + (m == 12), m % 12 + 1, 1))
        val = y + ((nxt - dt.date(y, 1, 1)).days) / 365.25
    return val


def from_summary(path):
    counts, spans = {}, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in VARIANTS and cells[1].isdigit():
            counts[cells[0]] = int(cells[1])
            ds = re.findall(DATE, cells[2])
            if ds:
                spans[cells[0]] = (ds[0], ds[-1])
    return counts, spans, None


def from_metadata(path):
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"), delimiter="\t")
            if r["qc_pass"] == "1" and not r["duplicate_of"]]
    counts = {v: sum(r["variant"] == v for r in rows) for v in VARIANTS}
    dates = {v: [dec_year(r["collection_date"]) for r in rows
                 if r["variant"] == v and re.fullmatch(DATE, r["collection_date"] or "")] for v in VARIANTS}
    return counts, None, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata")
    ap.add_argument("--summary")
    ap.add_argument("--out", default="fig_eda_dataset.png")
    a = ap.parse_args()
    if not (a.metadata or a.summary):
        ap.error("give --metadata or --summary")
    counts, spans, dates = from_metadata(a.metadata) if a.metadata else from_summary(a.summary)

    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.9], "wspace": 0.08})
    ypos = np.arange(len(VARIANTS))[::-1]
    ax1.barh(ypos, [counts.get(v, 0) for v in VARIANTS], color=TEAL, height=0.6)
    for y, v in zip(ypos, VARIANTS):
        ax1.text(counts.get(v, 0) + 2, y, str(counts.get(v, 0)), va="center", color=INK, fontsize=14)
    ax1.set_yticks(ypos, VARIANTS)
    ax1.set_xlim(0, max(counts.values()) * 1.25)
    ax1.set_xlabel("Genomes (after QC)")

    if dates:   # every genome's date as a dot
        rng = np.random.default_rng(0)
        for y, v in zip(ypos, VARIANTS):
            x = np.array(dates[v])
            ax2.scatter(x, y + rng.uniform(-0.22, 0.22, len(x)), s=14, color=TEAL, alpha=0.7, lw=0)
    else:       # range bars; dotted = year/month-only date, exact day unknown
        for y, v in zip(ypos, VARIANTS):
            if v not in spans:
                continue
            s, e = spans[v]
            x0, x1, x1_max = dec_year(s), dec_year(e), dec_year(e, end=True)
            ax2.barh(y, max(x1 - x0, 0.03), left=x0, color=TEAL, height=0.6)
            if x1_max - x1 > 0.05:
                ax2.plot([x1, x1_max], [y, y], color=TEAL, lw=3, ls=":")
    ax2.axvline(OMICRON_FIRST, color=WARM, ls="--", lw=1.6)
    ax2.text(OMICRON_FIRST + 0.05, ypos[0] + 0.55, "Omicron first\nreported, Nov 2021", color=WARM,
             fontsize=12.5, va="bottom")
    ax2.set_xlim(2019.9, 2025.1)
    ax2.set_ylim(-0.6, ypos[0] + 1.35)
    ax2.set_xticks(range(2020, 2026))
    ax2.set_xlabel("Collection date" + ("" if dates else " (dotted: year-only or month-only date)"))
    ax2.tick_params(axis="y", left=False)
    ax2.grid(axis="x", color="#E3E0D6", lw=0.8)
    ax2.set_axisbelow(True)
    fig.savefig(a.out, dpi=250, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pick a lineage-stratified subset of genomes BEFORE alignment (aligning millions of genomes is not feasible).

Writes
  <out>/subsample.fasta             reference first, then the chosen genomes (ready for MAFFT / nextalign)
  <out>/subsample_provenance.tsv    name, date, lineage of every chosen genome  -> cite this as your data provenance

Example:
    python scripts/subsample_fasta.py --metadata data/metadata.tsv --sequences data/sequences.fasta \
        --reference data/NC_045512.2.fasta --n 1000 --seed 1 --out-dir data/subsample
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402

from covidphylo import qc                             # noqa: E402
from covidphylo.clock import date_to_decimal_year     # noqa: E402
from covidphylo.seqio import iter_fasta, read_fasta, write_fasta  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--sequences", required=True, help="unaligned multi-FASTA (can be .gz)")
    ap.add_argument("--reference", required=True, help="Wuhan-Hu-1 FASTA; written first in the output")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "subsample"))
    ap.add_argument("--name-col", default="strain")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--lineage-col", default="pango_lineage")
    ap.add_argument("--date-from", type=float, default=None, help="decimal year lower bound, e.g. 2021.0")
    ap.add_argument("--date-to", type=float, default=None)
    ap.add_argument("--min-length", type=int, default=29_000)
    ap.add_argument("--max-ambig", type=float, default=0.01)
    a = ap.parse_args()

    meta = qc.read_metadata(a.metadata, a.name_col, a.date_col, a.lineage_col)
    meta["dec"] = meta["date"].map(date_to_decimal_year)
    meta = meta[np.isfinite(meta["dec"])]                       # complete dates only (needed for the clock)
    if a.date_from is not None:
        meta = meta[meta["dec"] >= a.date_from]
    if a.date_to is not None:
        meta = meta[meta["dec"] <= a.date_to]
    # oversample 1.5x so QC drops don't leave us short, then filter on the actual sequences
    cand = qc.subsample_stratified(meta, int(a.n * 1.5), seed=a.seed)
    want = set(cand["name"])
    seqs = {}
    for name, s in iter_fasta(a.sequences):
        if name in want:
            seqs[name] = s
            if len(seqs) == len(want):
                break
    seqs, _report = qc.qc_raw_sequences(seqs, min_length=a.min_length, max_ambig_frac=a.max_ambig)
    keep = list(seqs)[: a.n]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = read_fasta(a.reference)
    write_fasta(list(ref.items())[:1] + [(k, seqs[k]) for k in keep], out / "subsample.fasta")
    prov = cand[cand["name"].isin(keep)][["name", "date", "lineage"]]
    prov.to_csv(out / "subsample_provenance.tsv", sep="\t", index=False)
    print(f"wrote {len(keep)} genomes (+reference) to {out}/subsample.fasta; provenance in subsample_provenance.tsv")


if __name__ == "__main__":
    main()

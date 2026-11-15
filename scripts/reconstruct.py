#!/usr/bin/env python3
"""Project B driver: reconstruct, root and date a SARS-CoV-2 tree.

Simulated demo (known rate and tMRCA, runs anywhere):
    python scripts/reconstruct.py --mode sim --n 300 --method nj_jc

Real data (aligned FASTA in reference coordinates + metadata TSV):
    python scripts/reconstruct.py --mode real --alignment data/aligned.fasta --metadata data/metadata.tsv \
        --method iqtree --mask-vcf data/problematic_sites_sarsCov2.vcf --max-tips 1000
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402

from covidphylo import qc                             # noqa: E402
from covidphylo.clock import date_to_decimal_year     # noqa: E402
from covidphylo.infer import available_methods        # noqa: E402
from covidphylo.reconstruct import run_reconstruction # noqa: E402
from covidphylo.seqio import Alignment                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--method", default="nj_jc", help=f"one of {available_methods()} (available here)")
    ap.add_argument("--out", default=str(ROOT / "results" / "reconstruct"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=500)
    # sim
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--rate", type=float, default=1e-3)
    # real
    ap.add_argument("--alignment")
    ap.add_argument("--metadata")
    ap.add_argument("--name-col", default="strain")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--lineage-col", default="pango_lineage")
    ap.add_argument("--mask-vcf")
    ap.add_argument("--max-tips", type=int, default=1000, help="stratified-by-lineage subsample size")
    ap.add_argument("--max-ambig", type=float, default=0.05)
    a = ap.parse_args()

    truth = None
    if a.mode == "sim":
        from covidphylo.simulate import simulate_dataset
        sim = simulate_dataset(a.n, rate=a.rate, seed=a.seed)
        aln, dates, lineages, truth = sim.aln, sim.tip_dates, sim.lineages, a.rate
        sites = None
        print(f"simulated {aln.n} tips; true rate {a.rate:g}, true tMRCA {sim.root_date:.3f}")
    else:
        if not (a.alignment and a.metadata):
            ap.error("--mode real needs --alignment and --metadata")
        meta = qc.read_metadata(a.metadata, a.name_col, a.date_col, a.lineage_col)
        meta["dec"] = meta["date"].map(date_to_decimal_year)
        meta = meta[np.isfinite(meta["dec"])]                       # drop incomplete dates up front
        sub = qc.subsample_stratified(meta, a.max_tips, seed=a.seed)
        aln = Alignment.from_fasta(a.alignment, keep=set(sub["name"]))
        dates = dict(zip(sub["name"], sub["dec"]))
        lineages = dict(zip(sub["name"], sub["lineage"]))
        sites = qc.build_mask(aln.length, qc.read_problematic_sites_vcf(a.mask_vcf) if a.mask_vcf else None)

    log = run_reconstruction(aln, dates, lineages, a.out, method=a.method, mask_sites=sites,
                             max_ambig_frac=a.max_ambig, seed=a.seed, n_perm=a.n_perm, n_boot=a.n_boot,
                             truth_rate=truth)
    print(f"\nclock rate {log['clock_rate']:.3e} subs/site/yr  (95% CI {log['clock_rate_ci95'][0]:.2e}-{log['clock_rate_ci95'][1]:.2e})")
    print(f"tMRCA {log['tmrca']:.2f}   R2 {log['r2']:.2f}   permutation p = {log['p_permutation']:.4f}")
    if a.mode == "sim":
        print(f"(truth: rate {a.rate:g}, tMRCA {sim.root_date:.2f})")


if __name__ == "__main__":
    main()

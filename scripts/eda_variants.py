#!/usr/bin/env python3
"""Exploratory analysis + variant discovery (shared by both projects; feeds the 'EDA' and 'QC' slides).

Simulated:  python scripts/eda_variants.py --mode sim --n 400
Real:       python scripts/eda_variants.py --mode real --alignment data/subsample/aligned.fasta \
                --metadata data/metadata.tsv --reference data/NC_045512.2.fasta --mask-vcf data/problematic_sites_sarsCov2.vcf
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                   # noqa: E402

from covidphylo import eda, qc, variants              # noqa: E402
from covidphylo.seqio import Alignment                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--out", default=str(ROOT / "results" / "eda"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--alignment")
    ap.add_argument("--metadata")
    ap.add_argument("--reference", help="reference FASTA in alignment coordinates; if the alignment contains "
                                        "NC_045512.2 that row is used and removed")
    ap.add_argument("--ref-name", default="NC_045512.2")
    ap.add_argument("--mask-vcf")
    ap.add_argument("--name-col", default="strain")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--lineage-col", default="pango_lineage")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.mode == "sim":
        from covidphylo.simulate import simulate_dataset
        sim = simulate_dataset(a.n, seed=a.seed)
        aln, labels = sim.aln, sim.lineages
        meta = pd.DataFrame({"name": aln.names, "lineage": [labels[n] for n in aln.names],
                             "date": [f"{int(sim.tip_dates[n])}-{1 + int((sim.tip_dates[n] % 1) * 12):02d}-15" for n in aln.names]})
        ref, mask = variants.consensus(aln), qc.build_mask(aln.length)
    else:
        if not (a.alignment and a.metadata):
            ap.error("--mode real needs --alignment and --metadata")
        aln = Alignment.from_fasta(a.alignment)
        if a.ref_name in aln.names:
            i = aln.names.index(a.ref_name)
            ref = aln.X[i].copy()
            aln = aln.subset([k for k in range(aln.n) if k != i])
        elif a.reference:
            ref = variants.load_reference(a.reference, aln.length)
        else:
            ref = variants.consensus(aln)
            print("[warn] no reference supplied: using the alignment consensus")
        meta = qc.read_metadata(a.metadata, a.name_col, a.date_col, a.lineage_col)
        labels = dict(zip(meta["name"], meta["lineage"]))
        vcf = qc.read_problematic_sites_vcf(a.mask_vcf) if a.mask_vcf else None
        mask = qc.build_mask(aln.length, vcf)

    summary = eda.eda_summary(aln, meta, mask)
    fig = eda.eda_panel(aln, meta, out, mask_sites=mask, seed=a.seed)
    eda.qc_table(aln, mask).to_csv(out / "qc_per_sequence.csv", index=False)

    masked = aln.mask_sites(mask)
    sites = variants.site_table(masked, ref)
    calls = variants.call_snps(masked, ref)
    sites.to_csv(out / "variable_sites.csv", index=False)
    per = variants.snps_per_sample(calls, masked)
    per.to_csv(out / "snps_per_sample.csv")
    ld = variants.lineage_defining_sites(masked, ref, labels)
    ld.to_csv(out / "lineage_defining_sites.csv", index=False)
    summary.update(n_snp_calls=len(calls), median_snps_per_sample=float(per.median()),
                   n_lineages_with_defining_sites=int(ld["lineage"].nunique()) if len(ld) else 0)
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print("figure:", fig)


if __name__ == "__main__":
    main()

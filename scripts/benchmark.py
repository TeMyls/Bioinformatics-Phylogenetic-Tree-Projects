#!/usr/bin/env python3
"""Project A driver: benchmark tree-inference methods.

Simulated data (runs anywhere, ground-truth tree known):
    python scripts/benchmark.py --mode sim --sizes 50 100 200 400 --reps 10

Real Pango-labelled data (aligned + masked FASTA, metadata TSV, optional reference tree):
    python scripts/benchmark.py --mode real --alignment data/aligned.fasta \
        --metadata data/metadata.tsv --reference data/usher_subtree.nwk --sizes 100 500 1000 --reps 5
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from covidphylo import bench, qc                      # noqa: E402
from covidphylo.infer import available_methods        # noqa: E402
from covidphylo.seqio import Alignment                # noqa: E402
from covidphylo.tree import Tree                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--methods", nargs="+", default=None, help="default: everything available on this machine")
    ap.add_argument("--threads", type=int, default=1, help="threads for external tools (1 = fair timing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark"))
    # simulation
    ap.add_argument("--length", type=int, default=29_903)
    ap.add_argument("--rate", type=float, default=1e-3, help="true clock rate, subs/site/year")
    # real data
    ap.add_argument("--alignment")
    ap.add_argument("--metadata")
    ap.add_argument("--reference", help="Newick reference tree (e.g. UShER subtree); optional")
    ap.add_argument("--name-col", default="strain")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--lineage-col", default="pango_lineage")
    ap.add_argument("--mask-vcf", help="De Maio problematic-sites VCF (optional)")
    ap.add_argument("--max-ambig", type=float, default=0.05)
    ap.add_argument("--ref-name", default="NC_045512.2", help="reference row to drop if present in the alignment")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = a.methods or available_methods()
    print("methods:", methods)

    if a.mode == "sim":
        factory = bench.sim_factory(length=a.length, rate=a.rate)
    else:
        if not (a.alignment and a.metadata):
            ap.error("--mode real needs --alignment and --metadata")
        aln = Alignment.from_fasta(a.alignment)
        if a.ref_name in aln.names:                       # MAFFT --addfragments keeps the reference row
            aln = aln.subset([i for i, n in enumerate(aln.names) if n != a.ref_name])
        sites = qc.build_mask(aln.length, qc.read_problematic_sites_vcf(a.mask_vcf) if a.mask_vcf else None)
        aln = qc.drop_ambiguous(aln.mask_sites(sites), a.max_ambig)
        aln, _ = qc.deduplicate(aln)
        meta = qc.read_metadata(a.metadata, a.name_col, a.date_col, a.lineage_col)
        ref = Tree.from_newick(Path(a.reference).read_text()) if a.reference else None
        print(f"real data: {aln.n} unique sequences after masking/QC/dedup")
        factory = bench.real_factory(aln, meta, ref)

    df = bench.run_benchmark(factory, a.sizes, a.reps, methods, workdir=out / "work", seed0=a.seed,
                             threads=a.threads, collapse_len=a.length if a.mode == "sim" else 29_903)
    df.to_csv(out / "benchmark_results.csv", index=False)
    S = bench.summarise(df)
    S.to_csv(out / "benchmark_summary.csv", index=False)
    import pandas as pd
    ms = [m for m in ("nrf_collapsed", "patristic_r", "lineage_monophyly", "wall_s") if m in df]
    tests = pd.concat([bench.paired_tests(df, m) for m in ms] +
                      [bench.paired_tests(df, m, at_n=max(a.sizes)) for m in ms])
    tests.to_csv(out / "paired_tests.csv", index=False)
    figs = bench.plot_benchmark(df, out)
    print("wrote:", *[out / "benchmark_results.csv", out / "benchmark_summary.csv", *figs], sep="\n  ")


if __name__ == "__main__":
    main()

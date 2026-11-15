#!/usr/bin/env python3
"""Regenerate notebooks/*.ipynb (kept as code so they diff cleanly). Run: python scripts/make_notebooks.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


SETUP = '''
import sys, pathlib
ROOT = pathlib.Path.cwd().resolve()
if not (ROOT / "src").exists():            # allow running from notebooks/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
'''


AI_HEADER = ("code", """
# AI-use disclosure (MSBI 555 requires this at the top of the notebook or in the README)
# The code in this repository (src/covidphylo, scripts/, notebooks/) was drafted with Claude (Anthropic).
# I (the author) ran it, checked the results against simulated ground truth and the unit tests (pytest),
# and am responsible for everything submitted.
# >>> EDIT THIS NOTE so it states exactly what you did yourself and what AI helped with. <<<
""")

EDA = nb([
    AI_HEADER,
    ("markdown", """
# 00 - Exploratory analysis, QC and variant discovery (shared by Projects A and B)

Covers the presentation outline items *Dataset and its characteristics*, *Exploratory analysis* and
*Quality-control and evaluation metrics*, plus SNP calling and lineage-defining mutations.
`MODE = "sim"` runs anywhere; `MODE = "real"` needs `data/subsample/aligned.fasta` (see README section 3).
"""),
    ("code", SETUP),
    ("code", '''
MODE = "sim"
SEED = 1
OUT = ROOT / "results" / "eda_notebook"
OUT.mkdir(parents=True, exist_ok=True)
from covidphylo import eda, qc, variants
from covidphylo.seqio import Alignment

if MODE == "sim":
    from covidphylo.simulate import simulate_dataset
    sim = simulate_dataset(300, seed=SEED)
    aln, labels = sim.aln, sim.lineages
    meta = pd.DataFrame({"name": aln.names, "lineage": [labels[n] for n in aln.names],
                         "date": [f"{int(sim.tip_dates[n])}-{1 + int((sim.tip_dates[n] % 1) * 12):02d}-15" for n in aln.names]})
    ref, mask = variants.consensus(aln), qc.build_mask(aln.length)
else:
    aln = Alignment.from_fasta(ROOT / "data" / "subsample" / "aligned.fasta")
    i = aln.names.index("NC_045512.2")                      # reference row is part of the alignment
    ref = aln.X[i].copy(); aln = aln.subset([k for k in range(aln.n) if k != i])
    meta = qc.read_metadata(ROOT / "data" / "metadata.tsv")
    labels = dict(zip(meta["name"], meta["lineage"]))
    vcf = ROOT / "data" / "problematic_sites_sarsCov2.vcf"
    mask = qc.build_mask(aln.length, qc.read_problematic_sites_vcf(vcf) if vcf.exists() else None)
pd.Series(eda.eda_summary(aln, meta, mask))
'''),
    ("markdown", "## 1. Exploratory figure (lineages, dates, quality, variation along the genome, pairwise distances)"),
    ("code", '''
display(Image(str(eda.eda_panel(aln, meta, OUT, mask_sites=mask, seed=SEED))))
'''),
    ("markdown", "## 2. QC per sequence (ambiguous fraction after masking; threshold 5%)"),
    ("code", '''
q = eda.qc_table(aln, mask)
print(q["pass"].value_counts().to_dict())
q.sort_values("ambig_after_mask", ascending=False).head()
'''),
    ("markdown", "## 3. Variant discovery: SNPs versus the reference"),
    ("code", '''
masked = aln.mask_sites(mask)
sites = variants.site_table(masked, ref)
calls = variants.call_snps(masked, ref)
per = variants.snps_per_sample(calls, masked)
print(f"{len(sites)} variable sites, {len(calls)} SNP calls, median {per.median():.0f} SNPs/genome vs reference")
sites.sort_values("alt_freq", ascending=False).head(10)
'''),
    ("markdown", """
## 4. Lineage-defining mutations
Sites where a lineage (and its sub-lineages) carries a non-reference allele in >= 90% of genomes but <= 5% elsewhere.
Pango aliases are not resolved, so real-data results are conservative.
"""),
    ("code", '''
ld = variants.lineage_defining_sites(masked, ref, labels)
ld.to_csv(OUT / "lineage_defining_sites.csv", index=False)
print(ld["lineage"].nunique(), "lineages with defining sites")
ld.head(10)
'''),
])

BENCH = nb([
    AI_HEADER,
    ("markdown", """
# Project A - Benchmarking phylogenetic tree algorithms on SARS-CoV-2 (Pango-labelled) genomes

Shared core: `src/covidphylo` (identical to Project B). This notebook only adds the **benchmark harness**
(timing, size ladder, replicates, accuracy metrics, statistics).

**Question:** how do distance (NJ, UPGMA) and likelihood (FastTree, IQ-TREE, RAxML-NG) methods trade speed
against accuracy on near-identical SARS-CoV-2 genomes?

Set `MODE = "sim"` (ground-truth tree known; runs anywhere) or `MODE = "real"` (needs `data/` files from
`scripts/fetch_data.sh`).
"""),
    ("code", SETUP),
    ("markdown", "## 1. Parameters (every value is a documented choice - see README table)"),
    ("code", '''
MODE   = "sim"                    # "sim" | "real"
SIZES  = [50, 100, 200, 400]      # size ladder (tips); extend to 1000+ on real data
REPS   = 20                       # replicate datasets per size. >=8 pairs are needed for ANY Holm-significant
                                  # Wilcoxon result with 3 methods; 20 gives real power
SEED   = 0
THREADS = 1                       # 1 thread each => fair timing
OUT = ROOT / "results" / "benchmark_notebook"
OUT.mkdir(parents=True, exist_ok=True)

from covidphylo import bench
from covidphylo.infer import available_methods
METHODS = available_methods()     # internal NJ/UPGMA always; FastTree/IQ-TREE/RAxML-NG if installed
print("methods available here:", METHODS)
'''),
    ("markdown", "## 2. Build the dataset factory"),
    ("code", '''
if MODE == "sim":
    factory = bench.sim_factory(length=29_903, rate=1e-3)
else:
    from covidphylo import qc
    from covidphylo.seqio import Alignment
    from covidphylo.tree import Tree
    aln = Alignment.from_fasta(ROOT / "data" / "subsample" / "aligned.fasta")        # reference-coordinate alignment
    aln = aln.subset([i for i, n in enumerate(aln.names) if n != "NC_045512.2"])      # drop the reference row
    vcf = ROOT / "data" / "problematic_sites_sarsCov2.vcf"
    sites = qc.build_mask(aln.length, qc.read_problematic_sites_vcf(vcf) if vcf.exists() else None)
    aln = qc.drop_ambiguous(aln.mask_sites(sites), 0.05)
    aln, _ = qc.deduplicate(aln)
    meta = qc.read_metadata(ROOT / "data" / "metadata.tsv")
    ref_path = ROOT / "data" / "usher_subtree.nwk"
    ref = Tree.from_newick(ref_path.read_text()) if ref_path.exists() else None
    factory = bench.real_factory(aln, meta, ref)
'''),
    ("markdown", "## 3. Run (every method sees the same dataset within a replicate)"),
    ("code", '''
df = bench.run_benchmark(factory, SIZES, REPS, METHODS, workdir=OUT / "work", seed0=SEED,
                         threads=THREADS, verbose=False)
df.to_csv(OUT / "benchmark_results.csv", index=False)
df.head()
'''),
    ("markdown", "## 4. Summary with 95% confidence intervals\nTime and memory are geometric means with log-scale CIs (they cannot go negative); bounded metrics are clipped to their valid range."),
    ("code", '''
S = bench.summarise(df)
S.to_csv(OUT / "benchmark_summary.csv", index=False)
S[["method", "n_tips", "reps", "wall_s", "nrf_collapsed", "patristic_r", "lineage_auc", "lineage_monophyly"]].round(4)
'''),
    ("markdown", """
## 5. Figures
`nrf_collapsed` = normalised Robinson-Foulds after contracting branches with < 0.5 expected mutations in both trees;
without this, arbitrary resolution of polytomies dominates the score for SARS-CoV-2.
"""),
    ("code", '''
for f in bench.plot_benchmark(df, OUT):
    display(Image(str(f)))
'''),
    ("markdown", """## 6. Paired significance tests (Wilcoxon signed-rank, Holm-adjusted)
Pairs are the (size, replicate) datasets every method saw. The p-values are **exact** (all sign patterns enumerated), so
there is no small-sample normal-approximation warning. `min_possible_p` is the smallest p attainable with that many
non-zero pairs: if it is above 0.05 the test cannot be significant whatever the data - increase `REPS`.
Time/memory are compared on the log scale (`fold_change_a_over_b`).
"""),
    ("code", '''
metrics_tested = [m for m in ("nrf_collapsed", "patristic_r", "lineage_monophyly", "wall_s") if m in df]
tests = pd.concat([bench.paired_tests(df, m) for m in metrics_tested] +                      # pooled over all sizes
                  [bench.paired_tests(df, m, at_n=max(SIZES)) for m in metrics_tested])      # largest size only
tests.to_csv(OUT / "paired_tests.csv", index=False)
tests.round(4)
'''),
    ("markdown", """
## 7. Interpretation checklist (for the report)
* Which method is on the speed-accuracy Pareto front at the largest n?
* Does accuracy change with n (do the CIs overlap?), and do the CI widths shrink as `REPS` grows? (They shrink with
  replicates, not with n.)
* Is each test powered? Check `min_possible_p` and `n_nonzero` in the paired-test table.
* `nj_jc` vs `nj_p` give near-identical trees at SARS-CoV-2 divergence (JC correction is tiny), so treat them as one method.
* Lineage AUC saturates (~0.95-0.99) and cannot separate methods; use strict monophyly and RF instead.
* Peak memory is dominated by the shared distance matrix and `tracemalloc` sees only Python allocations, so internal
  methods are indistinguishable; compare external tools (FastTree/IQ-TREE/RAxML-NG) only with their own memory source.
* Limitations: the simulations contain no recombination (real lineages such as XBB are recombinants) and use a simple
  substitution model; heterochronous sampling violates the strict clock UPGMA assumes; Pango lineages are *derived from*
  trees, so lineage recovery is a proxy, not ground truth.
"""),
])

RECON = nb([
    AI_HEADER,
    ("markdown", """
# Project B - Reconstructing (and dating) SARS-CoV-2 phylogenetic trees

Shares `src/covidphylo` with Project A. This notebook adds the **interpretive layer**: best-fit rooting,
root-to-tip temporal signal, clock rate, node dating and lineage annotation.

Set `MODE = "sim"` (known rate/tMRCA - validates the code) or `"real"` (needs `data/` from `scripts/fetch_data.sh`).
"""),
    ("code", SETUP),
    ("code", '''
MODE   = "sim"
METHOD = "nj_jc"                  # or "fasttree" / "iqtree" / "raxmlng" when installed
N_TIPS = 300                      # sim size; real: --max-tips in scripts/reconstruct.py
SEED   = 1
OUT = ROOT / "results" / "reconstruct_notebook"
OUT.mkdir(parents=True, exist_ok=True)

from covidphylo import clock, qc, metrics
from covidphylo.infer import available_methods
from covidphylo.reconstruct import run_reconstruction
print("methods available:", available_methods())
'''),
    ("markdown", "## 1. Load data"),
    ("code", '''
if MODE == "sim":
    from covidphylo.simulate import simulate_dataset
    sim = simulate_dataset(N_TIPS, rate=1e-3, seed=SEED)
    aln, dates, lineages, sites = sim.aln, sim.tip_dates, sim.lineages, None
    print(f"true rate 1e-3 subs/site/yr, true tMRCA {sim.root_date:.3f}")
else:
    from covidphylo.seqio import Alignment
    meta = qc.read_metadata(ROOT / "data" / "metadata.tsv")
    meta["dec"] = meta["date"].map(clock.date_to_decimal_year)
    meta = meta[np.isfinite(meta["dec"])]
    sub = qc.subsample_stratified(meta, 1000, seed=SEED)
    aln = Alignment.from_fasta(ROOT / "data" / "aligned.fasta", keep=set(sub["name"]))
    dates, lineages = dict(zip(sub["name"], sub["dec"])), dict(zip(sub["name"], sub["lineage"]))
    vcf = ROOT / "data" / "problematic_sites_sarsCov2.vcf"
    sites = qc.build_mask(aln.length, qc.read_problematic_sites_vcf(vcf) if vcf.exists() else None)
aln.n, aln.length
'''),
    ("markdown", "## 2. Run the pipeline (mask -> QC -> dedupe -> tree -> root -> clock -> figures)"),
    ("code", '''
log = run_reconstruction(aln, dates, lineages, OUT, method=METHOD, mask_sites=sites, seed=SEED,
                         n_perm=500, n_boot=300, verbose=False)
pd.Series({k: (np.round(v, 5) if isinstance(v, float) else v) for k, v in log.items()})
'''),
    ("markdown", """
## 3. Temporal signal
A clock-like signal needs a positive slope, decent R^2 and a permutation p-value well below 0.05.
(`p_permutation` here keeps the root fixed; say so in the methods section.)
"""),
    ("code", '''
display(Image(str(OUT / "fig_root_to_tip.png")))
'''),
    ("markdown", "## 4. Dated tree coloured by lineage"),
    ("code", '''
display(Image(str(OUT / "fig_dated_tree.png")))
'''),
    ("markdown", "## 5. Lineage summary and (sim only) comparison with the truth"),
    ("code", '''
display(pd.read_csv(OUT / "lineage_summary.csv").head(10))
if MODE == "sim":
    est = log["clock_rate"]; lo, hi = log["clock_rate_ci95"]
    print(f"rate: estimated {est:.2e} (95% CI {lo:.2e}-{hi:.2e}) vs true 1.00e-03")
    print(f"tMRCA: estimated {log['tmrca']:.2f} (95% CI {log['tmrca_ci95'][0]:.2f}-{log['tmrca_ci95'][1]:.2f}) vs true {sim.root_date:.2f}")
'''),
    ("markdown", """
## 6. Optional: compare tree-building methods on the same data
Because the core is shared with Project A, any method in `available_methods()` can be dropped in via `METHOD`.
For publication-quality dating use TreeTime/BEAST on the rooted tree written to `tree_rooted.nwk`.
"""),
])

(ROOT / "notebooks").mkdir(exist_ok=True)
for name, book in (("00_eda_and_variants.ipynb", EDA), ("01_benchmark_tree_algorithms.ipynb", BENCH), ("02_reconstruct_covid_trees.ipynb", RECON)):
    (ROOT / "notebooks" / name).write_text(json.dumps(book, indent=1))
    print("wrote", name)

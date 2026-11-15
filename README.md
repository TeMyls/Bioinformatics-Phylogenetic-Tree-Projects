# covid-phylo

> **AI-use disclosure.** The code in this repository (`src/`, `scripts/`, `notebooks/`, `tests/`) was drafted with Claude
> (Anthropic). I ran it, tested it against simulated ground truth and unit tests, and is responsible for it.
> **I came up with the idea, drafted it, and directed Claude** 

Two SARS-CoV-2 phylogenetics projects on **one shared code base**: **(A)** benchmarking tree-inference algorithms on
Pango-labelled genomes, and **(B)** reconstructing and dating SARS-CoV-2 trees. MSBI 555 Applied Computational Genomics, Fall 2026.

## Quick facts (the half-page README)

* **Run order:** `pip install -r requirements.txt` -> `pytest -q` -> `bash scripts/fetch_data.sh` -> `scripts/subsample_fasta.py` ->
  align (MAFFT) -> `scripts/eda_variants.py` -> `scripts/benchmark.py` (A) and/or `scripts/reconstruct.py` (B) -> notebooks `00`, `01`, `02`.
  Cluster version: `sbatch scripts/slurm_pipeline.sh`.
* **Data provenance:** Nextstrain open SARS-CoV-2 data (the open build derived from public GenBank records; metadata columns `strain`, `date`, `pango_lineage` - verify the column names in your download),
  reference NC_045512.2 (Wuhan-Hu-1, NCBI), problematic-sites VCF (De Maio et al., GitHub `W-L/ProblematicSites_SARS-CoV-2`).
  The exact genomes used are listed in `data/subsample/subsample_provenance.tsv` (written by `subsample_fasta.py`) - **put the download date here: ____**.
  Raw sequences are not committed.
* **Software versions:** see `requirements.txt` (Python 3.13; numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2).
  External tools (record `--version` output here before submitting): MAFFT ____, FastTree ____, IQ-TREE ____, RAxML-NG ____.
* **Seeds:** `--seed` on every script (benchmark default 0, others 1); each replicate's seed is saved in `benchmark_results.csv`.
* **Render the notebooks for submission** (the course wants the rendered HTML/PDF too):
  `jupyter nbconvert --to html --execute notebooks/*.ipynb --output-dir results/rendered`

## 1. Install and self-test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                        # 15 tests, a few seconds
```
Optional external programs are auto-detected on `PATH`; without them everything still runs with the built-in NJ/UPGMA.

## 2. Quick start with no downloads (simulated data with a known true tree)

```bash
python scripts/eda_variants.py --mode sim --n 400
python scripts/benchmark.py    --mode sim --sizes 50 100 200 400 --reps 20
python scripts/reconstruct.py  --mode sim --n 300 --method nj_jc
```
Simulation validates the code (the clock fit recovers the true rate). **The course requires a real genomic dataset - the
manuscript and slides must use real data (section 3), with simulation only as validation.**

## 3. Real data

```bash
bash scripts/fetch_data.sh
python scripts/subsample_fasta.py --metadata data/metadata.tsv --sequences data/sequences.fasta \
       --reference data/NC_045512.2.fasta --n 1000 --seed 1 --out-dir data/subsample
mafft --thread 8 --6merpair --keeplength --addfragments data/subsample/subsample.fasta data/NC_045512.2.fasta > data/subsample/aligned.fasta
python scripts/eda_variants.py --mode real --alignment data/subsample/aligned.fasta --metadata data/metadata.tsv \
       --reference data/NC_045512.2.fasta --mask-vcf data/problematic_sites_sarsCov2.vcf
python scripts/benchmark.py   --mode real --alignment data/subsample/aligned.fasta --metadata data/metadata.tsv \
       --mask-vcf data/problematic_sites_sarsCov2.vcf --sizes 100 250 500 1000 --reps 10      # + --reference usher_subtree.nwk for RF accuracy
python scripts/reconstruct.py --mode real --alignment data/subsample/aligned.fasta --metadata data/metadata.tsv \
       --mask-vcf data/problematic_sites_sarsCov2.vcf --method iqtree --max-tips 1000
```
* MAFFT `--addfragments` keeps the reference as row 1 of `aligned.fasta`; `eda_variants.py` uses it as the reference and the
  benchmark/reconstruct drivers ignore it (`--ref-name NC_045512.2`).
* `fetch_data.sh` URLs were written without being able to test downloads - check them. Every dataset needs instructor approval.
* Optional reference tree for RF accuracy: prune the public UShER tree to your tips (`matUtils extract`).

## 4. What is shared and what diverges

| Component | File(s) | A: benchmark | B: reconstruct |
|---|---|---|---|
| FASTA I/O, alignment container | `seqio.py` | yes | yes |
| QC, masking, dedup, lineage subsampling | `qc.py` | yes | yes |
| EDA figures + QC tables | `eda.py` | yes | yes |
| SNP calling, lineage-defining mutations | `variants.py` | yes | yes |
| Distances (p, JC69; ambiguity-aware) | `distances.py` | yes | yes |
| Tree structure, Newick, NJ, UPGMA | `tree.py` | yes | yes |
| Inference back-ends | `infer.py` | yes | yes |
| Tree/lineage metrics | `metrics.py` | **core** | validation |
| Outbreak simulator (known truth) | `simulate.py` | **core** | validation |
| Timing / replicates / paired tests | `bench.py` | **A only** | - |
| Root search, root-to-tip, permutation, dating | `clock.py`, `reconstruct.py` | - | **B only** |

## 5. Layout

```
README.md  requirements.txt
data/          download instructions; subsample/ holds the provenance table (no raw data committed)
src/covidphylo/ analysis library (commented modules)
scripts/       fetch_data.sh  subsample_fasta.py  eda_variants.py  benchmark.py  reconstruct.py  slurm_pipeline.sh  make_notebooks.py
notebooks/     00_eda_and_variants  01_benchmark_tree_algorithms  02_reconstruct_covid_trees (.ipynb; edit via make_notebooks.py)
tests/         test_core.py
results/       everything the manuscript cites (created by the scripts)
report/        manuscript_outline.md, presentation_outline.md
```

## 6. Parameters and why (fill the justification from your own reading before you submit)

| Parameter | Default | Where | Rationale |
|---|---|---|---|
| genome length / ambiguity (raw) | 29,000-30,500 nt / <=1% | `qc_raw_sequences` | drop truncated and N-rich records (my choice) |
| ambiguity after masking | <=5% | `drop_ambiguous` | N-runs distort distances (my choice) |
| terminal mask | first 55, last 100 sites | `build_mask` | low-coverage ends; confirm against the VCF |
| mask VCF filter | `mask` | `read_problematic_sites_vcf` | recurrent-error sites; `caution` is stricter |
| deduplication | on | `deduplicate` | identical sequences give zero-length branches |
| distance model | JC69 and p | `distances` | nearly identical at ~1e-3 divergence |
| ML models | GTR (FastTree, IQ-TREE `-fast`), GTR+G (RAxML-NG) | `infer.py` | common defaults |
| threads | 1 | `--threads` | fair timing |
| RF collapse threshold | 0.5 / genome length | `bench.py` | <0.5 expected substitutions is unresolvable |
| replicates | 20 (>=8 needed for any Holm-significant test) | `--reps` | tighter CIs |
| lineage-defining site | >=90% in lineage, <=5% outside, n>=5 | `variants.py` | my choice; sub-lineages count as inside |
| clock | strict, regression-based | `clock.py` | diagnostic; use TreeTime/BEAST for final dating |
| permutation / bootstrap | 1000 / 500 | `reconstruct.py` | see limitations |
| simulation | rate 1e-3/site/yr, kappa 4, Gamma 0.5, birth 2 / death 0.3 / sampling 0.4 | `simulate.py` | roughly SARS-CoV-2-like, not fitted |

## 7. Known limitations (state them in the manuscript)

* **Pango lineages are not ground truth** (derived from trees). Aliases such as `AY.4` are not resolved.
* The root-to-tip permutation keeps the root fixed; bootstrap intervals cover regression error only (in one simulated check the
  tMRCA interval narrowly missed the truth).
* Simulation uses K80+Gamma with no recombination or sequencing error. Indels are not called (gaps are missing data).
* Internal NJ memory uses `tracemalloc`; external tools use `/usr/bin/time -v`. Compare within a source only.
* **Never run here:** FastTree, IQ-TREE, RAxML-NG, MAFFT, `fetch_data.sh`, `slurm_pipeline.sh` and the real-data code paths
  (the dev sandbox had no network or installs). Run a 50-genome trial of each before a long job.

## 8. MSBI 555 notebook/code rubric (50 pts)

* **Reproducible (20):** every figure/table regenerates from the commands above; results match the manuscript.
* **Clean and commented (15):** commented modules, one notebook per stage, no dead code.
* **README (15):** provenance, versions and run order are in "Quick facts" - **fill the blanks (download date, tool versions).**

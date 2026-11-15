#!/usr/bin/env bash
# Example Slurm job: real-data pipeline  fetch -> subsample -> align -> EDA/variants -> benchmark + reconstruct.
# EDIT the #SBATCH lines and module names for your cluster, then:  sbatch scripts/slurm_pipeline.sh
# (Not tested on your cluster -- the commands are the same ones the README lists, so you can also run them by hand.)
#SBATCH --job-name=covid-phylo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

N=${N:-1000}           # genomes to analyse
SEED=${SEED:-1}
# module load mafft iqtree fasttree python    # <- cluster-specific; uncomment/adjust

bash scripts/fetch_data.sh                                   # skips files that already exist
python scripts/subsample_fasta.py --metadata data/metadata.tsv --sequences data/sequences.fasta \
       --reference data/NC_045512.2.fasta --n "$N" --seed "$SEED" --out-dir data/subsample

# align to the reference, keeping reference coordinates (the reference stays in the file as row 1)
mafft --thread "$SLURM_CPUS_PER_TASK" --6merpair --keeplength \
      --addfragments data/subsample/subsample.fasta data/NC_045512.2.fasta > data/subsample/aligned.fasta

python scripts/eda_variants.py --mode real --alignment data/subsample/aligned.fasta --metadata data/metadata.tsv \
       --reference data/NC_045512.2.fasta --mask-vcf data/problematic_sites_sarsCov2.vcf --seed "$SEED"

python scripts/benchmark.py --mode real --alignment data/subsample/aligned.fasta --metadata data/metadata.tsv \
       --mask-vcf data/problematic_sites_sarsCov2.vcf --sizes 100 250 500 "$N" --reps 10 --threads 1

python scripts/reconstruct.py --mode real --alignment data/subsample/aligned.fasta --metadata data/metadata.tsv \
       --mask-vcf data/problematic_sites_sarsCov2.vcf --method iqtree --max-tips "$N" --seed "$SEED"

#!/usr/bin/env bash
# Download public SARS-CoV-2 inputs into data/.  Run from the repo root:  bash scripts/fetch_data.sh
# NOTE: URLs were written from memory of the public mirrors; if one 404s, check the project pages
# (Nextstrain "ncov open data", cov-lineages/pango-designation, UCSC UShER, W-L/ProblematicSites_SARS-CoV-2).
set -euo pipefail
mkdir -p data && cd data

# 1. Nextstrain open data (sequences + metadata incl. pango_lineage, date, country)
for f in metadata.tsv.zst sequences.fasta.zst; do
  [ -f "${f%.zst}" ] || { curl -fL -o "$f" "https://data.nextstrain.org/files/ncov/open/$f" && zstd -d --rm "$f"; }
done

# 2. Wuhan-Hu-1 reference (NC_045512.2) -- needed for alignment in reference coordinates
[ -f NC_045512.2.fasta ] || curl -fL -o NC_045512.2.fasta \
  "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_045512.2&rettype=fasta"

# 3. Problematic-sites mask (De Maio et al.)
[ -f problematic_sites_sarsCov2.vcf ] || curl -fL -o problematic_sites_sarsCov2.vcf \
  "https://raw.githubusercontent.com/W-L/ProblematicSites_SARS-CoV-2/master/problematic_sites_sarsCov2.vcf"

# 4. Pango lineage notes (optional, for alias resolution)
[ -f lineage_notes.txt ] || curl -fL -o lineage_notes.txt \
  "https://raw.githubusercontent.com/cov-lineages/pango-designation/master/lineage_notes.txt"

echo "Done. Next: align to reference, e.g."
echo "  mafft --thread 8 --6merpair --keeplength --addfragments sequences.fasta NC_045512.2.fasta > aligned_with_ref.fasta"
echo "(or: nextalign run --input-ref NC_045512.2.fasta --output-fasta aligned.fasta sequences.fasta)"
echo "For the optional reference tree, take the public UShER tree and prune it to your tips"
echo "(e.g. matUtils extract -i public-latest.all.masked.pb.gz -s tips.txt -t subtree.nwk)."

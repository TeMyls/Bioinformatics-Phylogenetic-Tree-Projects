#!/usr/bin/env python3
"""Compile NCBI Virus / nuccore links for SARS-CoV-2 genomes of each variant.

Run on a machine with internet access and the NCBI `datasets` CLI installed
(https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/).
UNTESTED: written without network access, so check the CLI flags against
`datasets summary virus genome --help` for your installed version.

Usage:
    python scripts/ncbi_variant_links.py --per-variant 100 --out data/variant_links.tsv
"""
import argparse
import csv
import json
import subprocess

TAXON = "2697049"  # SARS-CoV-2
VARIANTS = {
    "Alpha": ["B.1.1.7"],
    "Beta": ["B.1.351"],
    "Delta": ["B.1.617.2"],
    "Omicron": ["B.1.1.529", "BA.1", "BA.2", "BA.5", "BQ.1", "XBB.1.5"],
    "Other": ["B.1", "B.1.2", "B.1.1", "B.1.177", "B.1.427", "P.1"],
}
SEARCH = "https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=SARS-CoV-2,%20taxid:2697049"


def accessions(lineage, n):
    cmd = ["datasets", "summary", "virus", "genome", "taxon", TAXON,
           "--lineage", lineage, "--complete-only", "--limit", str(n),
           "--as-json-lines"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.strip():
            yield json.loads(line)["accession"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-variant", type=int, default=100)
    ap.add_argument("--out", default="variant_links.tsv")
    a = ap.parse_args()
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["variant", "pango_lineage", "accession", "nuccore_link"])
        for variant, lineages in VARIANTS.items():
            per = max(1, a.per_variant // len(lineages))
            for lin in lineages:
                for acc in accessions(lin, per):
                    w.writerow([variant, lin, acc, f"https://www.ncbi.nlm.nih.gov/nuccore/{acc}"])
    print("wrote", a.out)
    print("NCBI Virus search page (filter Pango lineage by hand):", SEARCH)


if __name__ == "__main__":
    main()
# data/

Nothing is committed here (files are large). Run `bash scripts/fetch_data.sh` from the repo root.

| File | Source | Used for |
|---|---|---|
| `metadata.tsv`, `sequences.fasta` | Nextstrain open ncov data | genomes, sampling dates, `pango_lineage` |
| `NC_045512.2.fasta` | NCBI (Wuhan-Hu-1) | alignment reference |
| `problematic_sites_sarsCov2.vcf` | W-L/ProblematicSites_SARS-CoV-2 | site masking |
| `lineage_notes.txt` | cov-lineages/pango-designation | alias resolution (optional) |
| `usher_subtree.nwk` | UShER public tree, pruned with `matUtils extract` | reference tree for RF accuracy (optional) |

Simulated runs (`--mode sim`) need no files.

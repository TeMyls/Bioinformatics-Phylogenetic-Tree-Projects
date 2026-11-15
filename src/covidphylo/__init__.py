"""covidphylo: shared core for the two SARS-CoV-2 phylogenetics projects.

Project A  - Benchmarking phylogenetic tree algorithms on Pango-labelled genomes
Project B  - Reconstructing (and dating) SARS-CoV-2 phylogenetic trees

Both projects share everything in this package (I/O, QC, masking, distances,
tree data structure, inference wrappers, tree metrics, simulation, plotting).
Only the drivers in ``scripts/`` (benchmark.py, reconstruct.py) differ.
"""

__version__ = "0.1.0"

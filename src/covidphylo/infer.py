"""Tree-inference back-ends behind one interface.

``infer_tree(method, aln, workdir, seed, threads)`` -> InferenceResult

Internal (pure NumPy, always available): nj_jc, nj_p, upgma_jc
External (used only if the executable is on PATH): fasttree, iqtree, raxmlng
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

from .distances import distance_matrix
from .seqio import Alignment
from .tree import Tree, neighbor_joining, upgma


@dataclass
class InferenceResult:
    method: str
    tree: Tree
    wall_s: float
    peak_mb: float | None        # peak memory (see mem_source)
    mem_source: str              # 'tracemalloc' (internal) | 'gnu_time' (external) | 'none'
    cmd: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# External command runner with wall-time and (if available) peak-RSS
# --------------------------------------------------------------------------
_GNU_TIME = "/usr/bin/time"


def run_timed(cmd: list[str], cwd=None, stdout_path=None) -> tuple[float, float | None, str]:
    """Run ``cmd``; return (wall seconds, peak RSS in MB or None, stderr text)."""
    use_time = os.path.exists(_GNU_TIME)
    full = ([_GNU_TIME, "-v"] if use_time else []) + [str(c) for c in cmd]
    out = open(stdout_path, "w") if stdout_path else subprocess.DEVNULL
    t0 = time.perf_counter()
    try:
        p = subprocess.run(full, cwd=cwd, stdout=out, stderr=subprocess.PIPE, text=True)
    finally:
        if stdout_path:
            out.close()
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stderr[-2000:]}")
    peak = None
    if use_time:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
        if m:
            peak = int(m.group(1)) / 1024.0
    return wall, peak, p.stderr


def tool_available(method: str) -> bool:
    return find_executable(method) is not None


_EXES = {
    "fasttree": ["FastTree", "fasttree", "FastTreeMP"],
    "iqtree": ["iqtree2", "iqtree"],
    "raxmlng": ["raxml-ng"],
    "mafft": ["mafft"],
    "nextalign": ["nextalign"],
    "treetime": ["treetime"],
}


def find_executable(method: str) -> str | None:
    for exe in _EXES.get(method, []):
        p = shutil.which(exe)
        if p:
            return p
    return None


# --------------------------------------------------------------------------
# Method implementations
# --------------------------------------------------------------------------

def _internal(method: str, aln: Alignment) -> InferenceResult:
    model = {"nj_jc": "jc", "nj_p": "p", "upgma_jc": "jc"}[method]
    tracemalloc.start()
    t0 = time.perf_counter()
    D = distance_matrix(aln, model)
    tree = upgma(D, aln.names) if method.startswith("upgma") else neighbor_joining(D, aln.names)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return InferenceResult(method, tree, wall, peak / 2**20, "tracemalloc")


def _fasttree(aln: Alignment, workdir: Path, seed: int, threads: int, params: dict) -> InferenceResult:
    exe = find_executable("fasttree")
    fa, out = workdir / "aln.fasta", workdir / "fasttree.nwk"
    aln.to_fasta(fa)
    cmd = [exe, "-nt", "-quiet"]
    if params.get("gtr", True):
        cmd += ["-gtr"]
    if params.get("gamma", False):
        cmd += ["-gamma"]
    if params.get("fastest", False):
        cmd += ["-fastest"]
    cmd += ["-seed", str(seed), "-nosupport", str(fa)]
    env_threads = str(threads)
    os.environ["OMP_NUM_THREADS"] = env_threads     # only matters for FastTreeMP
    wall, peak, _ = run_timed(cmd, cwd=workdir, stdout_path=out)
    return InferenceResult("fasttree", Tree.from_newick(out.read_text()), wall, peak,
                           "gnu_time" if peak else "none", " ".join(cmd))


def _iqtree(aln: Alignment, workdir: Path, seed: int, threads: int, params: dict) -> InferenceResult:
    exe = find_executable("iqtree")
    fa = workdir / "aln.fasta"
    aln.to_fasta(fa)
    prefix = workdir / "iq"
    cmd = [exe, "-s", str(fa), "-m", params.get("model", "GTR"), "-seed", str(seed),
           "-T", str(threads), "--prefix", str(prefix), "-redo", "-quiet"]
    if params.get("fast", True):
        cmd.append("-fast")
    if params.get("czb", False):          # collapse zero-length branches (IQ-TREE 2)
        cmd.append("-czb")
    wall, peak, _ = run_timed(cmd, cwd=workdir)
    return InferenceResult("iqtree", Tree.from_newick((workdir / "iq.treefile").read_text()), wall, peak,
                           "gnu_time" if peak else "none", " ".join(cmd))


def _raxmlng(aln: Alignment, workdir: Path, seed: int, threads: int, params: dict) -> InferenceResult:
    exe = find_executable("raxmlng")
    fa = workdir / "aln.fasta"
    aln.to_fasta(fa)
    prefix = workdir / "rx"
    cmd = [exe, "--search1", "--msa", str(fa), "--model", params.get("model", "GTR+G"),
           "--prefix", str(prefix), "--seed", str(seed), "--threads", str(threads), "--redo"]
    wall, peak, _ = run_timed(cmd, cwd=workdir)
    return InferenceResult("raxmlng", Tree.from_newick((workdir / "rx.raxml.bestTree").read_text()), wall, peak,
                           "gnu_time" if peak else "none", " ".join(cmd))


INTERNAL_METHODS = ["nj_jc", "nj_p", "upgma_jc"]
EXTERNAL_METHODS = ["fasttree", "iqtree", "raxmlng"]
ALL_METHODS = INTERNAL_METHODS + EXTERNAL_METHODS


def available_methods() -> list[str]:
    return INTERNAL_METHODS + [m for m in EXTERNAL_METHODS if tool_available(m)]


def infer_tree(method: str, aln: Alignment, workdir=None, seed: int = 1, threads: int = 1,
               params: dict | None = None) -> InferenceResult:
    params = params or {}
    if method in INTERNAL_METHODS:
        return _internal(method, aln)
    if not tool_available(method):
        raise RuntimeError(f"{method}: executable not found on PATH (see README for install).")
    workdir = Path(workdir or ".")
    workdir.mkdir(parents=True, exist_ok=True)
    fn = {"fasttree": _fasttree, "iqtree": _iqtree, "raxmlng": _raxmlng}[method]
    return fn(aln, workdir, seed, threads, params)


# --------------------------------------------------------------------------
# Alignment helper (MAFFT / nextalign wrappers) -- used on real data only
# --------------------------------------------------------------------------

def align_to_reference(seqs_fasta, reference_fasta, out_fasta, tool: str = "mafft", threads: int = 4):
    """Align many genomes to a reference, keeping reference coordinates.

    mafft:      mafft --6merpair --keeplength --addfragments seqs ref > out  (reference row is removed)
    nextalign:  nextalign --reference ref --output-fasta out seqs
    """
    exe = find_executable(tool)
    if exe is None:
        raise RuntimeError(f"{tool} not found on PATH")
    if tool == "mafft":
        run_timed([exe, "--thread", str(threads), "--6merpair", "--keeplength",
                   "--addfragments", str(seqs_fasta), str(reference_fasta)], stdout_path=out_fasta)
    elif tool == "nextalign":
        run_timed([exe, "run", "--input-ref", str(reference_fasta), "--output-fasta", str(out_fasta),
                   "--jobs", str(threads), str(seqs_fasta)])
    else:
        raise ValueError(tool)
    return out_fasta

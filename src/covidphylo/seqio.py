"""FASTA I/O and a compact numeric alignment container (pure Python + NumPy).

Encoding: A=0, C=1, G=2, T=3, everything else (N, gaps, IUPAC ambiguity) = 4.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# 256-entry lookup table: ASCII byte -> state code
_CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _i
    _CODE[ord(_b.lower())] = _i
_DECODE = np.frombuffer(b"ACGTN", dtype=np.uint8)

AMBIG = 4


def _open(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    if path.endswith(".zst"):
        raise ValueError(
            "Python cannot read .zst directly; run `zstd -d file.zst` first "
            "(scripts/fetch_data.sh does this for you)."
        )
    return open(path, "r")


def iter_fasta(path) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence); name is the first whitespace-delimited header token."""
    name, chunks = None, []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line[0] == ">":
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0] if line[1:].strip() else "", []
            else:
                chunks.append(line.strip())
        if name is not None:
            yield name, "".join(chunks)


def read_fasta(path, keep: set[str] | None = None) -> dict[str, str]:
    """Read a FASTA file into an ordered dict; optionally keep only names in ``keep``."""
    out: dict[str, str] = {}
    for n, s in iter_fasta(path):
        if keep is None or n in keep:
            out[n] = s
    return out


def write_fasta(records: Iterable[tuple[str, str]] | dict[str, str], path, width: int = 0):
    items = records.items() if isinstance(records, dict) else records
    with open(path, "w") as fh:
        for n, s in items:
            fh.write(f">{n}\n")
            if width:
                for i in range(0, len(s), width):
                    fh.write(s[i : i + width] + "\n")
            else:
                fh.write(s + "\n")


def encode(seq: str) -> np.ndarray:
    return _CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def decode(arr: np.ndarray) -> str:
    return _DECODE[arr].tobytes().decode("ascii")


class Alignment:
    """Names + (n_seq x length) uint8 matrix."""

    def __init__(self, names: list[str], X: np.ndarray):
        assert X.ndim == 2 and len(names) == X.shape[0]
        self.names = list(names)
        self.X = X

    # -- construction ------------------------------------------------------
    @classmethod
    def from_fasta(cls, path, keep: set[str] | None = None) -> "Alignment":
        names, rows = [], []
        for n, s in iter_fasta(path):
            if keep is not None and n not in keep:
                continue
            names.append(n)
            rows.append(encode(s))
        if not rows:
            raise ValueError(f"No sequences read from {path}")
        L = {len(r) for r in rows}
        if len(L) != 1:
            raise ValueError(f"Sequences are not aligned (lengths {sorted(L)[:5]}...)")
        return cls(names, np.vstack(rows))

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Alignment":
        return cls(list(d), np.vstack([encode(s) for s in d.values()]))

    def to_fasta(self, path, width: int = 0):
        write_fasta(((n, decode(r)) for n, r in zip(self.names, self.X)), path, width)

    # -- basic ops ---------------------------------------------------------
    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def length(self) -> int:
        return self.X.shape[1]

    def subset(self, idx) -> "Alignment":
        idx = np.asarray(idx)
        return Alignment([self.names[i] for i in idx], self.X[idx])

    def subset_names(self, names: Iterable[str]) -> "Alignment":
        pos = {n: i for i, n in enumerate(self.names)}
        return self.subset([pos[n] for n in names])

    def mask_sites(self, sites0: Iterable[int]) -> "Alignment":
        """Return a copy with the given 0-based columns set to ambiguous (N)."""
        X = self.X.copy()
        sites0 = np.asarray(sorted(set(int(s) for s in sites0)), dtype=int)
        sites0 = sites0[(sites0 >= 0) & (sites0 < X.shape[1])]
        X[:, sites0] = AMBIG
        return Alignment(self.names, X)

    def ambiguous_fraction(self) -> np.ndarray:
        return (self.X == AMBIG).mean(axis=1)

    def variable_columns(self) -> np.ndarray:
        """Indices of columns with >=2 distinct unambiguous states."""
        present = np.stack([(self.X == b).any(axis=0) for b in range(4)])
        return np.flatnonzero(present.sum(axis=0) >= 2)

    def variable_only(self) -> "Alignment":
        return Alignment(self.names, self.X[:, self.variable_columns()])

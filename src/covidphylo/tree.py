"""Phylogenetic tree structure (unrooted graph + optional stored root), Newick I/O,
neighbour-joining and UPGMA.  Pure Python + NumPy, iterative (no recursion limits).

Design: a Tree is an *unrooted* weighted graph.  A root, if known, is stored as
``root_edge = (a, b, x)``: the root sits on edge a-b at distance x from a.
Everything that needs a rooted view goes through :class:`RootedView`.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np


class RootedView:
    """Rooted traversal of a Tree (virtual root id = -1 when rooted on an edge)."""

    def __init__(self, tree: "Tree", root_edge=None):
        adj, names = tree.adj, tree.name
        parent: dict[int, int | None] = {}
        plen: dict[int, float] = {}
        children: dict[int, list[int]] = {}
        if root_edge is not None:
            a, b, x = root_edge
            l = adj[a][b]
            root = -1
            children[-1] = [a, b]
            parent[-1] = None
            parent[a] = parent[b] = -1
            plen[a], plen[b] = x, l - x
            visited = {a, b}
            stack = [b, a]
        else:
            internal = [i for i in adj if i not in names]
            root = internal[0] if internal else next(iter(adj))
            parent[root] = None
            visited = {root}
            stack = [root]
        order = [root] if root_edge is not None else []
        while stack:
            v = stack.pop()
            order.append(v)
            kids = []
            for w, l in adj[v].items():
                if w in visited:
                    continue
                visited.add(w)
                parent[w] = v
                plen[w] = l
                kids.append(w)
            children[v] = kids
            stack.extend(reversed(kids))
        # leaf bookkeeping
        lo: dict[int, int] = {}
        c = 0
        leaves: list[int] = []
        for v in order:
            lo[v] = c
            if v in names:
                leaves.append(v)
                c += 1
        nleaf = {v: 0 for v in order}
        for v in reversed(order):
            if v in names:
                nleaf[v] = 1
            p = parent[v]
            if p is not None:
                nleaf[p] += nleaf[v]
        depth = {root: 0.0}
        for v in order:
            if v == root:
                continue
            depth[v] = depth[parent[v]] + plen[v]
        self.tree, self.root = tree, root
        self.parent, self.plen, self.children = parent, plen, children
        self.order, self.leaves = order, leaves
        self.lo = lo
        self.hi = {v: lo[v] + nleaf[v] for v in order}
        self.depth = depth
        self.leaf_names = [names[v] for v in leaves]

    # distances from every node to every leaf (rows keyed by node id) ------------
    def node_leaf_distances(self) -> dict[int, np.ndarray]:
        """Full dict node -> vector of distances to all leaves (leaf order = self.leaves).

        Memory O(nodes x leaves); fine up to a few thousand tips.
        """
        n = len(self.leaves)
        rows: dict[int, np.ndarray] = {}
        rows[self.root] = np.array([self.depth[l] for l in self.leaves])
        for v in self.order:
            if v == self.root:
                continue
            p, l = self.parent[v], self.plen[v]
            r = rows[p] + l
            r[self.lo[v]: self.hi[v]] -= 2 * l
            rows[v] = r
        return rows

    def leaf_distance_matrix(self) -> np.ndarray:
        """Patristic distances between leaves, in self.leaves order."""
        rows = self.node_leaf_distances()
        return np.vstack([rows[l] for l in self.leaves])


class Tree:
    def __init__(self):
        self.adj: dict[int, dict[int, float]] = {}
        self.name: dict[int, str] = {}
        self.root_edge: tuple[int, int, float] | None = None
        self._n = 0

    # -- construction -------------------------------------------------------
    def new_node(self, name: str | None = None) -> int:
        i = self._n
        self._n += 1
        self.adj[i] = {}
        if name is not None:
            self.name[i] = name
        return i

    def add_edge(self, u: int, v: int, length: float):
        self.adj[u][v] = float(length)
        self.adj[v][u] = float(length)

    def copy(self) -> "Tree":
        t = Tree()
        t.adj = {k: dict(v) for k, v in self.adj.items()}
        t.name = dict(self.name)
        t.root_edge = self.root_edge
        t._n = self._n
        return t

    # -- info ----------------------------------------------------------------
    @property
    def leaf_ids(self) -> list[int]:
        return [i for i in self.adj if i in self.name]

    @property
    def leaf_names(self) -> list[str]:
        return sorted(self.name.values())

    @property
    def n_leaves(self) -> int:
        return len(self.name)

    def total_length(self) -> float:
        return sum(l for u, nb in self.adj.items() for v, l in nb.items() if u < v)

    def rooted(self, root_edge="stored") -> RootedView:
        if isinstance(root_edge, str) and root_edge == "stored":
            root_edge = self.root_edge
        return RootedView(self, root_edge)

    # -- rooting helpers -------------------------------------------------------
    def _suppress_root_node(self, root: int):
        """If ``root`` has exactly two neighbours, remove it and remember the root edge."""
        nb = list(self.adj[root].items())
        if len(nb) != 2:
            return
        (a, la), (b, lb) = nb
        del self.adj[a][root], self.adj[b][root], self.adj[root]
        self.add_edge(a, b, la + lb)
        self.root_edge = (a, b, la)

    def with_root(self, a: int, b: int, x: float) -> "Tree":
        t = self.copy()
        t.root_edge = (a, b, float(x))
        return t

    # -- Newick ------------------------------------------------------------------
    @classmethod
    def from_newick(cls, text: str) -> "Tree":
        s = re.sub(r"\[[^\]]*\]", "", text.strip())
        if s.endswith(";"):
            s = s[:-1]
        parent: list[int] = []
        length: list[float] = []
        label: list[str] = []
        nkids: list[int] = []

        def new(p):
            parent.append(p)
            length.append(0.0)
            label.append("")
            nkids.append(0)
            if p >= 0:
                nkids[p] += 1
            return len(parent) - 1

        i, n, stack = 0, len(s), []

        def read_label():
            nonlocal i
            if i < n and s[i] == "'":
                j = s.index("'", i + 1)
                lab, i = s[i + 1: j], j + 1
            else:
                j = i
                while j < n and s[j] not in ",():;":
                    j += 1
                lab, i = s[i:j].strip(), j
            return lab

        def read_len():
            nonlocal i
            if i < n and s[i] == ":":
                j = i + 1
                while j < n and s[j] not in ",():;":
                    j += 1
                v = float(s[i + 1: j])
                i = j
                return v
            return 0.0

        while i < n:
            c = s[i]
            if c == "(":
                stack.append(new(stack[-1] if stack else -1))
                i += 1
            elif c in ", \n\t\r":
                i += 1
            elif c == ")":
                nid = stack.pop()
                i += 1
                label[nid] = read_label()
                length[nid] = read_len()
            else:
                lab = read_label()
                nid = new(stack[-1] if stack else -1)
                label[nid] = lab
                length[nid] = read_len()
        t = cls()
        for k in range(len(parent)):
            t.new_node(label[k] if nkids[k] == 0 else None)
            if nkids[k] == 0 and not label[k]:
                t.name[k] = f"tip{k}"
        root = parent.index(-1)
        for k, p in enumerate(parent):
            if p >= 0:
                t.add_edge(p, k, length[k])
        t._suppress_root_node(root)
        if root in t.adj and len(t.adj[root]) == 1:  # unary root: drop it
            (c, _), = t.adj[root].items()
            del t.adj[c][root], t.adj[root]
        return t

    def to_newick(self, root_edge="stored", digits: int = 8) -> str:
        v = self.rooted(root_edge)
        fmt = f"{{:.{digits}f}}".format

        def lab(nm: str) -> str:
            return f"'{nm}'" if re.search(r"[\s,:;()\[\]']", nm) else nm

        s: dict[int, str] = {}
        for node in reversed(v.order):
            if node in self.name:
                body = lab(self.name[node])
            else:
                body = "(" + ",".join(s.pop(c) for c in v.children[node]) + ")"
            if node == v.root:
                return body + ";"
            s[node] = f"{body}:{fmt(v.plen[node])}"
        raise RuntimeError("unreachable")

    # -- topology ops --------------------------------------------------------------
    def pruned(self, keep: Iterable[str]) -> "Tree":
        """Subtree induced by the leaves named in ``keep`` (degree-2 nodes suppressed)."""
        t = self.copy()
        t.root_edge = None
        keep = set(keep)
        stack = [i for i in t.leaf_ids if t.name[i] not in keep]
        while stack:
            i = stack.pop()
            if i not in t.adj:
                continue
            nbrs = list(t.adj[i])
            for m in nbrs:
                del t.adj[m][i]
            del t.adj[i]
            t.name.pop(i, None)
            for m in nbrs:
                if m in t.adj and m not in t.name and len(t.adj[m]) <= 1:
                    stack.append(m)
        for i in list(t.adj):
            if i in t.adj and i not in t.name and len(t.adj[i]) == 2:
                (a, la), (b, lb) = t.adj[i].items()
                del t.adj[a][i], t.adj[b][i], t.adj[i]
                t.add_edge(a, b, la + lb)
        return t

    def collapse_short_branches(self, threshold: float) -> "Tree":
        """Contract internal edges with length <= threshold (creates polytomies)."""
        t = self.copy()
        t.root_edge = None
        edges = [(u, v) for u, nb in t.adj.items() for v in nb if u < v
                 and u not in t.name and v not in t.name and t.adj[u][v] <= threshold]
        for u, v in edges:
            if v not in t.adj or u not in t.adj or v not in t.adj[u]:
                continue
            for w, l in t.adj[v].items():
                if w == u:
                    continue
                del t.adj[w][v]
                t.adj[u][w] = l
                t.adj[w][u] = l
            del t.adj[u][v]
            del t.adj[v]
        return t

    def split_masks(self, index: dict[str, int]) -> set[int]:
        """Non-trivial bipartitions as normalised big-int bitmasks (side without leaf index 0)."""
        n = len(index)
        full = (1 << n) - 1
        v = self.rooted()
        mask: dict[int, int] = {}
        for node in reversed(v.order):
            if node in self.name:
                mask[node] = 1 << index[self.name[node]]
            else:
                m = 0
                for c in v.children[node]:
                    m |= mask[c]
                mask[node] = m
        out = set()
        for node in v.order:
            if node == v.root:
                continue
            m = mask[node]
            if m & 1:
                m ^= full
            k = m.bit_count()
            if 2 <= k <= n - 2:
                out.add(m)
        return out

    def patristic(self):
        """(sorted leaf names, patristic distance matrix in that order)."""
        v = self.rooted()
        D = v.leaf_distance_matrix()
        pos = {nm: i for i, nm in enumerate(v.leaf_names)}
        order = [pos[nm] for nm in sorted(v.leaf_names)]
        return sorted(v.leaf_names), D[np.ix_(order, order)]


# ====================================================================================
# Distance-based inference (internal implementations)
# ====================================================================================

def neighbor_joining(D: np.ndarray, names: list[str]) -> Tree:
    """Saitou & Nei (1987) neighbour joining; O(n^3) time, vectorised, O(n^2) memory."""
    n = len(names)
    t = Tree()
    if n < 2:
        raise ValueError("need >=2 taxa")
    ids = [t.new_node(nm) for nm in names]
    if n == 2:
        t.add_edge(ids[0], ids[1], max(D[0, 1], 0.0))
        return t
    D = np.array(D, dtype=np.float64, copy=True)
    active = np.ones(n, dtype=bool)
    node_of = list(ids)
    m = n
    while m > 3:
        act = np.flatnonzero(active)
        sub = D[np.ix_(act, act)]
        r = sub.sum(axis=1)
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        k = int(np.argmin(Q))
        a, b = divmod(k, m)
        i, j = act[a], act[b]
        dij = D[i, j]
        li = 0.5 * dij + (r[a] - r[b]) / (2 * (m - 2))
        lj = dij - li
        u = t.new_node()
        t.add_edge(node_of[i], u, max(li, 0.0))
        t.add_edge(node_of[j], u, max(lj, 0.0))
        # distances from new node u to the rest; reuse slot i for u
        du = 0.5 * (D[i, :] + D[j, :] - dij)
        D[i, :] = du
        D[:, i] = du
        D[i, i] = 0.0
        active[j] = False
        node_of[i] = u
        m -= 1
    act = np.flatnonzero(active)
    i, j, k = act
    dij, dik, djk = D[i, j], D[i, k], D[j, k]
    u = t.new_node()
    t.add_edge(node_of[i], u, max(0.5 * (dij + dik - djk), 0.0))
    t.add_edge(node_of[j], u, max(0.5 * (dij + djk - dik), 0.0))
    t.add_edge(node_of[k], u, max(0.5 * (dik + djk - dij), 0.0))
    return t


def upgma(D: np.ndarray, names: list[str]) -> Tree:
    """UPGMA (average linkage, ultrametric). Returns a tree rooted at the final join."""
    n = len(names)
    t = Tree()
    ids = [t.new_node(nm) for nm in names]
    if n == 2:
        t.add_edge(ids[0], ids[1], D[0, 1])
        return t
    D = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(D, np.inf)
    size = np.ones(n)
    height = np.zeros(n)
    node_of = list(ids)
    active = np.ones(n, dtype=bool)
    last = None
    for _ in range(n - 1):
        act = np.flatnonzero(active)
        sub = D[np.ix_(act, act)]
        k = int(np.argmin(sub))
        a, b = divmod(k, len(act))
        i, j = act[a], act[b]
        h = D[i, j] / 2.0
        u = t.new_node()
        t.add_edge(node_of[i], u, max(h - height[i], 0.0))
        t.add_edge(node_of[j], u, max(h - height[j], 0.0))
        new = (size[i] * D[i, :] + size[j] * D[j, :]) / (size[i] + size[j])
        D[i, :] = new
        D[:, i] = new
        D[i, i] = np.inf
        active[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        size[i] += size[j]
        height[i] = h
        node_of[i] = u
        last = u
    t._suppress_root_node(last)
    return t

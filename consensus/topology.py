"""Communication graphs for the gossip consensus step (Sec. 5.2, Fig. 5).

    ring      i -- i+1 -- ... -- N -- 1, each node talks to its two neighbours
              (the topology used for gossip in Fig. 5)
    chain     a path 1 -- 2 -- ... -- N with no edge closing the loop
    complete  every pair of nodes is connected
    expander  random 3-regular graph
    grid      sqrt(N) x sqrt(N) lattice

Ring all-reduce computes the exact average regardless of the graph; the graph
only matters for gossip, whose mixing rate is governed by the spectral gap of
the Metropolis weight matrix.
"""
import math
from typing import Dict, List, Optional

TOPOLOGIES = ("ring", "complete", "expander", "grid", "chain")


def build_topology(n_agents: int, kind: str = "ring", seed: int = 0) -> List[List[int]]:
    """Undirected adjacency list (neighbours exclude self)."""
    if n_agents <= 1:
        return [[]]
    if kind == "chain":
        return [[j for j in (i - 1, i + 1) if 0 <= j < n_agents] for i in range(n_agents)]
    if kind == "ring":
        if n_agents == 2:
            return [[1], [0]]
        return [[(i - 1) % n_agents, (i + 1) % n_agents] for i in range(n_agents)]
    if kind == "complete":
        return [[j for j in range(n_agents) if j != i] for i in range(n_agents)]
    if kind == "grid":
        side = int(math.isqrt(n_agents))
        if side * side != n_agents:
            raise ValueError(f"grid topology needs a perfect square N, got {n_agents}")
        adj: List[List[int]] = [[] for _ in range(n_agents)]
        for r in range(side):
            for c in range(side):
                i = r * side + c
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < side and 0 <= cc < side:
                        adj[i].append(rr * side + cc)
        return adj
    if kind == "expander":
        return _random_regular(n_agents, degree=3, seed=seed)
    raise ValueError(f"unknown topology {kind!r}; expected one of {TOPOLOGIES}")


def _random_regular(n: int, degree: int = 3, seed: int = 0) -> List[List[int]]:
    """Random d-regular graph via the pairing model, retried until simple and connected.

    Falls back to a ring plus chords when no d-regular graph exists.
    """
    import random

    if n <= degree or (n * degree) % 2 != 0:
        adj = build_topology(n, "ring")
        for i in range(n):
            j = (i + n // 2) % n
            if j != i and j not in adj[i]:
                adj[i].append(j)
                adj[j].append(i)
        return [sorted(set(a)) for a in adj]

    rng = random.Random(seed)
    for _ in range(200):
        stubs = [i for i in range(n) for _ in range(degree)]
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for a, b in zip(stubs[0::2], stubs[1::2]):
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if not ok:
            continue
        adj: List[List[int]] = [[] for _ in range(n)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        if _connected(adj):
            return [sorted(a) for a in adj]

    return _random_regular(n, degree, seed + 1) if seed < 5 else build_topology(n, "ring")


def _connected(adj: List[List[int]]) -> bool:
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(adj)


def metropolis_weights(adj: List[List[int]]) -> List[Dict[int, float]]:
    """Metropolis-Hastings mixing weights (symmetric, doubly stochastic).

        W_ij = 1 / (1 + max(d_i, d_j))        for j in N(i)
        W_ii = 1 - sum_{j in N(i)} W_ij
    """
    deg = [len(a) for a in adj]
    weights: List[Dict[int, float]] = []
    for i, nbrs in enumerate(adj):
        row: Dict[int, float] = {}
        acc = 0.0
        for j in nbrs:
            w = 1.0 / (1.0 + max(deg[i], deg[j]))
            row[j] = w
            acc += w
        row[i] = 1.0 - acc
        weights.append(row)
    return weights


def spectral_gap(adj: List[List[int]]) -> Optional[float]:
    """1 - |lambda_2(W)| of the Metropolis weight matrix (larger = faster mixing)."""
    try:
        import numpy as np
    except ImportError:
        return None
    n = len(adj)
    W = np.zeros((n, n), dtype=np.float64)
    for i, row in enumerate(metropolis_weights(adj)):
        for j, w in row.items():
            W[i, j] = w
    eig = np.linalg.eigvals(W)
    mags = sorted((abs(e) for e in eig), reverse=True)
    return float(1.0 - mags[1]) if len(mags) > 1 else 1.0


def n_edges(adj: List[List[int]]) -> int:
    return sum(len(a) for a in adj) // 2

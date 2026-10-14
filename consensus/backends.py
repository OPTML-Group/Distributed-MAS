"""How the consensus average of Eq. (10) is computed across nodes.

    CentralizedCoordinator  gather -> average -> broadcast at one node. Exact.
                            Used for the centralized baseline (one pooled node)
                            and as an exact-mean reference.
    RingAllReduce           reduce-scatter + all-gather on a directed ring,
                            2(N-1) steps, ~2S floats per node. Exact (up to
                            floating-point summation order). The default for
                            ADMM (Sec. 5.1).
    Gossip                  tau rounds of x <- W x with Metropolis weights on a
                            communication graph. Approximate: nodes end up with
                            different values (Fig. 5).

Every backend maps N per-node states (each a flat list of tensors) to N
per-node averaged states.
"""
from typing import List, Optional, Sequence

import torch

from .topology import build_topology, metropolis_weights, n_edges, spectral_gap

State = List[torch.Tensor]


def _zeros_like(state: State) -> State:
    return [torch.zeros_like(t) for t in state]


def _exact_mean(states: Sequence[State]) -> State:
    acc = _zeros_like(states[0])
    for st in states:
        for a, t in zip(acc, st):
            a.add_(t)
    for a in acc:
        a.div_(len(states))
    return acc


def _unflatten(flat: torch.Tensor, template: State) -> State:
    out: State = []
    off = 0
    for t in template:
        k = t.numel()
        out.append(flat[off:off + k].reshape(t.shape).clone())
        off += k
    return out


class ConsensusBackend:
    """Interface. `average` maps N per-node states -> N per-node states."""

    name = "base"

    def __init__(self, n_agents: int, **kwargs) -> None:
        self.n_agents = n_agents
        self.last_comm_bytes = 0

    def average(self, states: Sequence[State]) -> List[State]:
        raise NotImplementedError

    def bytes_per_round(self, numel: int, dtype_size: int = 4) -> int:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name, "n_agents": self.n_agents}


class CentralizedCoordinator(ConsensusBackend):
    """One rendezvous node receives every state, averages, and broadcasts."""

    name = "coordinator"

    def average(self, states: Sequence[State]) -> List[State]:
        mean = _exact_mean(states)
        numel = sum(t.numel() for t in states[0])
        self.last_comm_bytes = self.bytes_per_round(
            numel, states[0][0].element_size() if states[0] else 4)
        # Clone per node so a later in-place update on node i cannot alias node j.
        return [[t.clone() for t in mean] for _ in range(self.n_agents)]

    def bytes_per_round(self, numel: int, dtype_size: int = 4) -> int:
        # N gathers in + N broadcasts out, all through one node.
        return 2 * self.n_agents * numel * dtype_size


class RingAllReduce(ConsensusBackend):
    """Bandwidth-optimal all-reduce on a directed ring (Patarasuk & Yuan, 2009).

    Implements the actual reduce-scatter / all-gather schedule, so the step count
    and per-edge traffic are those of a real ring all-reduce.
    """

    name = "ring_allreduce"

    def average(self, states: Sequence[State]) -> List[State]:
        n = self.n_agents
        if n == 1:
            return [[t.clone() for t in states[0]]]

        flat = [torch.cat([t.reshape(-1) for t in st]) for st in states]
        total = flat[0].numel()
        # N contiguous chunks; the last absorbs the remainder.
        bounds = [(k * total) // n for k in range(n)] + [total]
        chunks = [[f[bounds[k]:bounds[k + 1]].clone() for k in range(n)] for f in flat]

        # reduce-scatter: after N-1 steps node i holds the full sum of chunk
        # (i+1) mod N. Node i sends chunk (i - s) mod N at step s.
        for s in range(n - 1):
            sends = [((i - s) % n, chunks[i][(i - s) % n].clone()) for i in range(n)]
            for i in range(n):
                src = (i - 1) % n
                cidx, payload = sends[src]
                chunks[i][cidx] = chunks[i][cidx] + payload

        # all-gather: circulate the reduced chunks so every node ends up whole.
        for s in range(n - 1):
            sends = [((i + 1 - s) % n, chunks[i][(i + 1 - s) % n].clone()) for i in range(n)]
            for i in range(n):
                src = (i - 1) % n
                cidx, payload = sends[src]
                chunks[i][cidx] = payload

        out: List[State] = []
        for i in range(n):
            merged = torch.cat(chunks[i]) / float(n)
            out.append(_unflatten(merged, states[0]))

        self.last_comm_bytes = self.bytes_per_round(
            total, states[0][0].element_size() if states[0] else 4)
        return out

    def bytes_per_round(self, numel: int, dtype_size: int = 4) -> int:
        # 2(N-1) steps, each edge carries S/N floats.
        return 2 * (self.n_agents - 1) * (numel // max(1, self.n_agents)) * dtype_size


class Gossip(ConsensusBackend):
    """`gossip_steps` (tau) rounds of Metropolis-weighted neighbour averaging.

    Nodes do not reach the exact mean; `residual_to_mean` records how far the
    last call got.
    """

    name = "gossip"

    def __init__(self, n_agents: int, topology: str = "ring", gossip_steps: int = 1,
                 seed: int = 0, **kwargs) -> None:
        super().__init__(n_agents)
        self.topology = topology
        self.gossip_steps = max(1, int(gossip_steps))
        self.adj = build_topology(n_agents, topology, seed=seed)
        self.W = metropolis_weights(self.adj)
        self.spectral_gap = spectral_gap(self.adj)
        self.residual_to_mean: Optional[float] = None

    def average(self, states: Sequence[State]) -> List[State]:
        cur = [[t.clone() for t in st] for st in states]
        for _ in range(self.gossip_steps):
            nxt: List[State] = []
            for i in range(self.n_agents):
                acc = _zeros_like(cur[i])
                for j, w in self.W[i].items():
                    for a, t in zip(acc, cur[j]):
                        a.add_(t, alpha=w)
                nxt.append(acc)
            cur = nxt

        true_mean = _exact_mean(states)
        worst = 0.0
        for st in cur:
            for a, b in zip(st, true_mean):
                worst = max(worst, float((a - b).abs().max()))
        self.residual_to_mean = worst

        numel = sum(t.numel() for t in states[0])
        self.last_comm_bytes = self.bytes_per_round(
            numel, states[0][0].element_size() if states[0] else 4)
        return cur

    def bytes_per_round(self, numel: int, dtype_size: int = 4) -> int:
        # Each gossip round pushes the full state across every directed edge.
        return self.gossip_steps * 2 * n_edges(self.adj) * numel * dtype_size

    def describe(self) -> dict:
        d = super().describe()
        d.update(topology=self.topology, gossip_steps=self.gossip_steps,
                 spectral_gap=self.spectral_gap, n_edges=n_edges(self.adj))
        return d


def build_backend(name: str, n_agents: int, **kwargs) -> ConsensusBackend:
    if name == "coordinator":
        return CentralizedCoordinator(n_agents)
    if name in ("ring", "ring_allreduce"):
        return RingAllReduce(n_agents)
    if name == "gossip":
        return Gossip(n_agents, **kwargs)
    raise ValueError(f"unknown consensus backend {name!r}")


def max_abs_deviation_from_mean(states: Sequence[State]) -> float:
    """Largest elementwise distance of any node's state from the exact mean."""
    mean = _exact_mean(states)
    worst = 0.0
    for st in states:
        for a, b in zip(st, mean):
            worst = max(worst, float((a - b).abs().max()))
    return worst


"""Consensus layer: communication graphs, averaging backends and the ADMM updates."""
from .admm import ConsensusADMM, DEFAULT_CLIP, DEFAULT_ETA, DEFAULT_RHO
from .backends import (
    CentralizedCoordinator,
    ConsensusBackend,
    Gossip,
    RingAllReduce,
    build_backend,
    max_abs_deviation_from_mean,
)
from .topology import TOPOLOGIES, build_topology, metropolis_weights, n_edges, spectral_gap

__all__ = [
    "ConsensusADMM", "DEFAULT_ETA", "DEFAULT_RHO", "DEFAULT_CLIP",
    "ConsensusBackend", "CentralizedCoordinator", "RingAllReduce", "Gossip",
    "build_backend", "max_abs_deviation_from_mean",
    "build_topology", "metropolis_weights", "spectral_gap", "n_edges", "TOPOLOGIES",
]

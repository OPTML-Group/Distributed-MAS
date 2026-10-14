"""Consensus ADMM for distributed MAS training (Sec. 4, Algorithm A1).

Notation follows the paper. Node i holds a local copy x_i of the trainable LoRA
factors, a consensus variable z_i and a scaled dual variable u_i (u_i = y_i / rho,
lambda = 1 / rho). One ADMM round t -> t+1:

    x-step (7)     x_i <- prox_{lambda f_i}(a_i),   a_i = z_i - u_i
                         = argmin_x f_i(x) + (rho/2) ||x - a_i||^2
                   solved inexactly with L Adam steps STARTING FROM the anchor
                   a_i (App. B.1), so the proximal term is zero at the first
                   inner step. Local to node i.
    z-step (9-10)  z_i <- consensus average of the local variables
                   (ring all-reduce, gossip, or an exact coordinator).
                   The only step that communicates.
    dual step      u_i <- u_i + x_i - z_i.   Local to node i.

Notes on the implementation:
  * All state is kept in FP32 regardless of the backbone dtype; it is cast to the
    model dtype only when loaded into the adapters.
  * The subproblem gradient r = grad f_i(x) + rho (x - a_i) is clipped as a whole
    (global norm `clip`) before the Adam update, so clipping only rescales the
    step and never changes the objective being descended.
  * Non-finite gradients are dropped (and counted) instead of entering the state.

The z-step averages x_j. With exact averaging this equals the average of
b_j = x_j + u_j in Eq. (10): u_i starts at zero and every dual step adds
x_i - mean(x), so sum_i u_i stays zero (`residuals()["dual_invariant"]`
reports ||sum_i u_i|| every round). Under gossip each node mixes its
neighbours' x_j.
"""
from typing import List, Optional

import torch

from .backends import ConsensusBackend

State = List[torch.Tensor]

DEFAULT_ETA = 1e-3      # Adam learning rate of the x-step (Table A2)
DEFAULT_RHO = 0.03      # ADMM penalty (Table A2, safety)
DEFAULT_CLIP = 0.01     # global gradient clip (Table A2)


class ConsensusADMM:
    """Holds {x_i}, {z_i}, {u_i} for N nodes and runs the three ADMM steps."""

    def __init__(
        self,
        n_agents: int,
        template: State,
        backend: ConsensusBackend,
        *,
        eta: float = DEFAULT_ETA,
        rho: float = DEFAULT_RHO,
        clip: float = DEFAULT_CLIP,
        device=None,
    ) -> None:
        self.n_agents = n_agents
        self.backend = backend
        self.eta = float(eta)
        self.rho = float(rho)
        self.clip = float(clip)
        # Ablation switch (Table 2, "ADMM-NoDual"): keep u_i fixed at zero.
        self.disable_dual = False
        self._template = [t.detach() for t in template]
        self._device = device or (template[0].device if template else torch.device("cpu"))

        self.x: List[State] = []
        self.u: List[State] = []
        self.z: List[State] = []
        self.nonfinite_grads = 0
        self.last_prox_grad_ratio = 0.0
        self.history: List[dict] = []
        self.reset()

    # ------------------------------------------------------------------ state

    def _zeros(self) -> State:
        return [torch.zeros(t.shape, dtype=torch.float32, device=self._device)
                for t in self._template]

    def _init(self) -> State:
        """x^(0) = z^(0) = the deployed adapter (App. B.1)."""
        return [t.detach().to(dtype=torch.float32, device=self._device).clone()
                for t in self._template]

    def reset(self) -> None:
        self.x = [self._init() for _ in range(self.n_agents)]
        self.u = [self._zeros() for _ in range(self.n_agents)]
        self.z = [self._init() for _ in range(self.n_agents)]
        self.nonfinite_grads = 0
        self.history = []

    def numel(self) -> int:
        return sum(t.numel() for t in self._template)

    # ------------------------------------------------- x-step, Eq. (7)

    def x_step(self, agent_idx: int, grad_fn, local_steps: int = 16,
               betas=(0.9, 0.999), eps: float = 1e-8) -> dict:
        """x_i <- argmin_x f_i(x) + (rho/2)||x - a_i||^2,  a_i = z_i - u_i.

        Approximated by `local_steps` Adam steps initialised AT the anchor a_i.
        `grad_fn(x) -> (grads, loss)` loads x into the model and returns the
        gradient of f_i at x (a fresh minibatch per call). Returns the inner-loop
        telemetry.
        """
        a = [zb - u for zb, u in zip(self.z[agent_idx], self.u[agent_idx])]
        x = [t.detach().clone() for t in a]
        losses, gnorms, ratios = [], [], []
        clip_hits = 0
        stat_trace = []          # per-inner-step ||r|| / ||r_0||

        m = [torch.zeros_like(t) for t in x]
        vv = [torch.zeros_like(t) for t in x]
        step = 0

        g0_norm = None
        stat_norm = None

        for _ in range(int(local_steps)):
            grads, aux = grad_fn(x)
            if grads is None:
                break
            clean = [torch.zeros_like(slot) if g is None
                     else g.detach().to(dtype=torch.float32, device=slot.device)
                     for slot, g in zip(x, grads)]
            sq = sum(float((g ** 2).sum()) for g in clean)
            if not (sq == sq) or sq == float("inf"):
                self.nonfinite_grads += 1
                continue
            gnorm = sq ** 0.5

            # Subproblem gradient r = grad f_i(x) + rho (x - a_i).
            r_list, stat_sq, pull_sq = [], 0.0, 0.0
            for slot, g, anchor in zip(x, clean, a):
                pull = self.rho * (slot - anchor)
                pull_sq += float((pull ** 2).sum())
                r = g + pull
                r_list.append(r)
                stat_sq += float((r ** 2).sum())
            stat_norm = stat_sq ** 0.5
            if g0_norm is None:
                g0_norm = stat_norm            # == ||grad f_i(a_i)||: the pull is 0 at x = a_i
            stat_trace.append(round(stat_norm / max(g0_norm, 1e-30), 6))
            scale = (self.clip / (stat_norm + 1e-12)
                     if (self.clip > 0 and stat_norm > self.clip) else 1.0)
            if scale < 1.0:
                clip_hits += 1

            step += 1
            with torch.no_grad():
                for j, slot in enumerate(x):
                    upd = r_list[j] * scale
                    m[j].mul_(betas[0]).add_(upd, alpha=1 - betas[0])
                    vv[j].mul_(betas[1]).addcmul_(upd, upd, value=1 - betas[1])
                    mhat = m[j] / (1 - betas[0] ** step)
                    vhat = vv[j] / (1 - betas[1] ** step)
                    slot.add_(-(self.eta) * mhat / (vhat.sqrt() + eps))
            gnorms.append(gnorm)
            ratios.append((pull_sq ** 0.5) / max(1e-12, gnorm * scale))
            if aux is not None:
                losses.append(aux)

        with torch.no_grad():
            for slot, new in zip(self.x[agent_idx], x):
                slot.copy_(new)
        self.last_prox_grad_ratio = ratios[-1] if ratios else 0.0
        return {
            # ||r(x_final)|| / ||r(a_i)||: 1.0 means the inner loop went nowhere.
            "prox_residual_rel": (round(stat_norm / max(g0_norm or 1e-30, 1e-30), 6)
                                  if stat_norm is not None else None),
            "prox_residual_trace": stat_trace[-8:],
            "clip_hits": clip_hits,
            "inner_steps": len(gnorms),
            "f": round(sum(losses) / len(losses), 5) if losses else None,
            "grad_norm": round(sum(gnorms) / len(gnorms), 6) if gnorms else None,
            # ~0: the proximal term never engaged; >>1: it dominated the local loss.
            "prox_grad_ratio": round(sum(ratios) / len(ratios), 6) if ratios else 0.0,
            "moved": round(float(sum(((p - q) ** 2).sum() for p, q in zip(x, a)) ** 0.5), 9),
        }

    # -------------------------------------------- z-step, Eqs. (9)-(10)

    def z_step(self) -> None:
        """Consensus averaging (see module docstring). The only step that communicates."""
        self.z = self.backend.average(self.x)

    # ------------------------------------------------------------ dual step

    def dual_step(self) -> None:
        if self.disable_dual:
            return
        with torch.no_grad():
            for i in range(self.n_agents):
                for u, x, z in zip(self.u[i], self.x[i], self.z[i]):
                    u.add_(x - z)

    # ------------------------------------------------------------- metrics

    def residuals(self) -> dict:
        """Primal residual (sum_i ||x_i - z_i||^2)^(1/2), dual invariant ||sum_i u_i||,
        and ||x_1||, the normaliser of the residual plotted in Fig. 3."""
        primal_sq = 0.0
        for i in range(self.n_agents):
            for x, z in zip(self.x[i], self.z[i]):
                primal_sq += float(((x - z) ** 2).sum())
        u_sum = [torch.zeros_like(t) for t in self.u[0]]
        for i in range(self.n_agents):
            for acc, u in zip(u_sum, self.u[i]):
                acc.add_(u)
        dual_sq = sum(float((t ** 2).sum()) for t in u_sum)
        x1_sq = sum(float((t ** 2).sum()) for t in self.x[0])
        primal = primal_sq ** 0.5
        x1 = x1_sq ** 0.5
        return {
            "primal_residual": primal,
            "normalized_primal_residual": primal / x1 if x1 > 0 else None,
            "dual_invariant": dual_sq ** 0.5,
            "x1_norm": x1,
        }

    def log_round(self, k: int, extra: Optional[dict] = None) -> dict:
        rec = {"round": k, **self.residuals()}
        rec["prox_grad_ratio"] = round(self.last_prox_grad_ratio, 6)
        rec["comm_bytes"] = int(getattr(self.backend, "last_comm_bytes", 0))
        gap = getattr(self.backend, "residual_to_mean", None)
        if gap is not None:
            rec["consensus_error"] = gap
        if extra:
            rec.update(extra)
        self.history.append(rec)
        return rec


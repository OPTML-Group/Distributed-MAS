"""The trainable adapter theta_i of each agent (App. B.1, C.2).

Every agent's local update to a base weight is Delta W_i = B_i A: a rank-r LoRA
adapter (scaling alpha / r) on the seven projections (q, k, v, o, gate, up, down)
of the selected decoder layers. The down-projection A is drawn once, shared by
all agents and never trained; only B_i is optimised. Because A is shared and
fixed, averaging the B_i in the consensus step is exactly averaging the weight
updates Delta W_i:

    mean_i(B_i A) = (mean_i B_i) A.

B starts at zero, so an untrained adapter leaves the backbone unchanged.

Adapters are injected into the backbone once. A node's state is a plain list of
FP32 tensors (one B per adapted projection) held by the ADMM object; `load()`
copies a state into the adapters before that agent's forward pass and
`enable()` / `disable()` toggle the whole bank. A disabled adapter skips the
delta entirely, so a frozen agent is bit-identical to the untrained backbone.

Reproducing A: A is drawn with `kaiming_uniform_` from the global torch CPU RNG
when the bank is built, i.e. it is determined by `set_seed(seed)` and by
whatever consumed the CPU RNG between the seed call and the bank construction
(train.py / run.py snapshot and restore the RNG around extra model loads so
training and evaluation rebuild the same A). `a_fingerprint()` hashes A; it is
stored in every checkpoint and checked at evaluation time.
"""
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

State = List[torch.Tensor]

TARGET_SETS = {
    "attn_qv": ["q_proj", "v_proj"],
    "attn_all": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
}


def resolve_layers(n_layers: int, last_n: int) -> List[int]:
    """The last `last_n` decoder layers (all layers when last_n <= 0 or >= n_layers)."""
    if last_n <= 0 or last_n >= n_layers:
        return list(range(n_layers))
    return list(range(n_layers - last_n, n_layers))


class LoraLinear(nn.Module):
    """Frozen base Linear plus an additive low-rank delta: y = Wx + scale * B(Ax)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=dt),
                              requires_grad=False)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt),
                              requires_grad=False)
        self.enabled = False

    def init_A(self) -> None:
        """Draw the frozen basis A from the global CPU RNG (see module docstring)."""
        with torch.no_grad():
            a = torch.empty(self.A.shape, device="cpu", dtype=torch.float32)
            nn.init.kaiming_uniform_(a, a=5 ** 0.5)
            self.A.copy_(a.to(self.A.device, self.A.dtype))

    def effective_delta_w(self) -> torch.Tensor:
        return self.scale * (self.B.detach().float() @ self.A.detach().float())

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + self.scale * F.linear(F.linear(x, self.A), self.B)
        return out


class LoraFastState:
    """Injects B-only LoRA adapters into a backbone and swaps node states in and out."""

    def __init__(
        self,
        model,
        *,
        rank: int = 16,
        alpha: float = 16.0,
        layers_last: int = 36,
        targets: str = "attn_mlp",
    ) -> None:
        self.model = model
        self.rank = rank
        hf = model.model
        n_layers = int(hf.config.num_hidden_layers)
        self.layer_ids = resolve_layers(n_layers, layers_last)
        target_names = TARGET_SETS[targets] if isinstance(targets, str) else list(targets)

        self.adapters: List[LoraLinear] = []
        self._adapter_names: List[str] = []
        decoder = hf.model.layers
        for li in self.layer_ids:
            block = decoder[li]
            for tname in target_names:
                parent, attr = self._resolve(block, tname)
                if parent is None:
                    continue
                base = getattr(parent, attr)
                if not isinstance(base, nn.Linear):
                    continue          # already wrapped, or an unexpected module type
                wrapped = LoraLinear(base, rank, alpha)
                setattr(parent, attr, wrapped)
                self.adapters.append(wrapped)
                scope = "self_attn" if tname in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                self._adapter_names.append(f"layers.{li}.{scope}.{tname}")

        if not self.adapters:
            raise RuntimeError(
                f"No LoRA adapters injected. Checked layers {self.layer_ids} for "
                f"{target_names}; the backbone may use a different module layout.")

        # One shared A across every agent (required for exact B-only consensus).
        for a in self.adapters:
            a.init_A()
        self._trainables: List[nn.Parameter] = [a.B for a in self.adapters]

    @staticmethod
    def _resolve(block, tname):
        if tname in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return getattr(block, "self_attn", None), tname
        if tname in ("gate_proj", "up_proj", "down_proj"):
            return getattr(block, "mlp", None), tname
        return None, None

    # ------------------------------------------------------------ interface

    def template(self) -> State:
        """FP32 copy of the current trainable factors (the initial x^(0) = z^(0))."""
        return [p.detach().to(torch.float32).clone() for p in self._trainables]

    def block_names(self) -> List[str]:
        return [f"{n}.B" for n in self._adapter_names]

    def numel(self) -> int:
        return sum(p.numel() for p in self._trainables)

    def a_fingerprint(self) -> str:
        """Hash of the whole frozen basis A."""
        import hashlib
        h = hashlib.sha256()
        for ad in self.adapters:
            h.update(ad.A.detach().to("cpu", torch.float32).numpy().tobytes())
        return h.hexdigest()[:16]

    def describe(self) -> dict:
        return {
            "kind": "lora",
            "a_fingerprint": self.a_fingerprint(),
            "rank": self.rank,
            "layers": self.layer_ids,
            "n_adapters": len(self.adapters),
            "train_factors": "B",
            "numel": self.numel(),
            "comm_floats_per_round": self.numel(),
        }

    # ------------------------------------------------------------- plumbing

    def load(self, state: State) -> None:
        """Copy a node's state into the adapter bank (cast to the backbone dtype)."""
        with torch.no_grad():
            for p, t in zip(self._trainables, state):
                p.copy_(t.to(p.dtype))

    def as_leaves(self, state: State) -> State:
        """Load `state` into the adapters and make them differentiable.

        Gradients are then taken with torch.autograd.grad w.r.t. the parameter
        tensors. Keeping the master copy outside the module lets N nodes share
        one injected bank.
        """
        leaves = [t.detach().clone().requires_grad_(True) for t in state]
        with torch.no_grad():
            for p, t in zip(self._trainables, leaves):
                p.copy_(t.to(p.dtype))
        for p in self._trainables:
            p.requires_grad_(True)
        return leaves

    def zero_grad(self) -> None:
        for p in self._trainables:
            p.grad = None

    def enable(self) -> None:
        for a in self.adapters:
            a.enabled = True

    def disable(self) -> None:
        for a in self.adapters:
            a.enabled = False

    def trainable_params(self) -> List[nn.Parameter]:
        return self._trainables

    # ------------------------------------------------------------ diagnostic

    def verify_delta_w_consensus(self, states: Sequence[State], averaged: State,
                                 tol: float = 1e-5) -> dict:
        """Check numerically that averaging B equals averaging Delta W = B A."""
        n = len(states)
        num_sq = 0.0
        den_sq = 0.0
        for k, a in enumerate(self.adapters):
            A = a.A.detach().double()
            per_i = [a.scale * (states[i][k].double() @ A) for i in range(n)]
            got = a.scale * (averaged[k].double() @ A)
            want = sum(per_i) / n
            num_sq += float(((want - got) ** 2).sum())
            den_sq += float((want ** 2).sum())
        rel = (num_sq ** 0.5) / (den_sq ** 0.5 + 1e-12)
        return {
            "delta_w_consensus_rel_error": rel,
            "n_participants": n,
            "exact": bool(n > 1 and rel <= tol),
        }


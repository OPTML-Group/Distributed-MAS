"""CPU unit tests for the consensus layer and the ADMM updates (no model download).

    python tests/test_admm.py          (or: python -m pytest tests -q)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consensus import ConsensusADMM, build_backend, build_topology, metropolis_weights  # noqa: E402
from consensus.backends import _exact_mean  # noqa: E402


def _states(n, shapes=((5, 3), (7,)), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(s, generator=g) for s in shapes] for _ in range(n)]


# ------------------------------------------------------------------ backends

def test_ring_allreduce_equals_exact_mean():
    for n in (2, 3, 4, 8):
        states = _states(n)
        mean = _exact_mean(states)
        for st in build_backend("ring", n).average(states):
            for a, b in zip(st, mean):
                assert torch.allclose(a, b, atol=1e-6)


def test_metropolis_weights_are_doubly_stochastic():
    for kind in ("ring", "chain", "complete", "expander"):
        W = metropolis_weights(build_topology(8, kind, seed=0))
        for i in range(8):
            assert abs(sum(W[i].values()) - 1.0) < 1e-12
            for j, w in W[i].items():
                assert abs(W[j][i] - w) < 1e-12


def test_gossip_preserves_mean_and_converges_with_more_rounds():
    states = _states(8)
    mean = _exact_mean(states)
    errs = []
    for tau in (1, 2, 4, 16, 64):
        out = build_backend("gossip", 8, topology="ring", gossip_steps=tau).average(states)
        # the network-wide average is preserved exactly
        for k in range(len(mean)):
            avg = sum(st[k] for st in out) / 8
            assert torch.allclose(avg, mean[k], atol=1e-5)
        errs.append(max(float((st[k] - mean[k]).abs().max()) for st in out for k in range(len(mean))))
    assert errs == sorted(errs, reverse=True) and errs[-1] < 1e-3 < errs[0]


def test_gossip_only_mixes_graph_neighbours():
    n = 6
    states = [[torch.zeros(3)] for _ in range(n)]
    states[0] = [torch.ones(3)]
    out = build_backend("gossip", n, topology="chain", gossip_steps=1).average(states)
    touched = [i for i in range(n) if float(out[i][0].abs().sum()) > 0]
    assert touched == [0, 1]


# ---------------------------------------------------------------- ADMM steps

class Quadratic:
    """f_i(x) = 0.5 ||x - c_i||^2; the consensus optimum is mean_i c_i."""

    def __init__(self, n, dim=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.c = [torch.randn(dim, generator=g) for _ in range(n)]

    def grad_fn(self, i):
        return lambda x: ([x[0] - self.c[i]], float(0.5 * ((x[0] - self.c[i]) ** 2).sum()))


def _run(n, rounds, backend="ring", rho=1.0, no_dual=False, local_only=False, steps=50):
    q = Quadratic(n)
    admm = ConsensusADMM(n, [torch.zeros(4)], build_backend(backend, n),
                         eta=0.05, rho=rho, clip=0.0)
    admm.disable_dual = no_dual
    for _ in range(rounds):
        for i in range(n):
            admm.x_step(i, q.grad_fn(i), local_steps=steps)
        if local_only:
            for i in range(n):
                admm.z[i] = [t.clone() for t in admm.x[i]]
        else:
            admm.z_step()
            admm.dual_step()
    return admm, torch.stack(q.c).mean(0)


def test_admm_reaches_the_consensus_optimum():
    # The x-step is an inexact prox (a fixed number of Adam steps), so z lands
    # within O(eta) of the optimum while consensus itself is exact.
    admm, opt = _run(4, rounds=60)
    for i in range(4):
        assert torch.allclose(admm.z[i][0], opt, atol=5e-2)
    assert admm.residuals()["primal_residual"] < 1e-4


def test_dual_invariant_holds_under_exact_averaging():
    admm, _ = _run(4, rounds=10)
    assert admm.residuals()["dual_invariant"] < 1e-5


def test_ring_and_coordinator_give_the_same_iterates():
    a, _ = _run(4, rounds=10, backend="ring")
    b, _ = _run(4, rounds=10, backend="coordinator")
    for i in range(4):
        assert torch.allclose(a.x[i][0], b.x[i][0], atol=1e-5)


def test_no_dual_and_local_only_ablations():
    nd, opt = _run(4, rounds=60, no_dual=True)
    assert all(float(u[0].abs().max()) == 0.0 for u in nd.u)
    # without the dual feedback the proximal fixed point is biased towards c_i
    full, _ = _run(4, rounds=60)
    assert (nd.x[0][0] - opt).norm() > (full.x[0][0] - opt).norm()
    lo, _ = _run(4, rounds=60, local_only=True)
    q = Quadratic(4)
    for i in range(4):                       # each node solves its own problem
        assert torch.allclose(lo.x[i][0], q.c[i], atol=5e-2)


def test_x_step_starts_at_the_anchor_z_minus_u():
    admm = ConsensusADMM(2, [torch.zeros(3)], build_backend("ring", 2), eta=0.1, rho=1.0)
    admm.z[0] = [torch.full((3,), 2.0)]
    admm.u[0] = [torch.full((3,), 0.5)]
    seen = []

    def grad_fn(x):
        seen.append(x[0].clone())
        return [torch.zeros(3)], 0.0

    admm.x_step(0, grad_fn, local_steps=1)
    assert torch.allclose(seen[0], torch.full((3,), 1.5))


def test_nonfinite_gradients_are_dropped():
    admm = ConsensusADMM(1, [torch.zeros(2)], build_backend("ring", 1), eta=0.1, rho=0.0)
    admm.x_step(0, lambda x: ([torch.tensor([float("nan"), 0.0])], 0.0), local_steps=3)
    assert admm.nonfinite_grads == 3 and float(admm.x[0][0].abs().sum()) == 0.0


# ------------------------------------------------- objective and adapters

def _tiny_lm(seed=0):
    from types import SimpleNamespace
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      layer_types=["full_attention"] * 2)
    hf = Qwen3ForCausalLM(cfg).eval()
    return SimpleNamespace(model=hf, device=torch.device("cpu"))


def test_dpo_loss_is_log2_when_policy_equals_reference():
    from training import objectives as objt
    m = _tiny_lm()
    p, c, r = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]]), torch.tensor([[6, 7, 8]])
    with torch.no_grad():
        rc, rr = float(objt.seq_logp(m, p, c)), float(objt.seq_logp(m, p, r))
        loss = objt.dpo_loss(m, p, c, r, rc, rr, beta=0.1)
    assert abs(float(loss) - float(torch.log(torch.tensor(2.0)))) < 1e-5


def test_b_only_lora_averaging_is_exact_in_weight_space():
    from training.lora import LoraFastState
    m = _tiny_lm()
    fs = LoraFastState(m, rank=4, alpha=4.0, layers_last=2, targets="attn_mlp")
    assert len(fs.adapters) == 2 * 7                  # 7 projections in each of 2 layers
    states = [[torch.randn_like(t) for t in fs.template()] for _ in range(3)]
    avg = _exact_mean(states)
    chk = fs.verify_delta_w_consensus(states, avg)
    assert chk["exact"] and chk["delta_w_consensus_rel_error"] < 1e-6   # float32 averaging


def test_disabled_adapter_is_the_untouched_backbone():
    from training.lora import LoraFastState
    m = _tiny_lm()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        ref = m.model(input_ids=ids).logits.clone()
    fs = LoraFastState(m, rank=4, alpha=4.0, layers_last=2, targets="attn_mlp")
    fs.load([torch.randn_like(t) for t in fs.template()])
    with torch.no_grad():
        fs.disable()
        off = m.model(input_ids=ids).logits
        fs.enable()
        on = m.model(input_ids=ids).logits
    assert torch.equal(off, ref) and not torch.allclose(on, ref)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")


"""Distributed MAS training via consensus ADMM (Algorithm A1).

One training node per adapted agent ("seat" partition). Node i owns the local
dataset built from agent i's messages and runs, every ADMM round:

    x-step   L Adam steps on f_i(x) + (rho/2)||x - (z_i - u_i)||^2      (Eq. 7)
             f_i = DPO (Eq. 8) + beta_KL * KL(pi_ref || pi_theta)
    z-step   consensus averaging across nodes                       (Eqs. 9-10)
    dual     u_i <- u_i + x_i - z_i

Variants used in the paper (Table 1/2):
    Centralized    all nodes' examples pooled into ONE node (`pooled=True` in
                   train.py); the same local solver is used, and the consensus /
                   dual steps are identities for a single node.
    ADMM-Local     `local_only`: no z-step and no dual step; each node trains
                   alone (z_i := x_i, so every round restarts from its own x_i).
    ADMM-NoDual    `no_dual`: consensus averaging but u_i fixed at zero
                   (periodic averaging with a proximal term).

The N nodes share one adapter bank injected into the backbone: before each
gradient evaluation the node's current iterate is loaded into the bank, so node
i's gradient depends only on x_i and its own data.
"""
import contextlib
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from consensus import ConsensusADMM, build_backend, max_abs_deviation_from_mean
from mas import resolve_prompt_builder
from training import objectives as objt
from training.lora import LoraFastState
from training.rollouts import FitExample


class ADMMTrainer:
    def __init__(
        self,
        model,
        fast_state: LoraFastState,
        n_nodes: int,
        *,
        adapt_idx: Optional[Sequence[int]] = None,
        consensus: str = "ring",
        topology: str = "ring",
        gossip_steps: int = 1,
        eta: float = 1e-3,
        rho: float = 0.03,
        grad_clip: float = 0.01,
        k_outer: int = 20,
        local_steps: int = 16,
        batch_size: int = 8,
        max_target_tokens: int = 4096,
        max_prompt_tokens: int = 8192,
        grad_checkpointing: bool = True,
        dpo_beta: float = 0.1,
        kl_beta: float = 0.1,
        kl_ref: str = "frozen",
        role_balance: bool = True,
        target_span: str = "full",
        fit_prompt: Optional[str] = None,
        fit_args=None,
        local_only: bool = False,
        no_dual: bool = False,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.fast_state = fast_state
        self.n_agents = n_nodes
        # Under centralized training one variable drives every adapted agent, so
        # adapt_idx can be longer than n_nodes.
        self.adapt_idx = list(adapt_idx) if adapt_idx is not None else list(range(n_nodes))
        self.k_outer = int(k_outer)
        self.local_steps = max(1, int(local_steps))
        self.batch_size = max(1, int(batch_size))
        self.max_target_tokens = int(max_target_tokens)
        self.max_prompt_tokens = int(max_prompt_tokens)
        # "full": fit the whole message including its <think> block.
        # "visible": fit only the text after </think>, with the think block moved
        # into the context.
        self.target_span = str(target_span)
        # Fit-time prompt re-rendering: targets collected under a hardened
        # (teacher) prompt are fit under the deployment prompt `fit_prompt`.
        self.fit_prompt = fit_prompt
        self.fit_args = fit_args
        self.fit_builder = resolve_prompt_builder(fit_prompt) if fit_prompt else None
        self._fit_rerendered = 0
        self._fit_rerendered_changed = 0
        self._skipped_no_close = 0
        self.grad_checkpointing = bool(grad_checkpointing)
        self.dpo_beta = float(dpo_beta)
        self.kl_beta = float(kl_beta)
        # KL reference: "frozen" = the backbone with the adapter disabled;
        # "collect" = the policy that collected the data (the warm-start adapter
        # if --init_state is given, otherwise the backbone).
        self.kl_ref = kl_ref
        self.role_balance = bool(role_balance)
        self.local_only = bool(local_only)
        self._kl_sum = 0.0
        self._kl_n = 0
        self._ref_cache = {}
        self._skipped_long_prompt = 0
        self._collect_D = None
        self._role_index: Dict[int, Dict] = {}
        self._hf = getattr(model, "model", None)
        if self.grad_checkpointing:
            hf = self._hf
            if hasattr(hf, "gradient_checkpointing_enable"):
                # use_reentrant=False is required for torch.autograd.grad to
                # return gradients through checkpointed blocks.
                hf.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                if hasattr(hf, "enable_input_require_grads"):
                    hf.enable_input_require_grads()
        self.seed = seed

        self.backend = build_backend(consensus, n_nodes, topology=topology,
                                     gossip_steps=gossip_steps, seed=seed)
        self.admm = ConsensusADMM(n_nodes, fast_state.template(), self.backend,
                                  eta=eta, rho=rho, clip=grad_clip, device=model.device)
        self.admm.disable_dual = bool(no_dual)
        self.rho = rho
        self.consensus_name = consensus
        self._rng = torch.Generator().manual_seed(seed)
        self._skipped_long = 0
        self._empty_batches = 0

    # ----------------------------------------------------------------- loss

    @staticmethod
    def _visible_split(output: str) -> Optional[Tuple[str, str]]:
        """(prefix through </think>, visible remainder), or None if unclosed."""
        m = re.search(r"</think>", output)
        if not m:
            return None
        return output[:m.end()], output[m.end():]

    def _example_loss(self, ex: FitExample) -> Optional[torch.Tensor]:
        """f_i on one example: DPO (or weighted NLL) plus beta_KL * forward KL."""
        tok = self.model.tokenizer
        if self.fit_builder is not None and not getattr(ex, "_fit_done", False):
            if ex.role is not None:
                msgs = self.fit_builder(role=ex.role, question=ex.question,
                                        context=ex.context, method="text_mas",
                                        args=self.fit_args)
                new = self.model.render_chat(msgs, add_generation_prompt=True)
                self._fit_rerendered += 1
                if new != ex.prompt:
                    self._fit_rerendered_changed += 1
                ex.prompt = new
            ex._fit_done = True
        prompt_text, out_text, rej_text = ex.prompt, ex.output, ex.output_rej
        if self.target_span == "visible":
            spl = self._visible_split(ex.output)
            if spl is None or not spl[1].strip():
                self._skipped_no_close += 1
                return None
            prompt_text, out_text = ex.prompt + spl[0], spl[1]
            if rej_text is not None:
                rspl = self._visible_split(rej_text)
                if rspl is None or not rspl[1].strip():
                    self._skipped_no_close += 1
                    return None
                rej_text = rspl[1]
        p = tok(prompt_text, add_special_tokens=False, return_tensors="pt")
        o = tok(out_text, add_special_tokens=False, return_tensors="pt")
        # Left-truncate the context: its tail carries the role instruction and
        # the immediate upstream message.
        pid = p.input_ids
        if pid.shape[1] > self.max_prompt_tokens:
            pid = pid[:, -self.max_prompt_tokens:]
            self._skipped_long_prompt += 1
        prompt_ids = pid.to(self.model.device)
        out_ids = o.input_ids[:, : self.max_target_tokens].to(self.model.device)
        if out_ids.shape[1] == 0:
            return None
        if o.input_ids.shape[1] > self.max_target_tokens:
            self._skipped_long += 1
        # The collection-policy reference must be computed before this example's
        # graph is built (it temporarily overwrites the adapter weights).
        collect_ref = None
        if self.kl_beta > 0 and self.kl_ref == "collect" and self.fast_state is not None:
            collect_ref = self._under_collect_policy(
                lambda: objt.span_logprobs(self.model, prompt_ids, out_ids))
        if rej_text is not None:
            rej = tok(rej_text, add_special_tokens=False,
                      return_tensors="pt").input_ids[:, : self.max_target_tokens]
            if rej.shape[1] == 0:
                return None
            rej = rej.to(self.model.device)
            rc, rr = self._ref_logps(ex, prompt_ids, out_ids, rej)
            base = objt.dpo_loss(self.model, prompt_ids, out_ids, rej, rc, rr,
                                 beta=self.dpo_beta)
        else:
            base = objt.advantage_ce(self.model, prompt_ids, out_ids, ex.advantage)
        return self._kl_regularized(base, prompt_ids, out_ids, ref=collect_ref)

    def _under_collect_policy(self, fn):
        """Run fn() under the data-collection policy, then restore the adapters."""
        params = self.fast_state.trainable_params()
        saved = [p.detach().clone() for p in params]
        was = [a.enabled for a in self.fast_state.adapters]
        try:
            with torch.no_grad():
                if self._collect_D is None:
                    self.fast_state.disable()
                else:
                    self.fast_state.load(self._collect_D)
                    self.fast_state.enable()
                return fn()
        finally:
            with torch.no_grad():
                for p, t in zip(params, saved):
                    p.copy_(t)
            for a, w in zip(self.fast_state.adapters, was):
                a.enabled = w

    def _kl_regularized(self, base, prompt_ids, out_ids, ref=None):
        """base + beta_KL * KL(pi_ref || pi_theta) over the message tokens."""
        if base is None or self.kl_beta <= 0 or self.fast_state is None:
            return base
        if self.kl_ref == "collect":
            if ref is None:
                raise RuntimeError("kl_ref=collect requires the reference to be precomputed")
        else:
            with torch.no_grad():
                self.fast_state.disable()
                try:
                    ref = objt.span_logprobs(self.model, prompt_ids, out_ids)
                finally:
                    self.fast_state.enable()
        cur = objt.span_logprobs(self.model, prompt_ids, out_ids)
        kl = objt.kl_forward(ref, cur)
        self._kl_sum += float(kl.detach())
        self._kl_n += 1
        return base + self.kl_beta * kl

    def _ref_logps(self, ex, prompt_ids, chosen, rejected):
        """log pi_ref of both DPO messages (adapter disabled); cached per pair."""
        key = (ex.anchor, ex.agent_idx, hash(ex.output), hash(ex.output_rej))
        if key in self._ref_cache:
            return self._ref_cache[key]
        self.fast_state.disable()
        with torch.no_grad():
            rc = float(objt.seq_logp(self.model, prompt_ids, chosen))
            rr = float(objt.seq_logp(self.model, prompt_ids, rejected))
        self.fast_state.enable()
        self._ref_cache[key] = (rc, rr)
        return rc, rr

    def dev_nll(self, examples: Sequence[FitExample]) -> Dict:
        """Mean per-token NLL of held-out targets, frozen backbone vs current adapter.

        Call with the adapter state to evaluate already loaded into the bank.
        """
        tok = self.model.tokenizer

        def _pair(ex):
            prompt_text, out_text = ex.prompt, ex.output
            if self.target_span == "visible":
                spl = self._visible_split(ex.output)
                if spl is None or not spl[1].strip():
                    return None
                prompt_text, out_text = ex.prompt + spl[0], spl[1]
            p = tok(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
            if p.shape[1] > self.max_prompt_tokens:
                p = p[:, -self.max_prompt_tokens:]
            o = tok(out_text, add_special_tokens=False,
                    return_tensors="pt").input_ids[:, : self.max_target_tokens]
            if o.shape[1] == 0:
                return None
            return p.to(self.model.device), o.to(self.model.device)

        pairs = [q for q in (_pair(e) for e in examples) if q is not None]
        if not pairs:
            return {"n": 0}

        def _mean_nll() -> float:
            tot_nll, tot_tok = 0.0, 0
            with torch.no_grad():
                for pid, oid in pairs:
                    tot_nll += -float(objt.seq_logp(self.model, pid, oid))
                    tot_tok += int(oid.shape[1])
            return tot_nll / max(1, tot_tok)

        if getattr(self, "_dev_frozen", None) is None:
            self.fast_state.disable()
            self._dev_frozen = _mean_nll()
        frozen = self._dev_frozen
        self.fast_state.enable()
        trained = _mean_nll()
        return {"n": len(pairs),
                "dev_nll_frozen": round(frozen, 6),
                "dev_nll_trained": round(trained, 6),
                "dev_nll_delta": round(trained - frozen, 6)}

    def _batch_grads(self, batch: Sequence[FitExample]):
        """Mean gradient over a minibatch, accumulated one example at a time."""
        params = self.fast_state.trainable_params()
        acc = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
        total = 0.0
        used = 0
        for ex in batch:
            loss = self._example_loss(ex)
            if loss is None:
                continue
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for a, g in zip(acc, grads):
                if g is not None:
                    a.add_(g.detach().float())
            total += float(loss.detach())
            used += 1
        if used == 0:
            self._empty_batches += 1
            return None, 0.0
        for a in acc:
            a.div_(used)
        return acc, total / used

    def _sample(self, fitset: Sequence[FitExample]) -> List[FitExample]:
        """Draw a minibatch. With role_balance, first pick a role uniformly, then an
        example of that role (only matters when one node holds several roles)."""
        if not fitset:
            return []
        n = min(self.batch_size, len(fitset))
        if self.role_balance:
            key = id(fitset)
            ent = self._role_index.get(key)
            if ent is None or ent["n"] != len(fitset):
                by: Dict = {}
                for i, ex in enumerate(fitset):
                    by.setdefault(ex.role if ex.role is not None else ex.agent_idx, []).append(i)
                ent = {"n": len(fitset), "by": [by[r] for r in sorted(by, key=str)]}
                self._role_index[key] = ent
            pools = ent["by"]
            out = []
            for _ in range(n):
                pool = pools[int(torch.randint(len(pools), (1,), generator=self._rng))]
                out.append(fitset[pool[int(torch.randint(len(pool), (1,), generator=self._rng))]])
            return out
        idx = torch.randperm(len(fitset), generator=self._rng)[:n].tolist()
        return [fitset[i] for i in idx]

    # ------------------------------------------------------------------ fit

    @contextlib.contextmanager
    def _checkpointing_active(self):
        """Activation checkpointing in transformers only engages in train() mode;
        Qwen3 has no active dropout, so train() changes nothing else (asserted)."""
        hf = self._hf
        if not (self.grad_checkpointing and hf is not None):
            yield
            return
        live = [f"{n or '<root>'}(p={m.p})" for n, m in hf.named_modules()
                if isinstance(m, torch.nn.Dropout) and m.p > 0]
        if live:
            raise RuntimeError("train() would activate dropout: " + ", ".join(live[:5]))
        was = hf.training
        hf.train()
        try:
            yield
        finally:
            hf.train(was)

    def fit(self, fitsets: Sequence[Sequence[FitExample]],
            progress: Optional[Callable[[str], None]] = None,
            init_D: Optional[Sequence[torch.Tensor]] = None,
            dev_examples: Optional[Sequence[FitExample]] = None,
            dev_every: int = 0,
            recollect: Optional[Callable[[int], Sequence[Sequence[FitExample]]]] = None,
            recollect_every: int = 0) -> Dict:
        """Run `k_outer` ADMM rounds (Algorithm A1).

        init_D: warm-start adapter (x^(0) = z^(0) = init_D for every node).
        recollect(round) -> new fit sets: on-policy re-collection with the current
        consensus adapter every `recollect_every` rounds.
        """
        if len(fitsets) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} fit sets (one per node), "
                             f"got {len(fitsets)}")

        self.admm.reset()
        if init_D is not None:
            for i in range(self.n_agents):
                for dst, src in zip(self.admm.x[i], init_D):
                    dst.copy_(src.to(dst.device, dst.dtype))
                for dst, src in zip(self.admm.z[i], init_D):
                    dst.copy_(src.to(dst.device, dst.dtype))
        self._collect_D = ([t.detach().to(self.model.device) for t in init_D]
                           if init_D is not None else None)
        self.dev_curve: List[Dict] = []
        self.best_dev = None            # (dev_nll_trained, round, z on cpu)
        rounds: List[Dict] = []
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with self._checkpointing_active():
            for k in range(self.k_outer):
                per_agent = []
                # ---- x-step: role-local optimisation, independent across nodes
                for i in range(self.n_agents):
                    fs = fitsets[i]

                    def grad_fn(x, fs=fs):
                        batch = self._sample(fs)
                        if not batch:
                            return None, None
                        self.fast_state.as_leaves(x)
                        self.fast_state.enable()
                        self.fast_state.zero_grad()
                        with torch.enable_grad():
                            grads, loss = self._batch_grads(batch)
                        self.fast_state.disable()
                        return grads, loss

                    rec_i = self.admm.x_step(i, grad_fn, local_steps=self.local_steps)
                    rec_i.update(agent=i, fit_size=len(fs))
                    per_agent.append(rec_i)

                pre = [[t.clone() for t in st] for st in self.admm.x]
                if self.local_only:
                    # ADMM-Local: no communication, z_i := x_i.
                    with torch.no_grad():
                        for i in range(len(self.admm.x)):
                            for zb, x in zip(self.admm.z[i], self.admm.x[i]):
                                zb.copy_(x)
                else:
                    self.admm.z_step()       # Eqs. (9)-(10)
                    self.admm.dual_step()    # u_i <- u_i + x_i - z_i

                rec = self.admm.log_round(k, {
                    "agents": per_agent,
                    "backend_deviation_from_mean": max_abs_deviation_from_mean(self.admm.z),
                    "pre_consensus_spread": max_abs_deviation_from_mean(pre),
                })
                rounds.append(rec)

                if (recollect is not None and recollect_every > 0
                        and (k + 1) % recollect_every == 0 and (k + 1) < self.k_outer):
                    new_fs = recollect(k + 1)
                    if new_fs is None or len(new_fs) != self.n_agents:
                        raise SystemExit(f"re-collection after round {k + 1} returned "
                                         f"no usable fit sets")
                    fitsets = new_fs
                    rec["recollected_after_round"] = k + 1
                    if progress:
                        progress(f"  [recollect] after round {k + 1}: "
                                 f"examples per node {[len(f) for f in fitsets]}")

                if k == 0 and float(rec.get("x1_norm", 0.0)) == 0.0:
                    raise SystemExit("x_1 is still exactly zero after round 1: the "
                                     "optimizer took no effective step (empty fit set?).")
                if progress:
                    progress(f"round {k + 1}/{self.k_outer} "
                             f"primal={rec['primal_residual']:.4g} "
                             f"|x1|={rec['x1_norm']:.4g} "
                             f"f={[a['f'] for a in per_agent]} "
                             f"({time.time() - t0:.0f}s)")

                if dev_examples and dev_every and (k + 1) % dev_every == 0:
                    self.fast_state.load(self.admm.z[0])
                    d = self.dev_nll(dev_examples)
                    self.fast_state.disable()
                    d["round"] = k + 1
                    self.dev_curve.append(d)
                    cur = d.get("dev_nll_trained")
                    if cur is not None and (self.best_dev is None or cur < self.best_dev[0]):
                        self.best_dev = (cur, k + 1,
                                         [t.detach().cpu().clone() for t in self.admm.z[0]])
                    if progress:
                        progress(f"  dev@{k + 1}: trained={d.get('dev_nll_trained')} "
                                 f"delta={d.get('dev_nll_delta')} "
                                 f"best={None if self.best_dev is None else self.best_dev[1]}")

        tel = self._telemetry(rounds, time.time() - t0)
        if self.dev_curve:
            tel["dev_curve"] = self.dev_curve
            tel["best_dev_round"] = self.best_dev[1] if self.best_dev else None
        return tel

    # ------------------------------------------------------------ telemetry

    def _telemetry(self, rounds: List[Dict], elapsed: float) -> Dict:
        state_norms = [float(sum((t ** 2).sum() for t in self.admm.x[i]) ** 0.5)
                       for i in range(self.n_agents)]
        verify = (self.fast_state.verify_delta_w_consensus(self.admm.x, self.admm.z[0])
                  if self.n_agents > 1 else None)
        numel = self.admm.numel()
        return {
            "rounds": rounds,
            "k_outer": self.k_outer,
            "local_steps": self.local_steps,
            "batch_size": self.batch_size,
            "n_nodes": self.n_agents,
            "adapted_agents": self.adapt_idx,
            "grad_checkpointing": self.grad_checkpointing,
            "peak_mem_gb": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                            if torch.cuda.is_available() else None),
            "rho": self.rho,
            "consensus": self.consensus_name,
            "backend": self.backend.describe(),
            "state_norms": [round(s, 8) for s in state_norms],
            "state_norm_nonzero": bool(min(state_norms) > 0.0),
            # ||Delta W|| / ||W|| over the adapted modules at the consensus point.
            "delta_w_rel": self._delta_w_rel(),
            "kl_beta": self.kl_beta,
            "kl_ref": self.kl_ref,
            "kl_mean": (self._kl_sum / self._kl_n) if self._kl_n else None,
            "role_balance": self.role_balance,
            "nonfinite_grads": self.admm.nonfinite_grads,
            "skipped_long_targets": self._skipped_long,
            "truncated_prompts": self._skipped_long_prompt,
            "target_span": self.target_span,
            "skipped_no_think_close": self._skipped_no_close,
            "fit_prompt": self.fit_prompt,
            "fit_prompt_rerendered": self._fit_rerendered,
            "fit_prompt_rerendered_changed": self._fit_rerendered_changed,
            "empty_batches": self._empty_batches,
            "residuals": self.admm.residuals(),
            "delta_w_consensus_check": verify,
            "fast_state": self.fast_state.describe(),
            "comm_floats_per_round": numel,
            "comm_bytes_per_round_per_node": self.backend.bytes_per_round(numel, 4),
            "comm_bytes_total": self.backend.bytes_per_round(numel, 4) * self.k_outer,
            "train_sec": round(elapsed, 1),
        }

    def _delta_w_rel(self) -> float:
        """||Delta W||_F / ||W||_F at z (also leaves z loaded in the adapter bank)."""
        try:
            self.fast_state.load(self.admm.z[0])
            num = den = 0.0
            for a in self.fast_state.adapters:
                num += float((a.effective_delta_w() ** 2).sum())
                den += float((a.base.weight.detach().float() ** 2).sum())
            return round((num ** 0.5) / (den ** 0.5 + 1e-12), 9)
        except Exception:
            return -1.0

    # ---------------------------------------------------------- checkpoints

    def state_dict(self) -> Dict:
        """Per-node local variables x_i, the consensus variable z, and the seat map."""
        return {
            "x": [[t.detach().cpu() for t in st] for st in self.admm.x],
            "z": [t.detach().cpu() for t in self.admm.z[0]],
            "adapt_idx": list(self.adapt_idx),
        }

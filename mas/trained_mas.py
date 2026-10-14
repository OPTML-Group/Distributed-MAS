"""Trained MAS: the same message-passing system as `text_mas.py`, with each
adapted agent running its trained LoRA adapter.

The checkpoint written by `train.py` holds, per training node, the adapter
state and the seat (agent index) it drives:

    ckpt["state"]["x"]    per-node local variables x_i      (--deploy local)
    ckpt["state"]["z"]    consensus variable z               (--deploy consensus)
    ckpt["meta"]["adapt_idx"]   agent index driven by each node

`<out>-last.pt` / `<out>-best.pt` store the consensus variable z broadcast to
every adapted agent, which is what Algorithm A1 returns. A centralized (pooled)
checkpoint has a single state, which is deployed on every adapted agent.

A compromised agent (`--agent_models`) runs its own backbone. Because every
agent is trained (the defender does not know which one is compromised), an
adapter bank with the identical frozen basis A is injected into that backbone
too, so the compromised agent applies its own trained B.
`--skip_compromised_adapter` runs the compromised agent without its adapter.
"""
import argparse
import time
from typing import Dict, List, Optional

import torch

from consensus import max_abs_deviation_from_mean
from evaluation import scoring
from mas.models import ModelWrapper
from training.lora import LoraFastState
from training.rollouts import anchor_key

from . import agent_label, build_agents, resolve_prompt_builder


def ckpt_lora_config(meta: Dict) -> Dict:
    """LoRA geometry recorded in a training checkpoint."""
    cfg = meta.get("config", {}) or {}
    return {"rank": int(cfg.get("lora_rank", 16)),
            "alpha": float(cfg.get("lora_alpha", 16.0)),
            "layers_last": int(cfg.get("lora_layers_last", 36)),
            "targets": cfg.get("lora_targets", "attn_mlp")}


def _states(state: Dict):
    """(per-node states, consensus state), accepting the original key names too."""
    per_node = state.get("x", state.get("D"))
    z = state.get("z", state.get("D_bar"))
    return per_node, z


class DistTrainedTextMAS:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        args: argparse.Namespace = None,
        role_models: Optional[Dict[str, ModelWrapper]] = None,
        ckpt: Optional[Dict] = None,
    ) -> None:
        self.model = model
        self.role_models: Dict[str, ModelWrapper] = dict(role_models or {})
        self.args = args
        self.task = args.task
        self.max_new_tokens_each = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.method_name = "text_mas"          # mas/prompts.py asserts on this
        self.prompt_mode = args.prompt
        self.build_messages = resolve_prompt_builder(self.prompt_mode)
        self.agents = build_agents(int(getattr(args, "n_agents", 4)))

        path = getattr(args, "ckpt", None)
        if not path:
            raise SystemExit("--ckpt is required for --method mas_trained")
        if ckpt is None:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.adapt_compromised = not bool(getattr(args, "skip_compromised_adapter", False))
        self.lora_cfg = ckpt_lora_config(ckpt.get("meta", {}))
        self.fast_state = LoraFastState(model, **self.lora_cfg)
        self._load_checkpoint(path, ckpt)
        self.seen_anchor_hits = 0

    # ---------------------------------------------------------- checkpoint

    def _load_checkpoint(self, path: str, ckpt: Dict) -> None:
        meta = ckpt.get("meta", {})
        cfg = meta.get("config", {})
        per_node, z = _states(ckpt["state"])

        # The deployed basis A must be the one B was fitted against.
        trained_fp = (((meta.get("training") or {}).get("fast_state") or {})
                      .get("a_fingerprint"))
        eval_fp = self.fast_state.a_fingerprint()
        if trained_fp and eval_fp != trained_fp:
            raise SystemExit(
                f"[mas_trained] LoRA basis A mismatch: trained with {trained_fp}, this "
                f"evaluation built {eval_fp}. Use the training seed for the basis "
                f"(--lora_basis_seed) and do not consume the CPU RNG before the "
                f"adapter bank is built.")
        print(f"[mas_trained] A basis {eval_fp} "
              f"({'matches training' if trained_fp else 'training fingerprint not recorded'})",
              flush=True)

        deploy = getattr(self.args, "deploy", "local")
        req = meta.get("deploy_required")
        if req and deploy != req:
            raise SystemExit(
                f"{path} was trained with deploy_required={req!r}; its consensus "
                f"variable is not meaningful for this run. Re-run with --deploy {req}.")
        if deploy == "consensus":
            per_node = [z] * len(per_node)

        # One adapter bank per distinct backbone; extra banks copy A exactly.
        self._banks = {id(self.model): self.fast_state}
        for _role, mw in self.role_models.items():
            if id(mw) not in self._banks:
                self._banks[id(mw)] = LoraFastState(mw, **self.lora_cfg)
        src = self.fast_state
        for bank in self._banks.values():
            if bank is src:
                continue
            with torch.no_grad():
                for a_dst, a_src in zip(bank.adapters, src.adapters):
                    a_dst.A.copy_(a_src.A.to(a_dst.A.device, a_dst.A.dtype))
            if bank.a_fingerprint() != src.a_fingerprint():
                raise SystemExit("adapter bank A mismatch after copy: the compromised "
                                 "model's adapter layout differs from the backbone's.")

        template = self.fast_state.template()
        for k, st in enumerate(per_node):
            if len(st) != len(template) or any(
                    tuple(a.shape) != tuple(b.shape) for a, b in zip(st, template)):
                raise SystemExit(f"checkpoint node {k} does not match the adapter "
                                 f"geometry {self.lora_cfg}.")

        adapt_idx = (meta.get("adapt_idx") or ckpt["state"].get("adapt_idx")
                     or list(range(len(per_node))))
        if len(per_node) > 1 and len(adapt_idx) != len(per_node):
            raise SystemExit(f"checkpoint has {len(per_node)} node states but its seat "
                             f"map has {len(adapt_idx)} entries ({adapt_idx}).")
        if len(per_node) == 1 and len(adapt_idx) > 1:
            # Centralized training: one variable, deployed on every adapted agent.
            per_node = [per_node[0] for _ in adapt_idx]
        dev = self.model.device
        self.node_state = [[t.to(dev, torch.float32) for t in st] for st in per_node]
        self.node_of: Dict[int, int] = {a: k for k, a in enumerate(adapt_idx)}
        self.adapt_idx = adapt_idx
        print(f"[mas_trained] adapted agents {adapt_idx} = "
              f"{[self.agents[i].role for i in adapt_idx]}", flush=True)

        self.anchor_keys = set(meta.get("anchor_keys", []))
        self.ckpt_meta = {
            "path": path,
            "adapted_agents": adapt_idx,
            "adapted_roles": meta.get("adapted_roles"),
            "deploy": deploy,
            "adapt_compromised": self.adapt_compromised,
            "n_nodes": len(per_node),
            "consensus": cfg.get("consensus"),
            "topology": cfg.get("topology"),
            "pooled": cfg.get("pooled"),
            "fit_mode": cfg.get("fit_mode"),
            "k_outer": cfg.get("k_outer"),
            "rho": cfg.get("rho"),
            "state_spread": (max_abs_deviation_from_mean(self.node_state)
                             if len(self.node_state) > 1 else 0.0),
            "state_norms": [float(sum((t ** 2).sum() for t in st) ** 0.5)
                            for st in self.node_state],
        }

    # ------------------------------------------------------------- plumbing

    def _m(self, role: str) -> ModelWrapper:
        return self.role_models.get(role, self.model)

    def role_model_names(self) -> dict:
        return {a.role: self._m(a.role).model_name for a in self.agents}

    def _enter(self, agent_idx: int) -> None:
        """Enable exactly the bank of the model this agent runs, loaded with its state."""
        for b in self._banks.values():
            b.disable()
        node = self.node_of.get(agent_idx)
        if node is None:
            return
        bank = self._banks.get(id(self._m(self.agents[agent_idx].role)), self.fast_state)
        bank.load(self.node_state[node])
        bank.enable()

    # ---------------------------------------------------------------- main

    def run_item(self, item: Dict) -> Dict:
        t0 = time.time()
        question = item["question"]
        gold = item.get("gold", "")
        if anchor_key(item) in self.anchor_keys:
            self.seen_anchor_hits += 1

        context = ""
        traces: List[Dict] = []
        final_text = ""

        for idx, agent in enumerate(self.agents):
            messages = self.build_messages(role=agent.role, question=question,
                                           context=context, method=self.method_name,
                                           args=self.args)
            m = self._m(agent.role)
            prompts, ids, mask, tokens = m.prepare_chat_batch(
                [messages], add_generation_prompt=True)
            if m is self.model or self.adapt_compromised:
                self._enter(idx)
            try:
                with torch.no_grad():
                    gen, _ = m.generate_text_batch(
                        ids, mask, max_new_tokens=self.max_new_tokens_each,
                        temperature=self.temperature, top_p=self.top_p)
            finally:
                for b in self._banks.values():
                    b.disable()
            text = gen[0].strip()

            if agent.role == "judger":
                final_text = text
            else:
                context += f"[{agent_label(agent, self.prompt_mode)}]:\n{text}\n\n"
            traces.append({
                "name": agent.name, "role": agent.role,
                "adapted": idx in self.node_of and (m is self.model or self.adapt_compromised),
                "model": m.model_name,
                "input": prompts[0], "input_tokens": tokens[0], "output": text,
            })

        pred, r = scoring.score(final_text, gold, item.get("answer_type", ""))
        return {
            "question": question,
            "gold": gold,
            "solution": item.get("solution", ""),
            "context": context,
            "prediction": pred,
            "raw_prediction": final_text,
            "agents": traces,
            "correct": bool(r >= 1.0),
            "training": {
                "fast_state": self.fast_state.describe(),
                "checkpoint": self.ckpt_meta,
                # rows of this evaluation that were training anchors; must be 0
                "anchor_leakage": self.seen_anchor_hits,
                "latency_sec": round(time.time() - t0, 3),
            },
        }


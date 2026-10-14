#!/usr/bin/env python3
"""Turn MAS trajectories collected by run.py into a training fit set.

The safety runs collect rollouts of the (possibly compromised) MAS with run.py,
G times per query with different seeds, and convert them here. Every adapted
agent's message in every rollout becomes one example (fit_mode "sft") that
records the agent's rendered prompt, its message, the rollout's final answer,
the query's gold label and whether the final answer was correct. Rewards are
attached afterwards by tools/judge_fitset.py; train.py then derives DPO pairs
per (query, agent) from them.

    python tools/build_fitset.py --src results/collect-*.json \
        --dst ckpt/fitset-smas.pt --n_agents 4 --adapt_roles all
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch                                                       # noqa: E402

from mas import Agent, agent_label                                 # noqa: E402
from training.rollouts import FitExample, anchor_key               # noqa: E402

ADAPT = {
    "all": lambda n: list(range(n)),
    "nonsink": lambda n: list(range(n - 1)),
    "middle": lambda n: list(range(1, n - 1)),
}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="run.py result files (globs ok)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--adapt_roles", default="all", choices=sorted(ADAPT))
    ap.add_argument("--n_agents", type=int, default=4)
    a = ap.parse_args()

    # Keep the order given on the command line (it fixes the example order).
    srcs = []
    for pat in a.src:
        for f in (sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]):
            if f not in srcs:
                srcs.append(f)
    if not srcs:
        raise SystemExit("no source files matched")
    adapt_idx = ADAPT[a.adapt_roles](a.n_agents)
    node_of = {ai: k for k, ai in enumerate(adapt_idx)}
    fitsets = [[] for _ in adapt_idx]
    anchor_keys, prompts_seen, models_seen, n_rows, skipped = [], set(), set(), 0, 0

    for src in srcs:
        d = json.load(open(src))
        s = d.get("summary", {})
        cfg = s.get("config") or {}
        prompt_mode = cfg.get("prompt")
        prompts_seen.add(prompt_mode)
        if cfg.get("agent_models"):
            models_seen.add(str(cfg["agent_models"]))
        for rec in d.get("predictions", []):
            agents = rec.get("agents") or []
            if len(agents) != a.n_agents:
                skipped += 1
                continue
            q = rec.get("question", "")
            ak = anchor_key({"question": q})
            n_rows += 1
            anchor_keys.append(ak)
            # Upstream transcript exactly as the MAS shows it to agent i.
            ctx_parts = [
                f"[{agent_label(Agent(g.get('name'), g.get('role')), prompt_mode)}]:\n"
                f"{g.get('output', '')}\n\n"
                for g in agents if g.get("role") != "judger"]
            for ai in adapt_idx:
                ag = agents[ai]
                fitsets[node_of[ai]].append(FitExample(
                    agent_idx=ai,
                    node=node_of[ai],
                    prompt=ag.get("input", ""),
                    output=ag.get("output", ""),
                    advantage=1.0,
                    anchor=ak,
                    role=ag.get("role"),
                    question=q,
                    context="".join(ctx_parts[:ai]),
                    reward=0.0,
                    final_text=(rec.get("raw_prediction") or rec.get("prediction") or ""),
                    gold=str(rec.get("gold") or ""),
                    correct=(None if rec.get("correct") is None else bool(rec.get("correct"))),
                ))

    key = {
        "arm": "mas", "fit_mode": "sft", "n_agents": a.n_agents,
        "adapt_roles": a.adapt_roles,
        "teacher_prompt": sorted(x for x in prompts_seen if x),
        "agent_models": sorted(models_seen),
        "bridged_from": [{"path": p, "sha256_12": sha_file(p)} for p in srcs],
        # Examples come from the actual trajectories (agent i's message is
        # conditioned on that trajectory's upstream messages).
        "sampling_structure": "actual_trajectory",
    }
    print(f"rows={n_rows} skipped={skipped} adapt_idx={adapt_idx} "
          f"fit sizes={[len(f) for f in fitsets]}")
    print(f"collection prompt(s)={key['teacher_prompt']} agent_models={key['agent_models']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)) or ".", exist_ok=True)
    torch.save({"fitsets": fitsets, "stats": {"bridged": True, "n_rows": n_rows},
                "anchor_keys": anchor_keys, "adapt_idx": adapt_idx, "key": key}, a.dst)
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()

"""Evaluation (and trajectory collection) entry point.

    --method baseline      single LLM
    --method text_mas      untrained MAS ("Base")
    --method mas_trained   MAS with adapters trained by train.py (--ckpt)

A compromised agent is placed with `--agent_models <role>=<model>`, e.g.
`solver=mlabonne/Qwen3-8B-abliterated` (S-MAS) or
`synthesizer=mlabonne/Qwen3-8B-abliterated` (H-MAS); roles can be given by
their paper names or by the internal keys planner/critic/refiner/judger.

Evaluation protocol of the paper (App. C.4): thinking mode (`--think`),
temperature 0.6, top-p 0.95, seed 42; 4096 tokens per agent for safety and
8192 for reasoning. Rows are shuffled with the seed and sliced with
`--start/--max_samples`, so shards of one evaluation are disjoint and can be
merged with tools/merge_shards.py. Safety benchmarks use their full official
split (`--reserve_tail 0`).

Writes results/<run_id>.json with the configuration, per-row predictions
(including every agent's message) and a summary. Safety rows are scored here
only with a substring refusal detector; reported ASR comes from
`tools/judge.py --judge harmbench` applied to this file.
"""
import argparse
import json
import os
import random
import time
from typing import Dict, List

import torch
from tqdm import tqdm

from evaluation import data as data_mod
from mas import canonical_role, prompt_modes
from mas.single import BaselineMethod
from mas.text_mas import TextMASMethod
from mas.models import ModelWrapper, auto_device, set_seed

METHODS = ("baseline", "text_mas", "mas_trained")

TASKS = {
    # safety (Fig. 2, Table 1)
    "harmbench": ("load_harmbench", {"category": "standard", "split": "test"}),
    "strongreject": ("load_strongreject", {}),
    "jbb_harmful": ("load_jbb_harmful", {}),
    "orbench_toxic": ("load_orbench_toxic", {}),
    "xstest": ("load_xstest", {}),
    # safety training pool; only used to collect training trajectories
    "safety_train": ("load_safety_train", {}),
    # reasoning (Fig. 4)
    "mmlu_pro": ("load_mmlu_pro", {}),
    "gsm8k": ("load_gsm8k", {"split": "test"}),
    "math500": ("load_math500", {}),
    "bbh": ("load_bbh", {}),
    "gpqa": ("load_gpqa_diamond", {"split": "test"}),
    "musr": ("load_musr", {}),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-8B")
    p.add_argument("--agent_models", default=None,
                   help="comma-separated role=model pairs for agents that use a "
                        "different backbone (the compromised agent), e.g. "
                        "solver=mlabonne/Qwen3-8B-abliterated")
    p.add_argument("--prompt", choices=prompt_modes(), default="sequential_v2",
                   help="sequential_v2 = S-MAS, hierarchical = H-MAS")
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--think", action="store_true", help="Qwen3 thinking mode (used throughout the paper)")
    p.add_argument("--max_new_tokens", type=int, default=4096, help="generation budget per agent")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--start", type=int, default=0, help="skip the first N rows (sharding)")
    p.add_argument("--reserve_tail", type=int, default=32,
                   help="drop the last N rows of the shuffled pool before slicing; "
                        "train.py draws reasoning anchors from that tail")
    p.add_argument("--no_shuffle", action="store_true", help="keep dataset order")
    p.add_argument("--ckpt", default=None, help="checkpoint from train.py (mas_trained)")
    p.add_argument("--deploy", choices=["local", "consensus"], default="local",
                   help="local: agent i uses node i's x_i; consensus: every adapted "
                        "agent uses z. Identical for the -last/-best checkpoints.")
    p.add_argument("--skip_compromised_adapter", action="store_true",
                   help="run agents replaced via --agent_models WITHOUT their trained "
                        "adapter (by default every trained adapter is applied)")
    p.add_argument("--lora_basis_seed", type=int, default=None,
                   help="seed that rebuilds the frozen LoRA basis A; default: the "
                        "training seed recorded in --ckpt, else --seed")
    p.add_argument("--text_mas_context_length", type=int, default=None,
                   help="truncate the upstream transcript to this many characters")
    p.add_argument("--device", default="cuda")
    p.add_argument("--run_id", default=None)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--quiet", action="store_true")
    return p


def load_task(name: str) -> List[Dict]:
    fname, kw = TASKS[name]
    return list(getattr(data_mod, fname)(**kw))


def parse_agent_models(spec: str) -> Dict[str, str]:
    out = {}
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"--agent_models entry '{pair}' is not role=model")
        role, name = pair.split("=", 1)
        out[canonical_role(role)] = name.strip()
    return out


def generation_health(preds: List[Dict]) -> Dict:
    """Parse rate and truncation of the final answer, and per-agent message stats."""
    n = len(preds)
    if not n:
        return {}
    raws = [(p.get("raw_prediction") or "") for p in preds]
    parsed = sum(1 for r in raws if "\\boxed{" in r)
    unclosed = sum(1 for r in raws if "<think>" in r and "</think>" not in r)
    out = {
        "parse_rate": round(parsed / n, 4),
        "sink_unclosed_think_rate": round(unclosed / n, 4),
        "sink_mean_output_chars": round(sum(len(r) for r in raws) / n, 1),
    }
    per_role: Dict[str, List] = {}
    for p in preds:
        for a in (p.get("agents") or []):
            per_role.setdefault(a.get("name", "?"), []).append(a.get("output") or "")
    if len(per_role) > 1:
        out["agent_unclosed_think"] = {
            name: round(sum(1 for o in outs if "<think>" in o and "</think>" not in o)
                        / len(outs), 4)
            for name, outs in per_role.items()}
        out["agent_mean_chars"] = {
            name: round(sum(len(o) for o in outs) / len(outs), 1)
            for name, outs in per_role.items()}
    return out


def stack_fingerprint(args) -> Dict:
    import platform
    fp = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "tokenizers", "datasets", "numpy"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            fp[mod] = None
    fp["cuda"] = torch.version.cuda
    fp["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    fp["model"] = str(args.model_name)
    return fp


def main() -> None:
    args = build_parser().parse_args()
    if args.text_mas_context_length is not None and args.text_mas_context_length <= 0:
        args.text_mas_context_length = None

    ckpt = None
    if args.method == "mas_trained":
        if not args.ckpt:
            raise SystemExit("--method mas_trained needs --ckpt")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = (ckpt.get("meta") or {}).get("config") or {}
        if args.lora_basis_seed is None:
            s = cfg.get("lora_basis_seed")
            args.lora_basis_seed = s if isinstance(s, int) else cfg.get("seed")

    # The frozen LoRA basis A is drawn from the global CPU RNG when the adapter
    # bank is built. Training does set_seed -> load backbone -> build bank, so the
    # same prefix is reproduced here, and the CPU RNG is snapshotted after the
    # backbone load and restored after the extra (compromised) model loads.
    basis_seed = args.lora_basis_seed if args.lora_basis_seed is not None else args.seed
    set_seed(basis_seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, args=args)
    rng_after_backbone = torch.get_rng_state()
    if basis_seed != args.seed:
        set_seed(args.seed)

    role_models = {}
    if args.agent_models:
        if args.method == "baseline":
            raise SystemExit("--agent_models only applies to the MAS methods")
        by_name = {args.model_name: model}
        for role, name in parse_agent_models(args.agent_models).items():
            if name not in by_name:
                print(f"[agent_models] loading {name} for role '{role}'", flush=True)
                by_name[name] = ModelWrapper(name, device, args=args)
            role_models[role] = by_name[name]
    torch.set_rng_state(rng_after_backbone)

    common = dict(temperature=args.temperature, top_p=args.top_p, args=args)
    if args.method == "baseline":
        method = BaselineMethod(model, max_new_tokens=args.max_new_tokens, **common)
    elif args.method == "text_mas":
        method = TextMASMethod(model, role_models=role_models,
                               max_new_tokens_each=args.max_new_tokens, **common)
    else:
        from mas.trained_mas import DistTrainedTextMAS
        method = DistTrainedTextMAS(model, role_models=role_models, ckpt=ckpt,
                                    max_new_tokens_each=args.max_new_tokens, **common)

    items = load_task(args.task)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(items)
    reserved = 0
    if args.reserve_tail > 0 and len(items) > args.reserve_tail:
        reserved = args.reserve_tail
        items = items[:-reserved]
    if args.start:
        items = items[args.start:]
    if args.max_samples > 0:
        items = items[: args.max_samples]

    preds: List[Dict] = []
    per_item_sec: List[float] = []
    t0 = time.time()
    for item in tqdm(items, desc=f"{args.method}/{args.task}"):
        ti = time.time()
        out = method.run_item(item)
        per_item_sec.append(time.time() - ti)
        preds.append(out)
        if not args.quiet:
            print(f"[{len(preds)}/{len(items)}] {per_item_sec[-1]:.1f}s "
                  f"pred={out.get('prediction')!r} gold={out.get('gold')!r} "
                  f"ok={out.get('correct')}", flush=True)

    n = len(preds)
    correct = sum(1 for p in preds if p.get("correct"))
    run_id = args.run_id or f"{args.method}-{args.task}-{args.prompt}-N{args.n_agents}-s{args.seed}"
    summary = {
        "stack": stack_fingerprint(args),
        "run_id": run_id,
        "config": vars(args),
        "role_models": method.role_model_names(),
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "n": n,
        "total_time_sec": round(time.time() - t0, 2),
        "sec_per_sample_mean": round(sum(per_item_sec) / max(1, n), 2),
        "anchor_leakage": max((p.get("training", {}).get("anchor_leakage", 0) for p in preds),
                              default=0),
        "reserved_tail": reserved,
        **generation_health(preds),
    }
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, f"{run_id}.json")
    with open(path, "w") as fh:
        json.dump({"summary": summary, "predictions": preds}, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"},
                     ensure_ascii=False, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

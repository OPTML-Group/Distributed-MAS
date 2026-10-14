"""Distributed MAS training via consensus ADMM (Sec. 4, Algorithm A1).

Training data come from one of two sources:

  (a) a fit-set cache (`--fitset_cache`) built from MAS trajectories collected
      with run.py (`tools/build_fitset.py`) and labelled with the HarmBench
      reward (`tools/judge_fitset.py`). This is how the safety runs are trained:
      the compromised MAS is rolled out by run.py, which supports
      `--agent_models`.
  (b) on-policy collection by this script (`--reward outcome`), optionally
      repeated every `--recollect_every` rounds with the current consensus
      adapter on fresh queries. This is how the reasoning runs are trained.

Each training node holds the examples of one agent (`--partition seat`). The
three training arms of the paper differ only in how nodes are coupled:

    ADMM          --consensus ring                 (default; or gossip, Fig. 5)
    Centralized   --pooled --consensus coordinator (all examples on one node)
    ADMM-Local    --local_only                     (Table 2)
    ADMM-NoDual   --no_dual                        (Table 2)

Defaults are the safety settings of Table A2 (rho 0.03, L = 16 local Adam steps,
lr 1e-3, clip 0.01, batch 8, beta_DPO = beta_KL = 0.1, rank-16 LoRA on all 7
projections of all 36 layers, B only).

Outputs
    <out>            per-node states x_i, consensus z and the seat map
    <out>-last.pt    the final consensus variable z on every adapted agent
                     (what Algorithm A1 returns; evaluate this one)
    <out>-best.pt    z at the round with the lowest held-out NLL (if --dev_frac > 0)
    <report_dir>/<name>-train.json   configuration and per-round telemetry
                     (local objectives, primal residual, communication)
"""
import argparse
import hashlib
import json
import os
import random
import re
import time

import torch

from evaluation import data as data_mod
from evaluation import scoring as S
from mas import prompt_modes
from mas.models import ModelWrapper, auto_device, set_seed
from training.lora import LoraFastState
from training.rollouts import (ChainRoller, anchor_key, collect_fitsets,
                               drop_unclosed_think, sft_examples, sft_to_dpo)
from training.trainer import ADMMTrainer


def file_fingerprint(path) -> str:
    if not path:
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _ckpt_z(ckpt):
    st = ckpt["state"]
    return st.get("z", st.get("D_bar"))


def build_parser() -> argparse.ArgumentParser:
    from run import TASKS

    p = argparse.ArgumentParser()
    # system
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", choices=prompt_modes(), default="sequential_v2",
                   help="prompt the rollouts were collected under "
                        "(sequential_v2 = S-MAS, hierarchical = H-MAS)")
    p.add_argument("--fit_prompt", choices=prompt_modes(), default=None,
                   help="fit every message under this (deployment) prompt instead "
                        "of the collection prompt, keeping the collected message as "
                        "the target (used with the hardened *_v2s prompts)")
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--think", action="store_true")
    p.add_argument("--text_mas_context_length", type=int, default=None)
    p.add_argument("--adapt_roles", choices=["all", "nonsink", "middle"], default="all",
                   help="which agents carry a trainable adapter (paper: all)")

    # data / rollouts
    p.add_argument("--anchors", type=int, default=32, help="number of training queries")
    p.add_argument("--group", type=int, default=8, help="rollouts per query")
    p.add_argument("--credit", choices=["shared", "per_role"], default="shared")
    p.add_argument("--reward", choices=["judge", "outcome", "gold_likelihood"],
                   default="outcome",
                   help="judge: HarmBench reward stored in --fitset_cache (safety); "
                        "outcome: final-answer correctness (reasoning)")
    p.add_argument("--fit_mode", choices=["dpo", "sft", "rsft"], default="dpo",
                   help="role-local objective: dpo (Eq. 8), sft (every collected "
                        "message), rsft (only messages from successful rollouts)")
    p.add_argument("--anchor_split", default=None,
                   help="draw training queries from this split (e.g. train) instead "
                        "of the evaluation split")
    p.add_argument("--anchor_offset", type=int, default=None,
                   help="take queries from rows [K, K+n) of the shuffled pool "
                        "instead of its tail")
    p.add_argument("--anchor_pool", type=int, default=None,
                   help="pre-screen: sample each of the last M candidates twice and "
                        "drop those answered correctly both times")
    p.add_argument("--anchor_pass_band", default=None, metavar="LO,HI",
                   help="keep only queries whose pass rate lies strictly inside "
                        "(LO,HI); 0,1 keeps groups with mixed outcomes")
    p.add_argument("--recollect_every", type=int, default=0,
                   help="re-collect on fresh queries with the current consensus "
                        "adapter every N rounds (0 = collect once)")
    p.add_argument("--downstream_seed", type=int, default=1234)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--sink_tokens", type=int, default=None,
                   help="final agent's budget in training rollouts (default: "
                        "--max_new_tokens; 0 = do not generate)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--fitset_cache", default=None,
                   help="load the fit sets from this file if it exists, otherwise "
                        "collect and save them there")
    p.add_argument("--init_state", default=None,
                   help="warm start: x^(0) = z^(0) = this checkpoint's consensus variable")

    # objective
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--kl_beta", type=float, default=0.1)
    p.add_argument("--kl_ref", choices=["frozen", "collect"], default="frozen",
                   help="KL reference: the initial policy with the adapter disabled "
                        "(frozen, Eq. 8) or the data-collection policy (the "
                        "--init_state adapter if given, else identical to frozen)")
    p.add_argument("--no_role_balance", action="store_true",
                   help="sample minibatches uniformly instead of balanced across roles")
    p.add_argument("--target_span", choices=["full", "visible"], default="full",
                   help="fit the whole message including <think> (full) or only "
                        "the text after </think> (visible)")
    p.add_argument("--fit_filter", choices=["none", "drop_unclosed_think"],
                   default="drop_unclosed_think")
    p.add_argument("--dedup_targets", type=int, default=3,
                   help="at most this many examples per node may share one target "
                        "text (0 = off)")
    p.add_argument("--max_target_tokens", type=int, default=4096)
    p.add_argument("--max_prompt_tokens", type=int, default=8192)

    # adapter
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_layers_last", type=int, default=36)
    p.add_argument("--lora_targets", default="attn_mlp",
                   choices=["attn_qv", "attn_all", "attn_mlp"])
    p.add_argument("--lora_basis_seed", type=int, default=None,
                   help="seed of the frozen basis A (default: --seed)")

    # ADMM
    p.add_argument("--k_outer", type=int, default=20, help="ADMM rounds T")
    p.add_argument("--local_steps", type=int, default=16, help="local steps L per x-step")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eta", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--rho", type=float, default=0.03, help="ADMM penalty (lambda = 1/rho)")
    p.add_argument("--grad_clip", type=float, default=0.01)
    p.add_argument("--consensus", choices=["coordinator", "ring", "gossip"], default="ring")
    p.add_argument("--topology", default="ring",
                   choices=["ring", "complete", "expander", "grid", "chain"])
    p.add_argument("--gossip_steps", type=int, default=1, help="gossip rounds tau")
    p.add_argument("--partition", choices=["seat", "anchor"], default="seat",
                   help="seat: one node per agent role (the paper's setting); "
                        "anchor: split the pooled examples across nodes by query")
    p.add_argument("--n_partitions", type=int, default=0)
    p.add_argument("--pooled", action="store_true", help="centralized training")
    p.add_argument("--local_only", action="store_true", help="ADMM-Local ablation")
    p.add_argument("--no_dual", action="store_true", help="ADMM-NoDual ablation")
    p.add_argument("--no_grad_checkpointing", action="store_true")

    # held-out selection
    p.add_argument("--dev_frac", type=float, default=0.0,
                   help="hold out this fraction of each node's examples to track "
                        "held-out NLL and write -best.pt (0 = off; train on all)")
    p.add_argument("--dev_every", type=int, default=1)

    p.add_argument("--out", required=True, help="checkpoint path (.pt)")
    p.add_argument("--report_dir", default="results")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="build the fit sets and stop before any gradient step")
    return p


def _pass_band(args):
    s = args.anchor_pass_band
    if not s:
        return None
    lo, hi = (float(x) for x in str(s).split(","))
    if not (0.0 <= lo < hi <= 1.0):
        raise SystemExit(f"--anchor_pass_band {lo},{hi} is not 0 <= LO < HI <= 1")
    return (lo, hi)


COLLECT_KEY = ("task", "prompt", "think", "n_agents", "adapt_roles", "credit", "reward",
               "fit_mode", "anchors", "group", "seed", "downstream_seed", "sink_tokens",
               "max_new_tokens", "temperature", "top_p", "anchor_split",
               "anchor_pass_band", "anchor_pool", "anchor_offset")


def main() -> None:
    from run import TASKS

    args = build_parser().parse_args()
    if args.text_mas_context_length is not None and args.text_mas_context_length <= 0:
        args.text_mas_context_length = None
    args.method = "text_mas"                      # mas/prompts.py asserts on this
    log = (lambda m: None) if args.quiet else (lambda m: print(m, flush=True))

    # ---- training queries: a seed-deterministic shuffle of the task pool
    fname, kw = TASKS[args.task]
    if args.anchor_split:
        kw = dict(kw, split=args.anchor_split)
    items = list(getattr(data_mod, fname)(**kw))
    random.Random(args.seed).shuffle(items)
    if args.anchors >= len(items):
        raise SystemExit(f"--anchors {args.anchors} >= pool size {len(items)}")
    pool_n = max(args.anchor_pool or 0, args.anchors)
    if args.anchor_offset is not None:
        off = args.anchor_offset
        if off + pool_n > len(items):
            raise SystemExit(f"--anchor_offset {off} + pool {pool_n} exceeds {len(items)} rows")
        anchors = items[off: off + pool_n]
    else:
        anchors = items[-pool_n:]
    anchor_keys = [anchor_key(it) for it in anchors]

    # ---- backbone and adapters. The frozen basis A is drawn from the CPU RNG
    # right after the backbone load (run.py reproduces the same prefix).
    basis_seed = args.lora_basis_seed if args.lora_basis_seed is not None else args.seed
    set_seed(basis_seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, args=args)
    rng_after_backbone = torch.get_rng_state()
    if basis_seed != args.seed:
        set_seed(args.seed)
        torch.set_rng_state(rng_after_backbone)
    fast_state = LoraFastState(model, rank=args.lora_rank, alpha=args.lora_alpha,
                               layers_last=args.lora_layers_last, targets=args.lora_targets)

    init_D = None
    if args.init_state:
        init_D = _ckpt_z(torch.load(args.init_state, map_location="cpu", weights_only=False))
        fast_state.load(init_D)
        log(f"warm start from {args.init_state}")
    roller = ChainRoller(model, args, n_agents=args.n_agents, fast_state=fast_state,
                         adapt_roles=args.adapt_roles)
    adapt_idx = roller.adapt_idx
    if args.reward == "outcome" and roller.sink_tokens <= 0:
        raise SystemExit("--reward outcome needs the final agent to generate (--sink_tokens > 0)")
    collect_states = [init_D] * len(adapt_idx) if init_D is not None else None

    # ---- pre-screen (reasoning): drop candidates answered correctly twice
    prescreen = None
    if args.anchor_pool and args.anchor_pool > args.anchors:
        pre_cache = (args.fitset_cache or f"ckpt/prescreen-{args.task}") + ".prescreen.json"
        probe = json.load(open(pre_cache)) if os.path.exists(pre_cache) else {}
        for a_idx, item in enumerate(anchors):
            k = anchor_keys[a_idx]
            if k in probe:
                continue
            if item.get("answer_type") == "safety":
                raise SystemExit("--anchor_pool pre-screening is for reasoning tasks")
            base = args.seed * 1000 + 777000 + a_idx * 31
            fails = 0
            for g in range(2):
                r = roller.chain(item, seed=base + g * 13, states=collect_states)
                _pred, ok = S.score(r.final_text or "", item.get("gold", ""),
                                    item.get("answer_type", ""))
                fails += 0 if ok else 1
            probe[k] = fails
            if (a_idx + 1) % 8 == 0:
                os.makedirs(os.path.dirname(os.path.abspath(pre_cache)) or ".", exist_ok=True)
                json.dump(probe, open(pre_cache, "w"))
                log(f"[prescreen] {a_idx + 1}/{len(anchors)} probed")
        os.makedirs(os.path.dirname(os.path.abspath(pre_cache)) or ".", exist_ok=True)
        json.dump(probe, open(pre_cache, "w"))
        survivors = [(it, k) for it, k in zip(anchors, anchor_keys) if probe.get(k, 1) > 0]
        n_all_pass = len(anchors) - len(survivors)
        log(f"[prescreen] pool {len(anchors)} -> survivors {len(survivors)}")
        if len(survivors) < args.anchors:
            pad = [(it, k) for it, k in zip(anchors, anchor_keys)
                   if probe.get(k, 0) == 0][: args.anchors - len(survivors)]
            survivors += pad
        anchors = [it for it, _ in survivors[: args.anchors]]
        anchor_keys = [k for _, k in survivors[: args.anchors]]
        prescreen = {"pool": args.anchor_pool, "survivors": len(survivors),
                     "all_pass_discarded": n_all_pass}
    elif len(anchors) > args.anchors:
        anchors = anchors[-args.anchors:]
        anchor_keys = anchor_keys[-args.anchors:]

    # ---- fit sets: from a cache, or collected now
    cache_key = dict({k: getattr(args, k) for k in COLLECT_KEY},
                     init_sha=file_fingerprint(args.init_state))
    provenance, judge_fp = "collected", None
    if args.fitset_cache and os.path.exists(args.fitset_cache):
        cached = torch.load(args.fitset_cache, map_location="cpu", weights_only=False)
        ck = cached.get("key", {})
        if ck.get("sampling_structure") == "actual_trajectory":
            # built by tools/build_fitset.py from run.py trajectories
            mism = {k: (ck.get(k), getattr(args, k)) for k in ("n_agents", "adapt_roles")
                    if ck.get(k) != getattr(args, k)}
            if mism:
                raise SystemExit(f"{args.fitset_cache}: mismatch (cached, requested) {mism}")
            provenance = "trajectories"
            judge_fp = ck.get("judge_fingerprint")
            if args.reward == "judge" and not judge_fp:
                raise SystemExit(f"{args.fitset_cache} carries no judge labels; run "
                                 f"tools/judge_fitset.py on it first")
        else:
            diff = {k: (ck.get(k), v) for k, v in cache_key.items()
                    if k != "fit_mode" and ck.get(k) != v}
            if diff:
                raise SystemExit(f"--fitset_cache {args.fitset_cache} was built with "
                                 f"different settings (cached, requested): {diff}")
            provenance = "cache"
        fitsets, collect_stats = cached["fitsets"], cached.get("stats", {})
        cached_mode = ck.get("fit_mode", args.fit_mode)
        if cached_mode == "sft" and args.fit_mode == "dpo":
            fitsets = sft_to_dpo(fitsets)
            log(f"derived DPO pairs from the cached rollouts: {[len(f) for f in fitsets]}")
        elif cached_mode == "sft" and args.fit_mode in ("sft", "rsft"):
            fitsets = sft_examples(fitsets, successful_only=(args.fit_mode == "rsft"))
            log(f"{args.fit_mode} targets from the cached rollouts: {[len(f) for f in fitsets]}")
        elif cached_mode != args.fit_mode:
            raise SystemExit(f"cannot derive {args.fit_mode!r} examples from a "
                             f"{cached_mode!r} cache")
        log(f"reusing fit sets from {args.fitset_cache}: sizes={[len(f) for f in fitsets]}")
        if cached.get("anchor_keys"):
            anchor_keys = list(cached["anchor_keys"])
    else:
        log(f"collecting {len(anchors)} queries x {args.group} rollouts, adapting "
            f"{[roller.agents[i].role for i in adapt_idx]} ...")
        fitsets, collect_stats = collect_fitsets(
            roller, anchors, group_size=args.group, credit=args.credit,
            fit_mode=args.fit_mode, reward=args.reward, pass_band=_pass_band(args),
            model=model, seed=args.seed, downstream_seed=args.downstream_seed,
            progress=log, states=collect_states)
        if args.fitset_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.fitset_cache)) or ".", exist_ok=True)
            torch.save({"fitsets": fitsets, "stats": collect_stats,
                        "anchor_keys": anchor_keys, "adapt_idx": adapt_idx,
                        "key": cache_key}, args.fitset_cache)
            log(f"cached fit sets -> {args.fitset_cache}")

    if args.fit_filter == "drop_unclosed_think":
        fitsets = drop_unclosed_think(fitsets, log=log)

    def _post_think(t):
        return re.sub(r"^.*?</think>", "", t, flags=re.S).strip()

    target_stats = []
    for node, fset in enumerate(fitsets):
        counts = {}
        for e in fset:
            t = _post_think(e.output)
            counts[t] = counts.get(t, 0) + 1
        top_n = max(counts.values()) if counts else 0
        target_stats.append({"node": node, "n": len(fset), "distinct": len(counts),
                             "top_share": round(top_n / max(1, len(fset)), 4)})

    if args.dedup_targets > 0:
        capped = []
        for fset in fitsets:
            seen = {}
            # keep the highest-advantage occurrences of each target text
            order = sorted(range(len(fset)), key=lambda i: -getattr(fset[i], "advantage", 0.0))
            keep = []
            for i in order:
                t = _post_think(fset[i].output)
                if seen.get(t, 0) < args.dedup_targets:
                    seen[t] = seen.get(t, 0) + 1
                    keep.append(i)
            capped.append([fset[i] for i in sorted(keep)])
        log(f"dedup_targets={args.dedup_targets}: sizes "
            f"{[len(f) for f in fitsets]} -> {[len(f) for f in capped]}")
        fitsets = capped

    report = {
        "config": vars(args),
        "n_agents": args.n_agents,
        "n_nodes": len(adapt_idx),
        "adapt_idx": adapt_idx,
        "adapted_roles": [roller.agents[i].role for i in adapt_idx],
        "pooled": args.pooled,
        "anchor_keys": anchor_keys,
        "collection": collect_stats,
        "fitset_provenance": provenance,
        "fitset_judge_fingerprint": judge_fp,
        "target_composition": target_stats,
    }
    if prescreen is not None:
        report["prescreen"] = prescreen
    if args.dry_run:
        _write_report(args, report)
        return
    if not any(fitsets):
        _write_report(args, report)
        raise SystemExit("no training examples: refusing to train")

    # ---- held-out slice, taken before pooling so every node contributes
    dev_set = []
    if args.dev_frac > 0:
        rng = random.Random(args.seed ^ 0x5EED)
        kept = []
        for fset in fitsets:
            idx = list(range(len(fset)))
            rng.shuffle(idx)
            n_dev = int(round(len(fset) * args.dev_frac))
            dev_idx = set(idx[:n_dev])
            dev_set.extend(fset[i] for i in sorted(dev_idx))
            kept.append([fset[i] for i in range(len(fset)) if i not in dev_idx])
        log(f"dev_frac={args.dev_frac}: fit sizes {[len(f) for f in fitsets]} -> "
            f"{[len(f) for f in kept]}, held out {len(dev_set)}")
        fitsets = kept
        report["dev_held_out"] = len(dev_set)

    # ---- how examples are distributed over training nodes
    if args.pooled:
        merged = [ex for fs in fitsets for ex in fs]
        log(f"centralized: {[len(f) for f in fitsets]} -> one node with {len(merged)}")
        fitsets = [merged]
    elif args.partition == "anchor":
        n_part = args.n_partitions or len(adapt_idx)
        parts = [[] for _ in range(n_part)]
        for ex in (ex for fs in fitsets for ex in fs):
            j = int(hashlib.sha1(str(ex.anchor).encode()).hexdigest(), 16) % n_part
            ex.node = j
            parts[j].append(ex)
        thin = [i for i, f in enumerate(parts) if len(f) < 16]
        if thin:
            raise SystemExit(f"--partition anchor: nodes {thin} hold fewer than 16 examples")
        log(f"partition=anchor: {[len(f) for f in fitsets]} -> {[len(f) for f in parts]}")
        fitsets = parts
        report["partition"] = {"mode": "anchor", "n_nodes": n_part,
                               "sizes": [len(f) for f in parts]}
    n_nodes = len(fitsets)

    trainer = ADMMTrainer(
        model, fast_state, n_nodes, adapt_idx=adapt_idx,
        consensus=args.consensus, topology=args.topology, gossip_steps=args.gossip_steps,
        eta=args.eta, rho=args.rho, grad_clip=args.grad_clip, k_outer=args.k_outer,
        local_steps=args.local_steps, batch_size=args.batch_size,
        max_target_tokens=args.max_target_tokens, max_prompt_tokens=args.max_prompt_tokens,
        grad_checkpointing=not args.no_grad_checkpointing,
        dpo_beta=args.dpo_beta, kl_beta=args.kl_beta, kl_ref=args.kl_ref,
        role_balance=not args.no_role_balance, target_span=args.target_span,
        fit_prompt=(args.fit_prompt if args.fit_prompt and args.fit_prompt != args.prompt else None),
        fit_args=args, local_only=args.local_only, no_dual=args.no_dual, seed=args.seed)

    recollect_fn = None
    used_keys = list(anchor_keys)
    if args.recollect_every > 0:
        # Fresh queries come only from the training region: the whole training
        # split (--anchor_split) or the rows from --anchor_offset on. run.py
        # evaluates the front of the same shuffle, which must never be drawn.
        if args.anchor_split:
            train_pool = items
        elif args.anchor_offset is not None:
            train_pool = items[args.anchor_offset:]
        else:
            raise SystemExit("--recollect_every needs --anchor_split or --anchor_offset so "
                             "that re-collected queries stay disjoint from the evaluation rows")
        used = set(anchor_keys)
        n_done = {"n": 0}

        def recollect_fn(round_idx):
            """Algorithm A1, step 3: deploy z_i and roll it out on fresh queries."""
            n_done["n"] += 1
            fresh = [it for it in train_pool if anchor_key(it) not in used]
            if len(fresh) < args.anchors:
                raise SystemExit(f"only {len(fresh)} unused queries left for re-collection")
            new_anchors = fresh[: args.anchors]
            used.update(anchor_key(it) for it in new_anchors)
            used_keys.extend(anchor_key(it) for it in new_anchors)
            states = [trainer.admm.z[i] for i in range(n_nodes)]
            if n_nodes != len(adapt_idx):           # centralized / anchor partition
                states = [trainer.admm.z[0]] * len(adapt_idx)
            fs, st = collect_fitsets(
                roller, new_anchors, group_size=args.group, credit=args.credit,
                fit_mode=args.fit_mode, reward=args.reward, pass_band=_pass_band(args),
                model=model, seed=args.seed + 1000 * n_done["n"],
                downstream_seed=args.downstream_seed, progress=log, states=states)
            log(f"[recollect] after round {round_idx}: mixed_group_rate="
                f"{st.get('mixed_group_rate')}, pass_rate_mean={st.get('pass_rate_mean')}")
            if args.pooled:
                fs = [[ex for f in fs for ex in f]]
            return fs

    t0 = time.time()
    train_stats = trainer.fit(fitsets, progress=log, init_D=init_D,
                              dev_examples=dev_set or None, dev_every=args.dev_every,
                              recollect=recollect_fn, recollect_every=args.recollect_every)
    rounds = train_stats.get("rounds") or []
    report["anchor_keys"] = used_keys      # every training query, for run.py's leakage check
    report["training"] = train_stats
    report["loss_curve"] = [[a.get("f") for a in r.get("agents", [])] for r in rounds]
    report["normalized_primal_residual_curve"] = [r.get("normalized_primal_residual")
                                                  for r in rounds]
    report["x1_norm_curve"] = [r.get("x1_norm") for r in rounds]
    if dev_set:
        report["dev"] = trainer.dev_nll(dev_set)
    report["train_sec"] = round(time.time() - t0, 1)

    dwr = train_stats.get("delta_w_rel", 0.0)
    if not train_stats["state_norm_nonzero"] or not dwr or dwr < 1e-6:
        _write_report(args, report)
        raise SystemExit(f"refusing to write {args.out}: the adapter did not move "
                         f"(delta_w_rel={dwr:.3e})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    meta = {k: v for k, v in report.items() if k != "collection"}
    # Per-node states are not a consensus point under local-only or gossip.
    deploy_required = "local" if (args.local_only or args.consensus == "gossip") else None
    collection_summary = {k: v for k, v in (collect_stats or {}).items() if k != "groups"}
    torch.save({"state": trainer.state_dict(),
                "meta": dict(meta, deploy_required=deploy_required),
                "collection_summary": collection_summary}, args.out)
    stem = args.out[:-3] if args.out.endswith(".pt") else args.out

    def _save_consensus(path, z, rnd, **extra):
        z = [t.detach().cpu().clone() for t in z]
        torch.save({"state": {"z": z, "x": [z], "adapt_idx": list(adapt_idx)},
                    "meta": dict(meta, selected_round=rnd, **extra),
                    "collection_summary": collection_summary}, path)
        log(f"wrote {path}")

    if args.local_only:
        # No consensus variable exists: evaluate <out> with --deploy local.
        _write_report(args, report)
        print(f"\nwrote {args.out} (local-only: evaluate it with --deploy local)")
        return
    _save_consensus(stem + "-last.pt", trainer.admm.z[0], args.k_outer,
                    selection="final consensus variable z")
    best = getattr(trainer, "best_dev", None)
    if best is not None:
        _save_consensus(stem + "-best.pt", best[2], best[1],
                        selection="lowest held-out NLL", dev_nll_trained=best[0])
    _write_report(args, report)
    print(json.dumps({"residuals": train_stats["residuals"],
                      "delta_w_rel": train_stats["delta_w_rel"],
                      "comm_bytes_per_round_per_node": train_stats["comm_bytes_per_round_per_node"],
                      "backend": train_stats["backend"]}, indent=2))
    print(f"\nwrote {args.out}")


def _write_report(args, report) -> None:
    os.makedirs(args.report_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.out))[0]
    path = os.path.join(args.report_dir, f"{name}-train.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()


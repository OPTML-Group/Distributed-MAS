#!/usr/bin/env python3
"""Attach the safety reward to a fit set built by tools/build_fitset.py.

Every distinct rollout (identified by its final answer) of a query is scored
with the system-level outcome reward (App. C.3):

    harmful query   r = 1 iff the HarmBench classifier judges the final answer not harmful
    benign query    r = 1 iff the final answer is correct

and turned into group-relative advantages within the query. train.py then pairs,
per (query, agent), the message with the highest advantage (m^+) against the one
with the lowest (m^-).

Per-agent condition (Sec. 4, App. C.3; on by default): agent i's reward r_i for
a rollout is 1 only if the rollout succeeds AND agent i's own message is
acceptable -- not judged harmful on a harmful query, not a refusal on a benign
query. Advantages are then normalised per (query, agent), so an unsafe
intermediate message is never preferred because a later agent repaired the
final answer. `--no_per_agent_check` scores every agent with the rollout-level
reward only.

    python tools/judge_fitset.py ckpt/fitset-smas.pt --out ckpt/fitset-smas-judged.pt
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch                                                       # noqa: E402

from evaluation.safety_judge import HarmJudge, row_kind, seat_output_ok, trajectory_reward  # noqa: E402
from training.objectives import group_advantages                  # noqa: E402


def label_trajectories(groups, judge, batch):
    """Judge every distinct (query, final answer) once."""
    uniq = {}
    for anchor, exs in groups.items():
        for ex in exs:
            ft = getattr(ex, "final_text", "") or ""
            if ft:
                uniq[(anchor, ft)] = (ex.question or anchor, ft)
    print(f"distinct rollouts to judge: {len(uniq)}")
    keys = list(uniq)
    labels = judge.label([uniq[k] for k in keys], batch=batch)
    print(f"judged harmful: {sum(labels)}/{len(labels)}")
    return dict(zip(keys, labels))


def assign_trajectory_rewards(groups, verdict):
    n_deg = 0
    for anchor, exs in groups.items():
        by_ft = defaultdict(list)
        for ex in exs:
            by_ft[getattr(ex, "final_text", "") or ""].append(ex)
        fts = [f for f in by_ft if f]
        if not fts:
            continue
        rewards = []
        for ft in fts:
            e0 = by_ft[ft][0]
            rewards.append(trajectory_reward(getattr(e0, "gold", "") or "refuse",
                                             verdict.get((anchor, ft)),
                                             getattr(e0, "correct", None), ft))
        advs, degenerate = group_advantages(rewards)
        n_deg += bool(degenerate)
        for ft, r, adv in zip(fts, rewards, advs):
            for ex in by_ft[ft]:
                ex.reward = float(r)
                ex.advantage = float(adv)
    print(f"queries with equal rewards across rollouts (no preference): {n_deg}/{len(groups)}")


def assign_per_agent_rewards(groups, judge, batch):
    """Trajectory reward, capped at 0 when the agent's own message is unacceptable."""
    traj, seat = {}, {}
    for anchor, exs in groups.items():
        for ex in exs:
            if row_kind(getattr(ex, "gold", "")) != "harmful":
                continue
            ft = getattr(ex, "final_text", "") or ""
            if not ft:
                continue
            q = ex.question or anchor
            traj[(anchor, ft)] = (q, ft)
            if ex.output:
                seat[(anchor, ft, ex.agent_idx)] = (q, ex.output)
    print(f"judge calls: {len(traj)} rollouts + {len(seat)} agent messages")
    keys = list(traj) + list(seat)
    labels = judge.label([traj[k] for k in traj] + [seat[k] for k in seat], batch=batch)
    v = dict(zip(keys, labels))

    n_pen = n_tot = n_deg = n_groups = 0
    for anchor, exs in groups.items():
        by_ft = defaultdict(list)
        for ex in exs:
            by_ft[getattr(ex, "final_text", "") or ""].append(ex)
        traj_r = {}
        for ft, es in by_ft.items():
            if not ft:
                continue
            e0 = es[0]
            traj_r[ft] = trajectory_reward(getattr(e0, "gold", "") or "refuse",
                                           v.get((anchor, ft)), getattr(e0, "correct", None), ft)
        by_seat = defaultdict(list)
        for ft, es in by_ft.items():
            if ft not in traj_r:
                continue
            for ex in es:
                r = traj_r[ft]
                n_tot += 1
                if r > 0 and not seat_output_ok(getattr(ex, "gold", ""),
                                                v.get((anchor, ft, ex.agent_idx)),
                                                ex.output or ""):
                    r = 0.0
                    n_pen += 1
                ex.reward = float(r)
                by_seat[ex.agent_idx].append(ex)
        for _seat, es in by_seat.items():
            advs, degenerate = group_advantages([e.reward for e in es])
            n_groups += 1
            n_deg += bool(degenerate)
            for e, adv in zip(es, advs):
                e.advantage = float(adv)
    print(f"agent messages capped to reward 0: {n_pen}/{n_tot}; "
          f"(query, agent) groups {n_groups}, degenerate {n_deg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--out", default=None, help="default: overwrite the input file")
    ap.add_argument("--no_per_agent_check", action="store_true",
                    help="use the rollout-level reward for every agent")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    d = torch.load(a.cache, map_location="cpu", weights_only=False)
    flat = [ex for node in d.get("fitsets") or [] for ex in node]
    if not flat:
        raise SystemExit(f"{a.cache}: no examples")
    if any(row_kind(getattr(ex, "gold", "")) == "unknown" for ex in flat):
        raise SystemExit("some examples carry no gold label; rebuild the fit set")
    groups = defaultdict(list)
    for ex in flat:
        groups[ex.anchor].append(ex)
    print(f"{a.cache}: {len(flat)} examples, {len(groups)} queries")

    judge = HarmJudge(device=a.device)
    if not a.no_per_agent_check:
        assign_per_agent_rewards(groups, judge, a.batch)
    else:
        assign_trajectory_rewards(groups, label_trajectories(groups, judge, a.batch))

    k = d.setdefault("key", {})
    k["reward"] = "judge"
    k["judge_per_agent_check"] = not a.no_per_agent_check
    k["judge_fingerprint"] = judge.fingerprint()
    out = a.out or a.cache
    torch.save(d, out)
    print(f"judge: {judge.fingerprint()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

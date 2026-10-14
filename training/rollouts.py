"""MAS rollouts, outcome rewards and role-local training examples (Sec. 4, App. C.3).

For each training query we sample several rollouts of the whole MAS, score each
rollout with a binary system-level outcome reward r, and turn the scored group
into per-agent training examples:

    dpo   for every adapted agent i, pair its message m_i^+ from the best rollout
          with its message m_i^- from the worst rollout, both scored under the
          preferred rollout's context c_i (Eq. 8). Groups whose rewards are all
          equal carry no preference and are dropped.
    sft   every rollout's message becomes a target with weight 1 (e.g. with
          `--fit_prompt`, to fit messages collected under a hardened prompt).
    rsft  rejection-sampling SFT: only messages from successful rollouts
          (reward 1) become targets, with weight 1.

The examples of agent i form the local dataset of the training node that holds
agent i ("seat" partition), so each node's objective f_i is built only from its
own role's messages.

Rewards
    outcome   1 if the final answer is correct                 (reasoning)
    judge     HarmBench-classifier reward read from the fit-set cache, written
              by `tools/judge_fitset.py`                          (safety)
    gold_likelihood
              the final agent's log-probability of the gold answer given its
              prompt; a graded score that needs no final-agent generation
              (only meaningful for collecting `sft` targets)
"""
import hashlib
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from evaluation import scoring as S
from mas import Agent, agent_label, build_agents, resolve_prompt_builder
from training import objectives as objt
from training.lora import LoraFastState


def anchor_key(item: Dict) -> str:
    """Stable id of a query (sha1 of the question), stored in every checkpoint."""
    return hashlib.sha1(str(item.get("question", "")).encode("utf-8")).hexdigest()[:16]


def resolve_adapt_roles(agents: Sequence[Agent], mode: str) -> List[int]:
    """Which agents carry a trainable adapter.

        all       every agent (the paper's setting: "all agents remain trainable")
        nonsink   every agent except the final one
        middle    every agent except the first and the final one
    """
    if mode == "all":
        return list(range(len(agents)))
    if mode == "nonsink":
        idx = [i for i, a in enumerate(agents) if a.role != "judger"]
        if not idx:
            raise ValueError("no non-sink agents in the roster")
        return idx
    if mode == "middle":
        idx = [i for i, a in enumerate(agents) if a.role not in ("judger", "planner")]
        if not idx:
            raise ValueError("no middle agents in the roster; --adapt_roles middle needs N >= 3")
        return idx
    raise ValueError(f"unknown adapt_roles {mode!r}")


# --------------------------------------------------------------------------
# data records
# --------------------------------------------------------------------------

@dataclass
class AgentTurn:
    agent_idx: int
    role: str
    prompt: str            # rendered prompt under the collection builder
    output: str
    question: str = ""     # raw inputs, kept so the prompt can be re-rendered
    context: str = ""      # under a different builder at fit time


@dataclass
class Rollout:
    turns: List[AgentTurn]
    final_text: str
    prediction: Optional[str]
    reward: float
    seed: int


@dataclass
class FitExample:
    agent_idx: int
    node: int                    # training node holding this example
    prompt: str                  # conditioning context c_i, rendered
    output: str                  # target message (m_i^+ for DPO)
    advantage: float             # SFT weight; group-relative score for DPO pairing
    anchor: str                  # anchor_key of the query
    output_rej: Optional[str] = None   # DPO only: m_i^-
    role: Optional[str] = None
    question: str = ""
    context: str = ""
    final_text: str = ""         # the rollout's final answer (for offline judging)
    reward: float = 0.0          # the rollout's outcome reward r
    gold: str = ""
    correct: Optional[bool] = None


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# rollouts
# --------------------------------------------------------------------------

class ChainRoller:
    """Samples MAS rollouts with the same prompts the evaluation uses."""

    def __init__(self, model, args, *, n_agents: int = 4,
                 fast_state: Optional[LoraFastState] = None,
                 adapt_roles: str = "all") -> None:
        self.model = model
        self.args = args
        self.agents: List[Agent] = build_agents(n_agents)
        self.fast_state = fast_state
        # agent index -> training node; agents absent from the map run the
        # untouched backbone.
        self.adapt_idx: List[int] = resolve_adapt_roles(self.agents, adapt_roles)
        self.node_of: Dict[int, int] = {a: k for k, a in enumerate(self.adapt_idx)}
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 4096))
        # Budget of the final agent during training rollouts. 0 = do not generate
        # (prompt only; valid for `gold_likelihood`, which scores the prompt).
        sink = getattr(args, "sink_tokens", None)
        self.sink_tokens = self.max_new_tokens if sink is None else int(sink)
        self.temperature = float(getattr(args, "temperature", 0.8))
        self.top_p = float(getattr(args, "top_p", 0.95))
        self.prompt_mode = getattr(args, "prompt", "sequential_v2")
        self.build_messages = resolve_prompt_builder(self.prompt_mode)

    def _messages(self, role: str, question: str, context: str):
        return self.build_messages(role=role, question=question, context=context,
                                   method="text_mas", args=self.args)

    def _generate(self, messages, max_new_tokens: int) -> Tuple[str, str]:
        prompts, ids, mask, _ = self.model.prepare_chat_batch(
            [messages], add_generation_prompt=True)
        with torch.no_grad():
            out, _ = self.model.generate_text_batch(
                ids, mask, max_new_tokens=max_new_tokens,
                temperature=self.temperature, top_p=self.top_p)
        return prompts[0], out[0].strip()

    def _load_agent_state(self, agent_idx: int, states: Optional[Sequence]) -> None:
        if self.fast_state is None:
            return
        node = self.node_of.get(agent_idx)
        if states is None or node is None or node >= len(states):
            self.fast_state.disable()
            return
        self.fast_state.load(states[node])
        self.fast_state.enable()

    def chain(self, item: Dict, *, seed: int, states: Optional[Sequence] = None,
              downstream_seed: Optional[int] = None,
              prefix: Optional[List[AgentTurn]] = None) -> Rollout:
        """One full MAS rollout.

        `prefix` replays fixed upstream turns; `downstream_seed` pins the RNG of
        every agent after the first resampled one (used by `--credit per_role`).
        """
        question = item["question"]
        turns: List[AgentTurn] = list(prefix or [])
        context = "".join(
            f"[{agent_label(self.agents[t.agent_idx], self.prompt_mode)}]:\n{t.output}\n\n"
            for t in turns if self.agents[t.agent_idx].role != "judger")
        final_text = ""

        first = len(turns)
        for idx in range(len(turns), len(self.agents)):
            agent = self.agents[idx]
            is_sink = agent.role == "judger"
            budget = self.sink_tokens if is_sink else self.max_new_tokens
            step_seed = (seed + idx if (downstream_seed is None or idx == first)
                         else downstream_seed + idx)
            _seed_everything(step_seed)
            self._load_agent_state(idx, states)
            try:
                messages = self._messages(agent.role, question, context)
                if budget <= 0:
                    prompt = self.model.render_chat(messages, add_generation_prompt=True)
                    text = ""
                else:
                    prompt, text = self._generate(messages, budget)
            finally:
                if self.fast_state is not None:
                    self.fast_state.disable()
            turns.append(AgentTurn(idx, agent.role, prompt, text,
                                   question=question, context=context))
            if is_sink:
                final_text = text
            else:
                context += f"[{agent_label(agent, self.prompt_mode)}]:\n{text}\n\n"

        at = item.get("answer_type", "")
        pred = S.canonical(S.extract_answer(final_text), at)
        return Rollout(turns=turns, final_text=final_text, prediction=pred,
                       reward=0.0, seed=seed)


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------

def gold_likelihood_rewards(model, rollouts: List[Rollout], item: Dict) -> List[float]:
    """Final agent's mean log-probability of `\\boxed{gold}` given its prompt."""
    gold = str(item.get("gold", ""))
    if not gold:
        return [0.0] * len(rollouts)
    tok = model.tokenizer
    tgt = tok(f"\\boxed{{{gold}}}", add_special_tokens=False,
              return_tensors="pt").input_ids.to(model.device)
    out: List[float] = []
    for r in rollouts:
        ids = tok(r.turns[-1].prompt, add_special_tokens=False,
                  return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out.append(-float(objt.seq_nll(model, ids, tgt)))
    return out


def score_group(rollouts: List[Rollout], item: Dict, *, reward: str = "outcome",
                model=None) -> Dict:
    """Assign a reward to every rollout of one query (mutates `rollouts`)."""
    at = item.get("answer_type", "")
    gold = item.get("gold", "")
    if reward != "gold_likelihood" and rollouts and not rollouts[-1].final_text:
        raise ValueError(f"reward={reward!r} scores the final agent's answer, but it "
                         f"produced no text (--sink_tokens 0).")
    if reward == "gold_likelihood":
        if model is None:
            raise ValueError("reward=gold_likelihood needs the model")
        rewards = gold_likelihood_rewards(model, rollouts, item)
    elif reward == "outcome":
        rewards = [1.0 if S.answers_match(S.extract_answer(r.final_text), gold, at)
                   else 0.0 for r in rollouts]
    elif reward == "judge":
        raise ValueError("reward=judge is computed offline: collect with run.py, build "
                         "the fit set with tools/build_fitset.py and label it with "
                         "tools/judge_fitset.py, then pass it via --fitset_cache.")
    else:
        raise ValueError(f"unknown reward {reward!r}")

    for r, rw in zip(rollouts, rewards):
        r.reward = rw

    spread = (max(rewards) - min(rewards)) if rewards else 0.0
    if gold and rollouts and rollouts[-1].final_text:
        n_pass = sum(1 for r in rollouts
                     if S.answers_match(S.extract_answer(r.final_text), gold, at))
        pass_rate = n_pass / max(1, len(rollouts))
    else:
        pass_rate = None
    return {"pass_rate": pass_rate, "n": len(rollouts), "reward_spread": spread,
            "degenerate": spread < 1e-9}


# --------------------------------------------------------------------------
# training examples
# --------------------------------------------------------------------------

def _examples_from_group(rollouts: List[Rollout], anchor: str, fit_mode: str,
                         node_of: Dict[int, int]) -> List[FitExample]:
    """Turn one scored group into training examples for the adapted agents."""
    out: List[FitExample] = []

    if fit_mode == "dpo":
        # One pair per group: best vs worst rollout.
        ranked = sorted(rollouts, key=lambda r: r.reward)
        lo, hi = ranked[0], ranked[-1]
        if hi.reward - lo.reward < 1e-9:
            return out                       # degenerate group, no preference
        rej = {t.agent_idx: t.output for t in lo.turns}
        for t in hi.turns:
            node = node_of.get(t.agent_idx)
            if node is None or t.agent_idx not in rej:
                continue
            out.append(FitExample(t.agent_idx, node, t.prompt, t.output, 1.0,
                                  anchor, output_rej=rej[t.agent_idx],
                                  reward=float(hi.reward), final_text=hi.final_text or ""))
        return out

    if fit_mode == "sft":
        for r in rollouts:
            for t in r.turns:
                node = node_of.get(t.agent_idx)
                if node is None:
                    continue
                out.append(FitExample(t.agent_idx, node, t.prompt, t.output, 1.0, anchor,
                                      role=t.role, question=t.question, context=t.context,
                                      reward=float(r.reward), final_text=r.final_text or ""))
        if not out:
            raise ValueError("fit_mode=sft produced no examples from a non-empty group")
        return out

    if fit_mode == "rsft":
        for r in rollouts:
            if r.reward <= 0.5:
                continue
            for t in r.turns:
                node = node_of.get(t.agent_idx)
                if node is None:
                    continue
                out.append(FitExample(t.agent_idx, node, t.prompt, t.output, 1.0, anchor,
                                      role=t.role, question=t.question, context=t.context,
                                      reward=float(r.reward), final_text=r.final_text or ""))
        return out

    raise ValueError(f"unknown fit_mode {fit_mode!r}")


def collect_fitsets(
    roller: ChainRoller,
    items: Sequence[Dict],
    *,
    group_size: int = 8,
    credit: str = "shared",
    fit_mode: str = "dpo",
    reward: str = "outcome",
    pass_band: Optional[Tuple[float, float]] = None,
    model=None,
    seed: int = 42,
    downstream_seed: int = 1234,
    progress: Optional[Callable[[str], None]] = None,
    states: Optional[Sequence] = None,
) -> Tuple[List[List[FitExample]], Dict]:
    """Roll the queries, score them, and return one local dataset per training node.

    credit = shared    G end-to-end rollouts per query; every adapted agent is
                       credited with the system outcome (the paper's setting).
    credit = per_role  for each adapted agent, fix the upstream turns of a
                       reference rollout, resample only that agent G times and
                       pin the downstream RNG.
    pass_band (lo, hi) keeps only queries whose pass rate lies strictly inside
                       the band; (0, 1) keeps the groups with mixed outcomes.
    states             adapter state deployed on each adapted agent during the
                       rollouts (one per adapted agent); None = untrained backbone.
    """
    node_of = roller.node_of
    fitsets: List[List[FitExample]] = [[] for _ in range(len(node_of))]
    groups: List[Dict] = []
    t0 = time.time()
    n_skipped = 0

    def _in_band(stats) -> bool:
        nonlocal n_skipped
        if pass_band is None:
            return True
        pr = stats.get("pass_rate")
        if pr is None:
            return True
        lo, hi = pass_band
        if lo < pr < hi:
            return True
        n_skipped += 1
        return False

    for a_idx, item in enumerate(items):
        anchor = anchor_key(item)
        base = seed * 1000 + a_idx * 97

        if credit == "shared":
            rolls = [roller.chain(item, seed=base + g * 13, states=states)
                     for g in range(group_size)]
            stats = score_group(rolls, item, reward=reward, model=model)
            groups.append(stats)
            if _in_band(stats):
                for ex in _examples_from_group(rolls, anchor, fit_mode, node_of):
                    fitsets[ex.node].append(ex)
        elif credit == "per_role":
            ref = roller.chain(item, seed=base, downstream_seed=downstream_seed,
                               states=states)
            for role_idx in roller.adapt_idx:
                prefix = ref.turns[:role_idx]
                rolls = [
                    roller.chain(item, seed=base + 7919 * (role_idx + 1) + g,
                                 downstream_seed=downstream_seed, prefix=list(prefix),
                                 states=states)
                    for g in range(group_size)
                ]
                stats = score_group(rolls, item, reward=reward, model=model)
                groups.append(stats)
                if _in_band(stats):
                    for ex in _examples_from_group(rolls, anchor, fit_mode,
                                                   {role_idx: node_of[role_idx]}):
                        fitsets[ex.node].append(ex)
        else:
            raise ValueError(f"unknown credit {credit!r}")

        if progress:
            done = a_idx + 1
            progress(f"[{done}/{len(items)}] queries, "
                     f"fit sizes={[len(f) for f in fitsets]}, "
                     f"{(time.time() - t0) / done:.0f}s/query")

    prs = [g["pass_rate"] for g in groups if g.get("pass_rate") is not None]
    telemetry = {
        "mixed_group_rate": (sum(1 for p in prs if 0.0 < p < 1.0) / len(prs)) if prs else None,
        "pass_rate_mean": (sum(prs) / len(prs)) if prs else None,
        "groups_skipped_by_band": n_skipped,
        "pass_band": list(pass_band) if pass_band else None,
        "credit": credit, "fit_mode": fit_mode, "reward": reward,
        "n_queries": len(items), "group_size": group_size, "n_groups": len(groups),
        "degenerate_group_rate": round(
            sum(1 for g in groups if g["degenerate"]) / max(1, len(groups)), 4),
        "fit_sizes": [len(f) for f in fitsets],
        "sec_per_query": round((time.time() - t0) / max(1, len(items)), 1),
    }
    return fitsets, telemetry


def sft_to_dpo(fitsets: List[List[FitExample]]) -> List[List[FitExample]]:
    """Derive DPO pairs from an `sft` fit set whose rollouts carry rewards.

    Used for fit sets built from run.py trajectories (`tools/build_fitset.py` +
    `tools/judge_fitset.py`): per (query, agent), the message with the highest
    group-relative advantage is m^+ and the one with the lowest is m^-, both under
    m^+'s context. Groups with equal rewards yield no pair.
    """
    have_adv = any(abs(getattr(e, "advantage", 0.0)) > 1e-9 for fs in fitsets for e in fs)
    have_rew = any(getattr(e, "reward", None) is not None for fs in fitsets for e in fs)
    if not (have_adv or have_rew):
        raise ValueError("this sft fit set carries neither advantages nor rewards; "
                         "label it with tools/judge_fitset.py first")
    if not have_adv:
        for fs in fitsets:
            grp: Dict = {}
            for e in fs:
                grp.setdefault((e.anchor, e.agent_idx), []).append(e)
            for es in grp.values():
                advs, _ = objt.group_advantages([float(e.reward) for e in es])
                for e, a in zip(es, advs):
                    e.advantage = float(a)
    out = []
    for fs in fitsets:
        grp: Dict = {}
        for e in fs:
            grp.setdefault((e.anchor, e.agent_idx), []).append(e)
        pairs = []
        for (anchor, ai), es in grp.items():
            if len(es) < 2:
                continue
            hi = max(es, key=lambda e: e.advantage)
            lo = min(es, key=lambda e: e.advantage)
            if hi.advantage - lo.advantage < 1e-9:
                continue
            pairs.append(FitExample(ai, hi.node, hi.prompt, hi.output, 1.0,
                                    anchor, output_rej=lo.output,
                                    role=hi.role, question=hi.question,
                                    context=hi.context, gold=hi.gold,
                                    final_text=hi.final_text,
                                    reward=float(getattr(hi, "reward", 0.0))))
        out.append(pairs)
    return out


def sft_examples(fitsets: List[List[FitExample]], successful_only: bool = False):
    """SFT targets (weight 1) from an `sft` fit set built from trajectories.

    successful_only=False  every message (plain SFT / distillation)
    successful_only=True   only messages whose reward is 1 (rejection-sampling SFT);
                           with `tools/judge_fitset.py` this is agent i's own r_i
    """
    out = []
    for fs in fitsets:
        keep = [e for e in fs if not successful_only or float(e.reward) > 0.5]
        for e in keep:
            e.advantage = 1.0
        out.append(keep)
    if successful_only and not any(out):
        raise SystemExit("no example has reward 1: label the fit set with "
                         "tools/judge_fitset.py first")
    return out


def drop_unclosed_think(fitsets, log=None, min_keep: int = 16):
    """Discard targets whose <think> block never closed within the budget (App. C.3)."""
    before = [len(f) for f in fitsets]
    fitsets = [[ex for ex in f
                if str(ex.output or "").count("<think>") <= str(ex.output or "").count("</think>")]
               for f in fitsets]
    if log:
        log(f"drop_unclosed_think: examples per node {before} -> {[len(f) for f in fitsets]}")
    thin = [i for i, f in enumerate(fitsets) if len(f) < min_keep]
    if thin:
        raise SystemExit(f"after filtering, nodes {thin} hold fewer than {min_keep} "
                         f"examples ({[len(fitsets[i]) for i in thin]}); refusing to train.")
    return fitsets


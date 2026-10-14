"""Agent roster, role names and the prompt registry shared by evaluation and training.

Roles are identified by four internal keys in message-passing order. The last
agent (`judger`) produces the system output.

    key       S-MAS (sequential_v2)   H-MAS (hierarchical)
    planner   Solver                  Analyzer 1
    critic    Verifier                Analyzer 2
    refiner   Corrector               Analyzer 3
    judger    Decider                 Synthesizer

For N != 4 the chain keeps the first and last roles and fills the middle with
alternating critic / refiner agents (App. D).
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    name: str
    role: str


def build_agents(n_agents: int = 4) -> List[Agent]:
    if n_agents == 3:
        roles = [("Planner", "planner"), ("Refiner", "refiner"), ("Judger", "judger")]
    elif n_agents == 4:
        roles = [("Planner", "planner"), ("Critic", "critic"),
                 ("Refiner", "refiner"), ("Judger", "judger")]
    elif n_agents == 8:
        roles = (
            [("Planner", "planner")]
            + [(f"Critic{i}", "critic") for i in (1, 2, 3)]
            + [(f"Refiner{i}", "refiner") for i in (1, 2, 3)]
            + [("Judger", "judger")]
        )
    else:
        if n_agents < 2:
            raise ValueError("n_agents must be >= 2")
        middle = n_agents - 2
        roles = [("Planner", "planner")]
        for i in range(middle):
            r = "critic" if i % 2 == 0 else "refiner"
            roles.append((f"{r.capitalize()}{i // 2 + 1}", r))
        roles.append(("Judger", "judger"))
    return [Agent(name=n, role=r) for n, r in roles]


# Paper names accepted on the command line (e.g. `--agent_models solver=...`).
ROLE_ALIASES = {
    "solver": "planner", "verifier": "critic", "corrector": "refiner", "decider": "judger",
    "analyzer1": "planner", "analyzer2": "critic", "analyzer3": "refiner",
    "synthesizer": "judger",
}


def canonical_role(name: str) -> str:
    n = name.strip().lower()
    return ROLE_ALIASES.get(n, n)


PROMPT_MODES = ("sequential_v2", "hierarchical", "sequential_v2s", "hierarchical_v2s")


def prompt_modes() -> list:
    return list(PROMPT_MODES)


def resolve_prompt_builder(mode: str):
    """`--prompt` mode -> builder in prompts.py. Raises on an unknown mode."""
    from mas import prompts as P
    builders = {
        "sequential_v2": P.build_agent_messages_sequential_v2_text_mas,
        "hierarchical": P.build_agent_messages_hierarchical_text_mas,
        "sequential_v2s": P.build_agent_messages_sequential_v2s_text_mas,
        "hierarchical_v2s": P.build_agent_messages_hierarchical_v2s_text_mas,
    }
    if mode not in builders:
        raise ValueError(f"unknown --prompt {mode!r}; known modes are {sorted(builders)}")
    return builders[mode]


# In the hierarchical prompts the upstream messages shown to the Synthesizer are
# labelled with these display names (prompt text reproduced from the paper runs).
_HIER_DISPLAY = {
    "Planner": "Math Agent", "Critic": "Science Agent",
    "Refiner": "Code Agent", "Judger": "Task Summrizer",
}


def agent_label(agent: Agent, prompt_mode: str) -> str:
    """Name shown in the transcript header `[<label>]:` of an agent's message."""
    if not str(prompt_mode).startswith("hierarchical"):
        return agent.name
    base = agent.name.rstrip("0123456789")
    num = agent.name[len(base):]
    disp = _HIER_DISPLAY.get(base)
    if disp is None:
        return agent.name
    return f"{disp} {num}" if num else disp


__all__ = ["Agent", "build_agents", "ROLE_ALIASES", "canonical_role", "prompt_modes",
           "resolve_prompt_builder", "agent_label", "PROMPT_MODES"]

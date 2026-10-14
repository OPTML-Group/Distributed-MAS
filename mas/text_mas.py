"""Untrained ("Base") MAS: agents exchange natural-language messages in order.

Agent i receives the query and the transcript of all upstream messages
(`[<label>]:\\n<message>\\n\\n` per agent) and generates its own message. The
last agent's message is the system output. In the hierarchical prompts the
Analyzers ignore the transcript, so only the Synthesizer sees their analyses.

`role_models` maps a role key to its own ModelWrapper; this is how a
compromised agent (an unaligned backbone with the same role prompt) is placed in
the system. Roles absent from the map use the shared backbone.
"""
import argparse
from typing import Dict, List

import scoring
from models import ModelWrapper

from . import agent_label, build_agents, resolve_prompt_builder


class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        role_models: dict = None,
        max_new_tokens_each: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.role_models = dict(role_models or {})
        self.max_new_tokens_each = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.args = args
        self.method_name = "text_mas"           # prompts.py asserts on this
        self.task = args.task
        self.prompt_mode = args.prompt
        self.build_messages = resolve_prompt_builder(self.prompt_mode)
        self.agents = build_agents(int(getattr(args, "n_agents", 4) or 4))

    def _m(self, role: str) -> ModelWrapper:
        """The model that plays `role`. Falls back to the shared one."""
        return self.role_models.get(role, self.model)

    def role_model_names(self) -> dict:
        return {a.role: self._m(a.role).model_name for a in self.agents}

    def run_item(self, item: Dict) -> Dict:
        context = ""
        traces: List[Dict] = []
        final_text = ""

        for agent in self.agents:
            messages = self.build_messages(role=agent.role, question=item["question"],
                                           context=context, method=self.method_name,
                                           args=self.args)
            m = self._m(agent.role)
            prompts, input_ids, attention_mask, tokens_batch = m.prepare_chat_batch(
                [messages], add_generation_prompt=True)
            generated, _ = m.generate_text_batch(
                input_ids, attention_mask, max_new_tokens=self.max_new_tokens_each,
                temperature=self.temperature, top_p=self.top_p)
            text = generated[0].strip()

            if agent.role != "judger":
                context = f"{context}[{agent_label(agent, self.prompt_mode)}]:\n{text}\n\n"
            else:
                final_text = text
            mask = attention_mask[0].bool()
            traces.append({
                "name": agent.name,
                "role": agent.role,
                "input": prompts[0],
                "input_ids": input_ids[0][mask].to("cpu").tolist(),
                "input_tokens": tokens_batch[0],
                "output": text,
            })

        gold = item.get("gold", "")
        pred, r = scoring.score(final_text, gold, item.get("answer_type", ""))
        return {
            "question": item["question"],
            "gold": gold,
            "solution": item["solution"],
            "context": context,
            "prediction": pred,
            "raw_prediction": final_text,
            "agents": traces,
            "correct": bool(r >= 1.0),
        }

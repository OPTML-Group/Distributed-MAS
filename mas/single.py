"""Single-LLM baseline: one model answers the query directly (Fig. 2b, Table 1, Fig. 4)."""
from typing import Dict

from evaluation import scoring

from .models import ModelWrapper
from .prompts import build_agent_messages_single_agent


class BaselineMethod:
    def __init__(self, model: ModelWrapper, *, max_new_tokens: int = 4096,
                 temperature: float = 0.6, top_p: float = 0.95, args=None) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.method_name = "baseline"            # mas/prompts.py asserts on this
        self.args = args
        self.task = args.task

    def role_model_names(self) -> dict:
        return {"single": self.model.model_name}

    def run_item(self, item: Dict) -> Dict:
        messages = build_agent_messages_single_agent(question=item["question"], args=self.args)
        prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
            [messages], add_generation_prompt=True)
        gen, _ = self.model.generate_text_batch(
            input_ids, attention_mask, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, top_p=self.top_p)
        text = gen[0]
        gold = item.get("gold", "")
        pred, r = scoring.score(text, gold, item.get("answer_type", ""))
        mask = attention_mask[0].bool()
        return {
            "question": item["question"],
            "gold": gold,
            "solution": item["solution"],
            "prediction": pred,
            "raw_prediction": text,
            "agents": [{
                "name": "SingleAgent",
                "role": "singleagent",
                "input": prompts[0],
                "input_ids": input_ids[0][mask].to("cpu").tolist(),
                "input_tokens": tokens_batch[0],
                "output": text,
            }],
            "correct": bool(r >= 1.0),
        }


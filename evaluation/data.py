"""Dataset loaders for the benchmarks used in the paper.

Every loader yields dicts with at least `question`, `solution` and `gold`.
Safety rows carry `gold in {"refuse", "comply"}` and `answer_type="safety"`.

Reasoning benchmarks are read from the HuggingFace hub (`datasets`). The safety
benchmarks are plain CSV files placed under `$SAFETY_DATA_DIR` (default:
`$HF_HOME/safety_raw`); `scripts/download_safety_data.sh` fetches them.
"""
import os
import re
from typing import Dict, Iterable, Optional

from datasets import load_dataset


def extract_gold(text: str) -> Optional[str]:
    """GSM8K gold answer: the number after `####`."""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


# --------------------------------------------------------------------------
# reasoning benchmarks (Fig. 4)
# --------------------------------------------------------------------------

def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


BBH_SUBTASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "navigate",
    "logical_deduction_five_objects",
    "tracking_shuffled_objects_three_objects",
)


def _letters(n: int):
    return [chr(ord("A") + i) for i in range(n)]


def load_bbh(subtasks=None, split: str = "test", cache_dir=None):
    """BIG-Bench Hard. Option targets '(B)' become 'b'; word targets are lowercased."""
    subs = list(subtasks) if subtasks else list(BBH_SUBTASKS)
    for sub in subs:
        ds = load_dataset("lukaemon/bbh", sub, split=split, cache_dir=cache_dir)
        for item in ds:
            target = str(item["target"]).strip()
            if len(target) == 3 and target[0] == "(" and target[2] == ")":
                gold = target[1].lower()
            else:
                gold = target.lower()
            yield {
                "question": item["input"].strip(),
                "solution": target,
                "gold": gold,
                "subtask": sub,
            }


def load_math500(split: str = "test", cache_dir=None):
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split, cache_dir=cache_dir)
    for item in ds:
        yield {
            "question": item["problem"].strip(),
            "solution": item["solution"],
            "gold": str(item["answer"]).strip(),
            "level": item.get("level", ""),
            "subject": item.get("subject", ""),
        }


def load_mmlu_pro(split: str = "test", cache_dir=None, category=None):
    """MMLU-Pro, ten options (A-J) rendered into the question."""
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split, cache_dir=cache_dir)
    for item in ds:
        if category and item.get("category") != category:
            continue
        options = list(item["options"])
        labels = _letters(len(options))
        body = "\n".join(f"({l}) {o}" for l, o in zip(labels, options))
        yield {
            "question": f"{item['question'].strip()}\nOptions:\n{body}",
            "solution": str(item["answer"]),
            "gold": str(item["answer"]).strip().lower(),
            "category": item.get("category", ""),
        }


def load_musr(subsets=("murder_mysteries", "object_placements", "team_allocation"),
              split: str = "train", cache_dir=None):
    """MuSR. Per-item choices are rendered as lettered options."""
    for sub in subsets:
        # MuSR ships its three domains as splits of a single config.
        try:
            ds = load_dataset("TAUR-Lab/MuSR", split=sub, cache_dir=cache_dir)
        except Exception:
            continue
        for item in ds:
            choices = item["choices"]
            if isinstance(choices, str):
                import ast
                choices = ast.literal_eval(choices)
            labels = _letters(len(choices))
            body = "\n".join(f"({l}) {c}" for l, c in zip(labels, choices))
            idx = int(item["answer_index"])
            yield {
                "question": f"{item['narrative'].strip()}\n\n{item['question'].strip()}\nOptions:\n{body}",
                "solution": str(item.get("answer_choice", "")),
                "gold": labels[idx].lower(),
                "subtask": sub,
            }


# --------------------------------------------------------------------------
# safety benchmarks (Table 1, Fig. 2)
# --------------------------------------------------------------------------

def _safety_raw(name: str) -> str:
    root = os.environ.get("SAFETY_DATA_DIR") or os.path.join(
        os.environ.get("HF_HOME", ""), "safety_raw")
    path = os.path.join(root, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run scripts/download_safety_data.sh, or point "
            f"$SAFETY_DATA_DIR at the directory holding the safety CSVs.")
    return path


def load_xstest(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    """XSTest: 250 safe prompts (gold=comply) and 200 unsafe prompts (gold=refuse).

    The paper reports the refusal rate on the safe half (over-refusal).
    """
    import csv
    rows = []
    with open(_safety_raw("xstest_prompts.csv"), newline="", encoding="utf-8") as fh:
        for item in csv.DictReader(fh):
            unsafe = item["label"].strip().lower() == "unsafe"
            rows.append({
                "question": item["prompt"].strip(),
                "solution": "",
                "gold": "refuse" if unsafe else "comply",
                "answer_type": "safety",
                "subject": item["type"].strip(),
                "subtask": item["label"].strip(),
            })
    for r in rows:
        yield r


def load_harmbench(split: str = "test", cache_dir: Optional[str] = None,
                   category: str = "standard") -> Iterable[Dict]:
    """HarmBench text behaviours, using the official val/test split.

        split="val"   41 standard rows (training)
        split="test" 159 standard rows (evaluation)
        split="all"  200 standard rows

    `contextual` rows get their ContextString prepended to the behaviour.
    """
    import csv
    files = {"val": "harmbench_behaviors_text_val.csv",
             "test": "harmbench_behaviors_text_test.csv",
             "all": "harmbench_behaviors_text_all.csv"}
    if split not in files:
        raise ValueError(f"load_harmbench split={split!r} not in {sorted(files)}")
    with open(_safety_raw(files[split]), newline="", encoding="utf-8") as fh:
        for item in csv.DictReader(fh):
            if category and item["FunctionalCategory"].strip() != category:
                continue
            ctx = (item.get("ContextString") or "").strip()
            q = item["Behavior"].strip()
            if ctx:
                q = f"{ctx}\n\n{q}"
            yield {
                "question": q,
                "solution": "",
                "gold": "refuse",
                "answer_type": "safety",
                "subject": item["SemanticCategory"].strip(),
                "subtask": item["BehaviorID"].strip(),
            }


def load_advbench(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    """AdvBench `harmful_behaviors` (520 rows).

    Rows that also appear in HarmBench or XSTest are dropped, then the rest is
    split once with a fixed seed: `train` = 320 rows, `heldout` = the remainder.
    """
    import csv
    import random as _random
    seen = set()
    for r in load_harmbench(split="all"):
        seen.add(r["question"].strip().lower())
    for r in load_xstest():
        seen.add(r["question"].strip().lower())
    rows = []
    with open(_safety_raw("harmful_behaviors.csv"), newline="", encoding="utf-8") as fh:
        for item in csv.DictReader(fh):
            q = item["goal"].strip()
            if q.lower() in seen:
                continue
            rows.append({
                "question": q,
                "solution": "",
                "gold": "refuse",
                "answer_type": "safety",
                "subject": "advbench",
                "subtask": "harmful",
                "advbench_target": item["target"].strip(),
            })
    _random.Random(20260908).shuffle(rows)
    n_train = 320
    if split == "train":
        rows = rows[:n_train]
    elif split == "heldout":
        rows = rows[n_train:]
    elif split != "all":
        raise ValueError(f"load_advbench: unknown split {split!r}")
    for r in rows:
        yield r


def load_safety_train(n_benign: int = 300, cache_dir: Optional[str] = None) -> Iterable[Dict]:
    """Safety training pool (Sec. 5.1 / App. C.3).

    harmful (gold=refuse), 380 rows:
        AdvBench train split ............ 320
        HarmBench val, standard .........  41   (disjoint from the 159-row test)
        HarmBench val, contextual .......  19
    benign (gold = the MMLU-Pro answer letter), 300 rows:
        MMLU-Pro test slice ............. 300   (over-refusal / stay-helpful control)

    Rows are de-duplicated and shuffled once with a fixed seed so that any
    prefix is a mixture of harmful and benign rows. Benign rows keep their
    native answer type and are scored as ordinary questions.
    """
    import hashlib as _hl
    import random as _rnd
    seen = set()
    rows = []

    def _add(it, split_tag, source, gold, answer_type):
        q = str(it.get("question", "")).strip()
        key = q.lower()
        if not q or key in seen:
            return False
        seen.add(key)
        r = dict(it)
        r["question"] = q
        r["gold"] = gold
        r["sd1_split"] = split_tag
        r["source"] = source
        if answer_type is not None:
            r["answer_type"] = answer_type
        rows.append(r)
        return True

    for it in load_advbench(split="train"):
        _add(it, "harmful", "advbench_train", "refuse", "safety")
    for it in load_harmbench(split="val", category="standard"):
        _add(it, "harmful", "harmbench_val_standard", "refuse", "safety")
    for it in load_harmbench(split="val", category="contextual"):
        _add(it, "harmful", "harmbench_val_contextual", "refuse", "safety")
    n_harm = len(rows)

    added = 0
    for it in load_mmlu_pro(split="test", cache_dir=cache_dir):
        if added >= n_benign:
            break
        if _add(it, "benign", "mmlu_pro_test", str(it.get("gold", "")), None):
            added += 1
    n_ben = len(rows) - n_harm
    sha = _hl.sha256("\n".join(r["question"] for r in rows).encode()).hexdigest()[:12]
    _rnd.Random(20260910).shuffle(rows)
    print(f"[load_safety_train] {len(rows)} rows = {n_harm} harmful + {n_ben} benign "
          f"(mmlu_pro); question-set sha={sha}", flush=True)
    for r in rows:
        yield r


def _safety_csv_task(fname, qcol, gold, extra=None):
    import csv as _csv
    with open(_safety_raw(fname), newline="", encoding="utf-8") as fh:
        for it in _csv.DictReader(fh):
            q = (it.get(qcol) or "").strip()
            if not q:
                continue
            row = {"question": q, "solution": "", "gold": gold,
                   "answer_type": "safety", "subject": fname.split(".")[0],
                   "category": (it.get("category") or it.get("Category") or "").strip()}
            for k, src in (extra or {}).items():
                row[k] = (it.get(src) or "").strip()
            yield row


def load_jbb_harmful(split: str = "test", cache_dir=None):
    """JailbreakBench, 100 harmful behaviours (Goal column)."""
    yield from _safety_csv_task("jbb_harmful_behaviors.csv", "Goal", "refuse",
                                extra={"jbb_target": "Target"})


def load_strongreject(split: str = "test", cache_dir=None):
    """StrongREJECT forbidden prompts (313 rows)."""
    yield from _safety_csv_task("strongreject_dataset.csv", "forbidden_prompt", "refuse")


def load_orbench_toxic(split: str = "test", cache_dir=None):
    """OR-Bench-Toxic: genuinely harmful prompts."""
    yield from _safety_csv_task("or-bench-toxic.csv", "prompt", "refuse")

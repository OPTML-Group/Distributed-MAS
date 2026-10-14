import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_gold(text: str) -> Optional[str]:
    """GSM8K gold answer: the number after `####`."""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def latex_equal(a: Optional[str], b: Optional[str]) -> bool:
    """String comparison after whitespace/delimiter normalization."""
    if a is None or b is None:
        return False

    def norm(s: str) -> str:
        s = s.strip().replace(" ", "").replace("\\left", "").replace("\\right", "")
        s = s.replace("\\!", "").replace("\\,", "").replace("dfrac", "frac").replace("tfrac", "frac")
        return s.rstrip(".").lower()

    return norm(a) == norm(b)

# utils_common.py
from typing import Optional
import torch

# Один раз выбираем девайс
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
torch.set_num_threads(1)

def ensure_str(x: Optional[object]) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    return str(x)

import math, numpy as np

def perplexity(nll: float) -> float:
    return float(math.exp(nll))

def accuracy(y_true, y_pred) -> float:
    if not y_true: return 0.0
    return float(np.mean([a==b for a,b in zip(y_true, y_pred)]))

def exact_match(a: str, b: str) -> bool:
    return a.strip() == b.strip()

def contains_norm(text: str, needle: str) -> bool:
    t = " ".join(text.lower().split())
    n = " ".join(needle.lower().split())
    return n in t

# Evaluation skeleton
# - accuracy proxies, hallucination checks (heuristic)
# - latency and cost accounting (stubs)
from dataclasses import dataclass

@dataclass
class EvalResult:
    name: str
    score: float
    notes: str = ""

def exact_match(pred: str, gold: str) -> EvalResult:
    return EvalResult(name="exact_match", score=float(pred.strip() == gold.strip()))

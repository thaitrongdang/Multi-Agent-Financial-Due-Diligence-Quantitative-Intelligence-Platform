import re
from typing import Any

from pydantic import BaseModel


class EvalSampleResult(BaseModel):
    question_id: str
    ticker: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    latency_ms: float
    was_self_corrected: bool = False


def context_precision(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Mean Average Precision @ K over retrieved chunk IDs."""
    if not retrieved_ids or not expected_ids:
        return 0.0
    expected_set = set(expected_ids)
    hits = 0
    precisions: list[float] = []
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_set:
            hits += 1
            precisions.append(hits / rank)
    if not precisions:
        return 0.0
    return round(sum(precisions) / len(expected_set), 4)


def context_recall(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of ground-truth relevant chunks successfully retrieved."""
    if not expected_ids:
        return 1.0
    expected_set = set(expected_ids)
    retrieved_set = set(retrieved_ids)
    return round(len(expected_set & retrieved_set) / len(expected_set), 4)


def _extract_numeric_tokens(text: str) -> set[str]:
    raw_nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    normalized: set[str] = set()
    for num in raw_nums:
        val = float(num)
        normalized.add(f"{val:g}")
    return normalized


def faithfulness_score(
    answer: str,
    contexts: list[dict[str, Any]],
    financial_metrics: dict[str, Any],
) -> float:
    """Verify that numerical claims in the generated report exist in verified sources."""
    answer_nums = _extract_numeric_tokens(answer)
    if not answer_nums:
        return 1.0 if answer.strip() else 0.0

    source_blob = " ".join(str(c.get("content", "")) for c in contexts)
    source_blob += " " + " ".join(str(c.get("page_number", "")) for c in contexts)
    for val in financial_metrics.values():
        if isinstance(val, int | float):
            source_blob += f" {val:g} {round(float(val), 2):g}"

    allowed_nums = _extract_numeric_tokens(source_blob) | {"1", "2", "3", "4", "5", "2025", "2026"}
    grounded = answer_nums & allowed_nums
    return round(len(grounded) / len(answer_nums), 4)


def answer_relevance_score(
    question: str,
    answer: str,
    expected_facts: list[str],
) -> float:
    """Measure coverage of ground-truth key facts and question term alignment."""
    if not answer.strip():
        return 0.0
    lower_answer = answer.lower()

    fact_hits = sum(1 for fact in expected_facts if fact.lower() in lower_answer)
    fact_ratio = fact_hits / max(len(expected_facts), 1)

    q_tokens = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 3}
    a_tokens = set(re.findall(r"[a-z0-9]+", lower_answer))
    lexical_overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)

    return round((0.75 * fact_ratio) + (0.25 * lexical_overlap), 4)

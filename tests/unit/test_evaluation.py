from evaluation.metrics import (
    answer_relevance_score,
    context_precision,
    context_recall,
    faithfulness_score,
)
from evaluation.run_benchmark import load_benchmark_dataset, run_ablation_benchmark


def test_context_precision_and_recall_metrics() -> None:
    retrieved = ["c1", "noise_1", "c2"]
    expected = ["c1", "c2"]

    precision = context_precision(retrieved, expected)
    recall = context_recall(retrieved, expected)

    assert precision == round((1.0 / 1 + 2.0 / 3) / 2, 4)
    assert recall == 1.0


def test_faithfulness_penalizes_ungrounded_numbers() -> None:
    contexts = [{"chunk_id": "c1", "page_number": 5, "content": "Bond principal is 1250 VND Bn."}]
    metrics = {"current_ratio": 1.5}

    faithful_text = "Current ratio is 1.5 and bond principal is 1250 VND Bn [Chunk: c1, Page 5]."
    hallucinated_text = "Current ratio is 1.5 and bond principal is 9999 VND Bn."

    assert faithfulness_score(faithful_text, contexts, metrics) == 1.0
    assert faithfulness_score(hallucinated_text, contexts, metrics) < 1.0


def test_answer_relevance_scores_fact_coverage() -> None:
    question = "Assess corporate bond maturity and current liquidity ratio"
    good_answer = "The current liquidity ratio is 1.5 with corporate bond maturing in Q3 2026."
    weak_answer = "The company operates in the technology sector."

    good_score = answer_relevance_score(question, good_answer, ["1.5", "Q3 2026"])
    weak_score = answer_relevance_score(question, weak_answer, ["1.5", "Q3 2026"])

    assert good_score > 0.85
    assert weak_score < 0.25


def test_ablation_benchmark_proves_full_system_outperforms_baseline() -> None:
    dataset = load_benchmark_dataset()
    assert len(dataset) >= 3

    summaries = run_ablation_benchmark(use_live_llm=False)
    by_variant = {s["variant"]: s for s in summaries}

    baseline = by_variant["dense_baseline"]
    full = by_variant["full_multi_agent"]

    assert full["avg_faithfulness"] > baseline["avg_faithfulness"]
    assert full["avg_faithfulness"] >= 0.95
    assert full["avg_context_recall"] >= baseline["avg_context_recall"]
    assert full["self_correction_rate"] == 1.0

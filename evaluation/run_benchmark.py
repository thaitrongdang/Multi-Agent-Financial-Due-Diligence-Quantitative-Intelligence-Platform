import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from evaluation.metrics import (
    EvalSampleResult,
    answer_relevance_score,
    context_precision,
    context_recall,
    faithfulness_score,
)
from src.agents.graph import build_due_diligence_graph
from src.agents.llm_factory import get_chat_model
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import FinancialReranker
from src.rag.schemas import DocumentChunk
from src.rag.vector_store import QdrantVectorStore

DEFAULT_DATASET_PATH = Path("evaluation/datasets/financial_benchmark_v1.json")
DEFAULT_REPORT_PATH = Path("evaluation/benchmark_results.md")


class BenchmarkEmbedder:
    """Deterministic lightweight embedder for reproducible offline ablation runs."""

    @property
    def dimension(self) -> int:
        return 64

    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = digest[0] % self.dimension
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


class AblationSimulatedLLM(BaseChatModel):
    """Simulates grounded vs ungrounded synthesis and Critic self-correction."""

    hallucinate_first_pass: bool = False
    call_counter: int = 0

    @property
    def _llm_type(self) -> str:
        return "ablation-benchmark-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_msg = str(messages[0].content)
        human_msg = str(messages[-1].content)
        self.call_counter += 1

        if "Fact-Checking Auditor" in system_msg:
            has_hallucination = "99.99" in human_msg
            verdict = {
                "is_faithful": not has_hallucination,
                "feedback": "Detected unverified figure 99.99." if has_hallucination else "",
            }
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=json.dumps(verdict)))]
            )

        is_retry = "PREVIOUS DRAFT WAS REJECTED" in human_msg
        if self.hallucinate_first_pass and not is_retry:
            draft = "Preliminary memo: Unverified leverage ratio 99.99 without filing citation."
        else:
            draft = self._synthesize_from_prompt(human_msg)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=draft))])

    @staticmethod
    def _synthesize_from_prompt(prompt: str) -> str:
        metrics_part = prompt.split("VERIFIED QUANTITATIVE METRICS:\n", 1)[1].split(
            "\n\nRETRIEVED FILING CONTEXTS:", 1
        )[0]
        contexts_part = prompt.split("RETRIEVED FILING CONTEXTS:\n", 1)[1].split(
            "\n\nMARKET SENTIMENT SUMMARY:", 1
        )[0]
        metrics = json.loads(metrics_part)
        contexts = json.loads(contexts_part)

        lines = [
            f"## Due Diligence Memo: {metrics.get('ticker')}",
            f"- Current Ratio: {metrics.get('current_ratio')}",
            f"- Debt-to-Equity: {metrics.get('debt_to_equity')}",
            f"- Net Debt / EBITDA: {metrics.get('net_debt_to_ebitda')}",
        ]
        for ctx in contexts:
            lines.append(
                f"- Filing Evidence [Chunk: {ctx['chunk_id']}, Page {ctx['page_number']}]: "
                f"{ctx['content']}"
            )
        return "\n".join(lines)


def load_benchmark_dataset(dataset_path: Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def _evaluate_variant(
    dataset: list[dict[str, Any]],
    variant: str,
    use_live_llm: bool = False,
) -> dict[str, Any]:
    embedder = BenchmarkEmbedder()
    results: list[EvalSampleResult] = []

    for idx, item in enumerate(dataset):
        store = QdrantVectorStore(
            embedder=embedder,
            collection_name=f"bench_{variant}_{idx}",
            in_memory=True,
        )
        chunks = [DocumentChunk.model_validate(c) for c in item["corpus_chunks"]]
        retriever = HybridRetriever(vector_store=store)
        retriever.index_chunks(chunks)

        start_ts = time.perf_counter()

        if variant == "dense_baseline":
            dense_hits = store.dense_search(
                query=item["question"],
                ticker=item["ticker"],
                fiscal_year=item["fiscal_year"],
                limit=2,
            )
            retrieved_ids = [c.chunk_id for c, _ in dense_hits]
            contexts = [
                {
                    "chunk_id": c.chunk_id,
                    "page_number": c.page_number,
                    "content": c.content,
                }
                for c, _ in dense_hits
            ]
            llm = (
                get_chat_model()
                if use_live_llm
                else AblationSimulatedLLM(hallucinate_first_pass=True)
            )
            graph = build_due_diligence_graph(llm=llm, retriever=None)
            output = graph.invoke(
                {
                    "ticker": item["ticker"],
                    "fiscal_year": item["fiscal_year"],
                    "query": item["question"],
                    "financial_input": item["financial_input"],
                    "extracted_contexts": contexts,
                    "max_retries": 0,
                }
            )
        else:
            llm = (
                get_chat_model()
                if use_live_llm
                else AblationSimulatedLLM(hallucinate_first_pass=(variant == "full_multi_agent"))
            )
            max_retries = 0 if variant == "hybrid_no_critic" else 2
            graph = build_due_diligence_graph(
                llm=llm,
                retriever=retriever,
                reranker=FinancialReranker(),
            )
            output = graph.invoke(
                {
                    "ticker": item["ticker"],
                    "fiscal_year": item["fiscal_year"],
                    "query": item["question"],
                    "financial_input": item["financial_input"],
                    "news_headlines": item["news_headlines"],
                    "max_retries": max_retries,
                }
            )
            retrieved_ids = [c["chunk_id"] for c in output.get("extracted_contexts", [])]

        latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
        report = output.get("draft_report", "")
        metrics = output.get("financial_metrics", {})
        contexts_out = output.get("extracted_contexts", [])

        results.append(
            EvalSampleResult(
                question_id=item["question_id"],
                ticker=item["ticker"],
                context_precision=context_precision(retrieved_ids, item["expected_chunk_ids"]),
                context_recall=context_recall(retrieved_ids, item["expected_chunk_ids"]),
                faithfulness=faithfulness_score(report, contexts_out, metrics),
                answer_relevance=answer_relevance_score(
                    item["question"], report, item["expected_facts"]
                ),
                latency_ms=latency_ms,
                was_self_corrected=output.get("retry_count", 0) > 0,
            )
        )

    latencies = [r.latency_ms for r in results]
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)
    sorted_lat = sorted(latencies)

    return {
        "variant": variant,
        "sample_count": len(results),
        "avg_context_precision": round(statistics.mean(r.context_precision for r in results), 4),
        "avg_context_recall": round(statistics.mean(r.context_recall for r in results), 4),
        "avg_faithfulness": round(statistics.mean(r.faithfulness for r in results), 4),
        "avg_answer_relevance": round(statistics.mean(r.answer_relevance for r in results), 4),
        "p50_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(sorted_lat[p95_idx], 2),
        "self_correction_rate": round(
            sum(1 for r in results if r.was_self_corrected) / len(results), 4
        ),
    }


def run_ablation_benchmark(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    use_live_llm: bool = False,
) -> list[dict[str, Any]]:
    dataset = load_benchmark_dataset(dataset_path)
    variants = ["dense_baseline", "hybrid_no_critic", "full_multi_agent"]
    return [_evaluate_variant(dataset, v, use_live_llm=use_live_llm) for v in variants]


def render_markdown_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# RAG & Multi-Agent Evaluation Benchmark Results",
        "",
        "| Architecture Variant | Context Precision | Context Recall | Faithfulness | "
        "Answer Relevance | P50 Latency (ms) | Self-Correction Rate |",
        "| --- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    labels = {
        "dense_baseline": "1. Dense-Only RAG (No Reranker / No Critic)",
        "hybrid_no_critic": "2. Hybrid RAG + Reranker (No Critic Loop)",
        "full_multi_agent": "3. Full Multi-Agent + Hybrid RAG + Critic",
    }
    for row in summaries:
        name = labels.get(row["variant"], row["variant"])
        lines.append(
            f"| {name} | {row['avg_context_precision']:.4f} | "
            f"{row['avg_context_recall']:.4f} | {row['avg_faithfulness']:.4f} | "
            f"{row['avg_answer_relevance']:.4f} | {row['p50_latency_ms']:.2f} | "
            f"{row['self_correction_rate']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ragas-style financial evaluation benchmark.")
    parser.add_argument("--live-llm", action="store_true", help="Use configured live LLM provider.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    summaries = run_ablation_benchmark(use_live_llm=args.live_llm)
    markdown = render_markdown_report(summaries)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

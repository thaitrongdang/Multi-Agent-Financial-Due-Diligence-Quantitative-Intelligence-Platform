# RAG & Multi-Agent Evaluation Benchmark Results

| Architecture Variant | Context Precision | Context Recall | Faithfulness | Answer Relevance | P50 Latency (ms) | Self-Correction Rate |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Dense-Only RAG (No Reranker / No Critic) | 0.6667 | 0.6667 | 0.0000 | 0.0185 | 7.24 | 100.00% |
| 2. Hybrid RAG + Reranker (No Critic Loop) | 1.0000 | 1.0000 | 1.0000 | 0.8417 | 8.50 | 0.00% |
| 3. Full Multi-Agent + Hybrid RAG + Critic | 1.0000 | 1.0000 | 1.0000 | 0.8417 | 9.88 | 100.00% |

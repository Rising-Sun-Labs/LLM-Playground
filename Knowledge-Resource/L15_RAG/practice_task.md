# Practice:

- Build `corpus.jsonl` from 50-200 pages/files with `{doc_id, source, meta, text}`.
- Create FAISS index + (optional) BM25; stand up `/rag_answer`.
- Add **MMR + reranker**, restrict to **6 passages** in prompt, enforce citations.
- Create `rag_qa.jsonl` (20-50 QAs) and implement EM/F1 + groundedness check; report p50/p95 latency.
- Add cache keyed by `(query, corpus_version)` for retrieve + answer.

Stretch:
    - Swap in a stronger embedder (e.g., `bge-large-en`) and compare recall/latency.
    - Implement **multimodel RAG** for screenshots (CLIP retrieval + MM generations).
    - Add **snippet highlighting**: pick top 2-3 sentences per passage (e.g., using Maximal Marginal Relevance at sentence level).
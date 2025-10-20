# Lesson 15: RAG = Retrieval augmented generation
    - RAG with a proper embedder, chunker, reranker, and citations - plus multimodel RAG.
    - Plug your model into external knowledge so it answers with facts (and citations), not vibes. this is RAG: chunk -> embed -> search -> rerank -> compose -> answer -> cite.

0) Goals => RAG + Citations
    - build a **RAG Pipeline (ingest -> index -> retrieve -> rerank -> prompt)**.
    - support **hybrid search** (BM25 + vectors) and **cited answers**.
    - add a reranker and **context windowing** that avoids token bloat.
    - handle **code/docs/long PDFs and images
    - evaluate factuality (EM/F1) + groundedness (citation checks) + latency.


1) data model & chunking
    - Why chunk? smaller passages = better recall; too small = lose context. start simple:
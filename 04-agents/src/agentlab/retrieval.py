"""Hybrid retriever for the agent's search tool: dense + BM25, fused with RRF, optional re-rank.

This is the Project 2 recipe (bge-small + BM25 with a stopword tokenizer, weighted
reciprocal rank fusion, cross-encoder re-ranking of a candidate pool) wrapped in a class
instead of notebook globals. Two differences from Project 2: there is no chunking
(HotpotQA paragraphs are already short), and the title is part of the indexed text.

The models are passed in as plain functions, so the logic is testable without a GPU:
    embed_fn(list[str])             -> array (n, d), rows L2-normalized
    rerank_fn(list[(query, text)])  -> list[float], higher is better

Needs numpy and rank_bm25. Import it explicitly (`from agentlab.retrieval import ...`);
the core package does not depend on them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .tools import Tool

STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "is", "was", "were", "are", "and", "or",
             "for", "by", "with", "as", "what", "which", "who", "when", "where", "how", "did", "does",
             "do", "from", "that", "this", "it", "its", "be", "been", "has", "have", "had"}

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def rrf_fuse(ranked_lists: Sequence[Sequence], weights: Sequence[float], k_constant: int = 60, top_k: int = 10) -> list:
    """Weighted reciprocal rank fusion. An item scores sum(weight / (k_constant + rank)) over the lists."""
    scores: dict = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + weight / (k_constant + rank + 1)
    return sorted(scores, key=lambda item: -scores[item])[:top_k]


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str

    @property
    def indexed(self) -> str:
        return f"{self.title}. {self.text}"


class HybridRetriever:
    def __init__(
        self,
        passages: Sequence[Passage],
        embed_fn: Callable[[list[str]], np.ndarray],
        rerank_fn: Callable[[list[tuple[str, str]]], Sequence[float]] | None = None,
        pool: int = 20,
        candidates: int = 50,
    ):
        """
        pool        how many hybrid results the re-ranker sees (ignored without a re-ranker).
                    A re-ranker only reorders its pool, so recall@pool is its ceiling.
        candidates  how many results each of dense and BM25 contributes to the fusion
        """
        if not passages:
            raise ValueError("need at least one passage")
        if len({p.id for p in passages}) != len(passages):
            raise ValueError("passage ids must be unique")
        self.passages = list(passages)
        self.embed_fn, self.rerank_fn = embed_fn, rerank_fn
        self.pool, self.candidates = pool, candidates
        texts = [p.indexed for p in self.passages]
        self._emb = np.asarray(embed_fn(texts))
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    @classmethod
    def from_models(
        cls,
        passages: Sequence[Passage],
        embedder_id: str = DEFAULT_EMBEDDER,
        reranker_id: str | None = DEFAULT_RERANKER,
        **kwargs,
    ) -> "HybridRetriever":
        """Load the real models (needs sentence-transformers). Pass reranker_id=None to skip re-ranking."""
        from sentence_transformers import CrossEncoder, SentenceTransformer

        embedder = SentenceTransformer(embedder_id)
        embed_fn = lambda texts: embedder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        rerank_fn = None
        if reranker_id:
            ce = CrossEncoder(reranker_id, max_length=512)
            rerank_fn = lambda pairs: ce.predict(pairs, batch_size=32, show_progress_bar=False)
        return cls(passages, embed_fn, rerank_fn, **kwargs)

    def _hybrid(self, query: str, top_k: int) -> list[int]:
        q_emb = np.asarray(self.embed_fn([query]))[0]
        dense = np.argsort(-(self._emb @ q_emb))[: self.candidates]
        sparse = np.argsort(-self._bm25.get_scores(tokenize(query)))[: self.candidates]
        return rrf_fuse([dense.tolist(), sparse.tolist()], [1.0, 1.0], top_k=top_k)

    def search(self, query: str, k: int = 3) -> list[Passage]:
        if not query.strip() or k < 1:
            return []
        if self.rerank_fn is None:
            return [self.passages[i] for i in self._hybrid(query, k)]
        candidates = self._hybrid(query, max(self.pool, k))
        scores = self.rerank_fn([(query, self.passages[i].indexed) for i in candidates])
        order = sorted(range(len(candidates)), key=lambda j: -scores[j])
        return [self.passages[candidates[j]] for j in order[:k]]


def make_search_tool(retriever: HybridRetriever, k: int = 3, max_chars: int = 600) -> Tool:
    """Wrap a retriever as the agent's `search` tool. Results look like: [Title] passage text."""

    def search(query: str) -> str:
        hits = retriever.search(query, k)
        if not hits:
            return "No results."
        lines = []
        for p in hits:
            text = p.text if len(p.text) <= max_chars else p.text[:max_chars] + "..."
            lines.append(f"[{p.title}] {text}")
        return "\n".join(lines)

    return Tool(
        "search",
        f"Search the document collection. Returns the top {k} passages, each as [Title] text.",
        {"query": "keywords or a short question describing what to find"},
        search,
    )

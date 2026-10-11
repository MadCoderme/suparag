"""
retrieval/reranking/flashrank.py
FlashRank Reranking module after retrieval.
"""

import os
from typing import List

from flashrank import Ranker, RerankRequest

from retrieval import BaseReranker
from schema.retrieval import ScoredChunk, FlashRankModel

class FlashRankReranker(BaseReranker):
    def __init__(self, model: FlashRankModel = "ms-marco-MiniLM-L-6-v2"):
        self.ranker = Ranker(model_name=model, cache_dir="/tmp")

    def rerank(self, query: str, scored_chunks: List[ScoredChunk], top_k: int = 3) -> List[ScoredChunk]:
        if not scored_chunks or len(scored_chunks) == 0:
            return []

        formatted_docs = [{"id": str(i), "text": chunk.chunk.content} for i, chunk in enumerate(scored_chunks)]

        req = RerankRequest(
            query, passages=formatted_docs
        )

        results = self.ranker.rerank(req)

        reranked_chunks = [scored_chunks[c.id] for c in results[:top_k]]
        return reranked_chunks
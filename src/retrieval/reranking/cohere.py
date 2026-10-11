"""
retrieval/reranking/cohere.py
Cohere Reranking module after retrieval.
"""

import os
from typing import List

from cohere import Client

from retrieval import BaseReranker
from schema.retrieval import ScoredChunk, CohereModel

co = Client(api_key=os.environ.get("COHERE_API_KEY", None))

class CohereReranker(BaseReranker):
    def __init__(self, model: CohereModel = "rerank-v3.5"):
        self.model = model
    def rerank(self, query: str, scored_chunks: List[ScoredChunk], top_k: int = 3) -> List[ScoredChunk]:
        if not scored_chunks or len(scored_chunks) == 0:
            return []

        response = co.rerank(
            model=self.model,
            query=query,
            documents=[chunk.chunk.content for chunk in scored_chunks],
            top_n=top_k
        )

        reranked_chunks = [scored_chunks[c.index] for c in response.results]
        return reranked_chunks
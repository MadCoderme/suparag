"""
retrieval/reranking/cross_encoder.py
Cross Encoder Reranker after retrieval.
"""

from typing import List

import torch
from sentence_transformers import CrossEncoder

from retrieval import BaseReranker
from schema.retrieval import ScoredChunk, CrossEncoderModel


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: CrossEncoderModel = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, scored_chunks: List[ScoredChunk], top_k: int = 3) -> List[ScoredChunk]:
        if not scored_chunks or len(scored_chunks) == 0:
            return []
        pairs = [[query, chunk.chunk.content] for chunk in scored_chunks]
        scores = self.model.predict(pairs, convert_to_tensor=True)
        ranked_indices = torch.argsort(scores, descending=True).tolist()

        reranked_chunks = [scored_chunks[i] for i in ranked_indices[:top_k]]

        return reranked_chunks
        
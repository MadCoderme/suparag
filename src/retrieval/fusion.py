"""
retrieval/fusion.py
RRF (Reciprocal Rank Fusion) of multiple retrieval results.
"""

from typing import List

from schema.retrieval import ScoredChunk


class RRF:
    def __init__(self, k=60):
        self.k = k

    def fuse(self, sparse_results: List[ScoredChunk], dense_results: List[ScoredChunk]):
        rrf_scores = {}

        for s_chunk in sparse_results:
            rrf_scores[s_chunk.chunk.id] = rrf_scores[s_chunk.chunk.id] + (1/(self.k + s_chunk.score_breakdown.sparse_score))
        for d_chunk in dense_results:
            rrf_scores[d_chunk.chunk.id] = rrf_scores.get(d_chunk.chunk.id, 0) + (1/(self.k + d_chunk.score_breakdown.dense_score))

        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused_results
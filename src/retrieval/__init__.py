"""
retrieval/__init__.py
Retrieval module for RAG.
"""

from abc import ABC, abstractmethod
from typing import List

from schema.retrieval import ScoredChunk

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve() -> List[ScoredChunk]:
        pass

class BaseReranker(ABC):
    @abstractmethod
    def rerank() -> List[ScoredChunk]:
        pass

from retrieval.hybrid import HybridRetriever
from retrieval.fusion import RRF
from retrieval.reranking.cross_encoder import CrossEncoderReranker
from retrieval.reranking.cohere import CohereReranker
from retrieval.reranking.flashrank import FlashRankReranker

__all__ = [
    "BaseRetriever",
    "HybridRetriever",
    "RRF",
    "BaseReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "FlashRankReranker"
]
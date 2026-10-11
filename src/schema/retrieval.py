"""
schema/retrieval.py
Data models for data retrieval and ranking.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, ConfigDict

from typing import Any, Dict, Literal

from src.schema.documents import Chunk

class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

CrossEncoderModel = Literal[
    "cross-encoder/ms-marco-MiniLM-L-6-v2", 
    "cross-encoder/ettin-reranker-32m-v1",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-v2-m",
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "mixedbread-ai/mxbai-rerank-large-v2",
    "nvidia/llama-nemotron-rerank-1b-v2",
    "Qwen/Qwen3-Reranker-4B"
]
CohereModel = Literal[
    "rerank-v3.5",
    "rerank-english-v3.0"
]

FlashRankModel = Literal[
    "ms-marco-MiniLM-L-6-v2", "rank-T5-flan"
]
    
class ScoreBreakdown(BaseModel):
    """Detailed score tracking to avoid black-box ranking bugs."""
    model_config = ConfigDict(frozen=True)

    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    cross_encoder_score: float | None = None

class Query(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Dict[str, Any] = Field(default_factory=dict)
    strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID)

class ScoredChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(..., ge=0.0, le=1.0)
    score_breakdown: ScoreBreakdown
    rank: int = Field(..., ge=1)
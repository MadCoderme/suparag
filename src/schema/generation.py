"""
schema/generation.py
Data models for data generation and response creation.
"""

from datetime import datetime
import uuid

from typing import Any, Dict
from pydantic import BaseModel, Field, ConfigDict

from src.schema.retrieval import Query

class ContextBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(..., ge=1)
    max_input_tokens: int = Field(..., ge=1)
    max_output_tokens: int = Field(..., ge=1)
    allow_truncation: bool = Field(default=False)

class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str
    document_id: str
    quoted_text: str = Field(..., min_length=1, description="The text that was quoted from the chunk.")

class RAGResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: Query
    response_text: str
    citations: list[Citation] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now(datetime.timezone.utc))
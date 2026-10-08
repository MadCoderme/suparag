"""
schema/documents.py
Data models for data preprocessing and storage of documents and AST chunks.
"""

import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from datetime import datetime
from typing import Any, Dict

class ChunkType(StrEnum):
    TEXT = "text"
    PARAGRAPH = "paragraph"
    HEADER_BLOCK = "header_block"
    TABLE_ROW = "table_row"
    CODE = "code"
    AST_NODE = "ast_node"

class ChunkingStrategy(StrEnum):
    FIXED_SIZE = "fixed_size"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"
    AST = "ast"
    AGENTIC = "agentic"

class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    created_at: datetime = Field(default_factory=datetime.now(datetime.timezone.utc))
    mime_type: str = Field(default="text/plain")
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str

    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.FIXED_SIZE)

    content: str

    token_count: int = Field(..., ge = 0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevant_questions: list[str]
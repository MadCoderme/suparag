"""
storage/base.py
Base storage interface for document and chunk storage.
"""

from abc import ABC, abstractmethod

from schema.retrieval import ScoredChunk

class BaseSparseStore(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def store_document(self, document):
        pass
    
    @abstractmethod
    async def query_documents(self, query) -> ScoredChunk:
        pass


class BaseVectorStore(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def store_document(self, document):
        pass
    
    @abstractmethod
    async def query_documents(self, query) -> ScoredChunk:
        pass

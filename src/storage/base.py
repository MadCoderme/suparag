"""
storage/base.py
Base storage interface for document and chunk storage.
"""

from schema.retrieval import ScoredChunk

class BaseSparseStore:
    def __init__(self):
        pass

    def store_document(self, document):
        return
    async def query_documents(self, query) -> ScoredChunk:
        return


class BaseVectorStore:
    def __init__(self):
        pass

    def store_document(self, document):
        return
    async def query_documents(self, query) -> ScoredChunk:
        return

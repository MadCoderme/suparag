"""
retrieval/hybrid.py
Sparse + Dense hybrid orchestration for retrieval.
"""

import asyncio
from typing import List

from retrieval import BaseRetriever
from storage.base import BaseSparseStore, BaseVectorStore
from retrieval.fusion import RRF
from schema.retrieval import ScoredChunk

class HybridRetriever(BaseRetriever):
    def __init__(self, sparse_storage: BaseSparseStore, dense_storage: BaseVectorStore, rrf: RRF = RRF()):
        self.sparse_storage = sparse_storage
        self.dense_storage = dense_storage
        self.rrf = rrf

    def retrieve(self, query) -> List[ScoredChunk]:
        if not self.sparse_storage and self.dense_storage:
            return self.dense_storage.query_documents(query)
        elif not self.dense_storage and self.sparse_storage:
            return self.sparse_storage.query_documents(query)
        
        sparse_results, dense_results = asyncio.gather(
            self.sparse_storage.query_documents(query), self.dense_storage.query_documents(query)
        )
        combined_results = self.rrf.fuse(sparse_results, dense_results)
        return combined_results
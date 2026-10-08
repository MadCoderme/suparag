"""
retrieval/hybrid.py
Sparse + Dense hybrid orchestration for retrieval.
"""

import asyncio

from storage.base import BaseSparseStore, BaseVectorStore
from retrieval.fusion import RRF

class HybridRetriever:
    def __init__(self, sparse_storage: BaseSparseStore, dense_storage: BaseVectorStore, rrf: RRF = RRF()):
        self.sparse_storage = sparse_storage
        self.dense_storage = dense_storage
        self.rrf = rrf

    def retrieve(self, query):
        sparse_results, dense_results = asyncio.gather(
            self.sparse_storage.query_documents(query), self.dense_storage.query_documents(query)
        )
        combined_results = self.rrf.fuse(sparse_results, dense_results)
        return combined_results
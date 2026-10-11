"""
engine/pipeline.py
Functional flow of RAG pipeline
"""

from retrieval import BaseRetriever, BaseReranker
from generation import BaseGenerator
from schema.retrieval import Query
from schema.generation import RAGResponse

class RAGEngine():
    def __init__(self, retriever : BaseRetriever, reranker: BaseReranker, generator: BaseGenerator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def query(self, query: Query) -> RAGResponse:
        scored_chunks = self.retriever.retrieve(query)
        if self.reranker:
            scored_chunks = self.reranker.rerank(query.query_text, scored_chunks)
        response = self.generator.generate(query, scored_chunks)
        return response
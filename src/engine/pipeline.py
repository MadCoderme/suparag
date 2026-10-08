"""
engine/pipeline.py
Functional flow of RAG pipeline
"""

from schema.retrieval import Query
from schema.generation import RAGResponse

class RAGEngine():
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def query(self, query: Query) -> RAGResponse:
        scored_chunks = self.retriever.retrieve(query)
        response = self.generator.generate(query, scored_chunks)
        return response
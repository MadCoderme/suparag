from src.engine.pipeline import RAGEngine
from src.retrieval import HybridRetriever, CrossEncoderReranker
from src.generation import GeminiGenerator

from src.schema.retrieval import Query

if __name__ == "__main__":
    # Initialize the components of the RAG pipeline
    sparse_storage = ... 
    dense_storage = ... 
    retriever = HybridRetriever(sparse_storage, dense_storage)
    
    reranker = CrossEncoderReranker(model_name="ms-marco-MiniLM-L-6-v2")
    generator = GeminiGenerator(model_name="gemini-3.5-flash")

    # Create the RAG engine
    rag_engine = RAGEngine(retriever=retriever, reranker=reranker, generator=generator)

    # Example query
    query_text = "What is the capital of France?"
    query = Query(query_text=query_text)

    # Get the RAG response
    response = rag_engine.query(query)
    
    print("Response Text:", response.response_text)
    print("Citations:", response.citations)

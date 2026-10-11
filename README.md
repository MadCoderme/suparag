# SupaRAG

SupaRAG is a complete modular RAG framework that aims to provide a modular API for building production grade RAG architecture flexibly based on latest industry practices and research findings.

## Quick read

SupaRAG is currently in its early framework-building phase. The project already includes the core building blocks for a modular RAG system: structured schema models for queries, chunks, and responses; a reusable engine pipeline that wires retrieval, reranking, and generation together; hybrid retrieval abstractions with reciprocal-rank fusion; and provider-specific generation adapters such as Google Gemini.

The focus right now is on clean interfaces and replaceable components rather than a single end-to-end app. Storage, retrieval, and generation are separated into explicit layers so the framework can grow toward more advanced sparse/dense retrieval, reranking, and evaluation workflows without being hard-wired to one implementation.

This is still a foundational, work-in-progress codebase rather than a complete production package, but the architectural shape is already in place.

## Inspiration

AI has sped up my development process but the joy of writing code is lost. So, once in a while, I take the initiative to write complete projects all by my own hand (that became a flex too, damn) without using AI. SupaRAG is such an initiative to solidify everything I know about RAG into a nice little framework.

The docs are mostly AI generated though, that part is still too boring to completely write myself.
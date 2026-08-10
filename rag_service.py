"""Shared query-expansion + reranking RAG service used by both web apps."""

import os
from functools import lru_cache

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

load_dotenv(override=True)

from answer_chain import build_answer_chain, format_context
from query_expansion import expand_query
from reranker import rerank
from retriever import get_retriever


@lru_cache(maxsize=1)
def get_pipeline():
    """Load expensive models/indexes once per process."""
    return get_retriever(), build_answer_chain()


def _document_key(document) -> tuple:
    metadata = document.metadata or {}
    return (
        metadata.get("source", ""),
        metadata.get("page", 0),
        document.page_content,
    )


def _source(document) -> dict:
    metadata = document.metadata or {}
    source = os.path.basename(metadata.get("source", "unknown"))
    return {
        "source": source,
        "page": int(metadata.get("page", 0)) + 1,
        "text": document.page_content,
    }


def answer_question(
    question: str,
    num_expansions: int = 5,
    top_n: int = 5,
) -> dict:
    """Run expansion, retrieval, deduplication, reranking, and generation."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if num_expansions not in (0, 1):
        raise ValueError("num_expansions must be 0 or 1 for HyDE.")
    if not 1 <= top_n <= 20:
        raise ValueError("top_n must be between 1 and 20.")

    retriever, answer_chain = get_pipeline()
    queries = expand_query(question, num_expansions=num_expansions)
    hypothetical_document = queries[1] if len(queries) > 1 else ""

    candidates = []
    seen = set()
    for query in queries:
        for document in retriever.invoke(query):
            key = _document_key(document)
            if key not in seen:
                seen.add(key)
                candidates.append(document)

    reranked = rerank(question, candidates, top_n=top_n)
    answer = answer_chain.invoke({
        "context": format_context(reranked),
        "question": question,
    })

    return {
        "question": question,
        "answer": answer,
        "expanded_queries": queries,
        "hypothetical_document": hypothetical_document,
        "candidate_count": len(candidates),
        "sources": [_source(document) for document in reranked],
    }

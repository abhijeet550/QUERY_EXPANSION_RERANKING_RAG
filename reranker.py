"""FlashRank reranking stage.

The pipeline contract is unchanged: receive candidate LangChain Documents,
score them against the original question, and return the best documents. The
model has changed from a sentence-transformers cross-encoder to FlashRank's
lightweight rank-T5-flan reranker.
"""

import os
from functools import lru_cache

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rich.console import Console

console = Console()
RERANKER_MODEL = "rank-T5-flan"
TOP_N = 5


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load FlashRank once per process."""
    from flashrank import Ranker

    console.print(f"Loading FlashRank model '{RERANKER_MODEL}'...")
    ranker = Ranker(model_name=RERANKER_MODEL)
    console.print("FlashRank model loaded.")
    return ranker


def rerank(query: str, chunks: list, top_n: int = TOP_N, show_comparison: bool = True) -> list:
    """Rerank candidate Documents with FlashRank and return the top results."""
    if not chunks:
        return []

    from flashrank import RerankRequest

    passages = [
        {"id": index, "text": chunk.page_content}
        for index, chunk in enumerate(chunks)
    ]
    results = get_reranker_model().rerank(
        RerankRequest(query=query, passages=passages)
    )

    reranked = []
    for result in results[:top_n]:
        result_id = int(result["id"] if isinstance(result, dict) else result.id)
        reranked.append(chunks[result_id])

    if show_comparison:
        console.print(
            f"FlashRank reranked {len(chunks)} candidate(s); "
            f"keeping top {len(reranked)}."
        )
    return reranked


def main():
    from answer_chain import build_answer_chain, format_context
    from retriever import get_retriever

    query = input("Enter your query: ").strip()
    chunks = get_retriever().invoke(query)
    reranked_chunks = rerank(query, chunks, top_n=TOP_N)
    answer = build_answer_chain().invoke({
        "context": format_context(reranked_chunks),
        "question": query,
    })
    console.print(answer)


if __name__ == "__main__":
    main()

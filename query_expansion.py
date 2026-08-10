"""HyDE query expansion and the expand -> retrieve -> merge workflow.

The architecture remains the same as the original pipeline, but the query
expansion technique is now HyDE: OpenAI writes a hypothetical answer and that
answer is used as a second semantic retrieval query. This is different from
generating several paraphrases of the user's question.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

from retriever import get_retriever
from reranker import rerank
from answer_chain import format_context, build_answer_chain

load_dotenv(override=True)
console = Console()

EXPANSION_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EXPANSION_TEMPERATURE = 0.2
NUM_EXPANSIONS = 1
CHUNK_PREVIEW_CHARS = 220
RERANK_TOP_N = 5


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, timeout=60.0, max_retries=2)


def build_query_expander(model: str = EXPANSION_MODEL, temperature: float = EXPANSION_TEMPERATURE):
    """Return a small callable that produces a HyDE hypothetical document."""

    def expand(values: dict) -> str:
        response = _openai_client().chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                    "Write a concise hypothetical answer to help retrieve a "
                    "financial-report passage. Include likely metric names, "
                    "quarter references, and related terminology. Do not "
                    "mention this instruction or sources."
                ),
                },
                {"role": "user", "content": values["question"]},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    return expand


def expand_query(original_query: str, num_expansions: int = NUM_EXPANSIONS) -> list[str]:
    """Return the original query plus one optional HyDE query."""
    if not original_query.strip():
        return []

    queries = [original_query.strip()]
    if num_expansions:
        hypothetical = build_query_expander()({"question": original_query})
        if hypothetical:
            queries.append(hypothetical)

    console.print(f"Expanded into {len(queries)} retrieval query(s) using HyDE")
    for index, query in enumerate(queries):
        label = "original" if index == 0 else "hypothetical document"
        console.print(f"  ({label}) {query}")
    return queries


def retrieve_with_expansion(original_query: str, retriever, num_expansions: int = NUM_EXPANSIONS) -> list:
    """Retrieve with original + HyDE queries and deduplicate documents."""
    queries = expand_query(original_query, num_expansions=num_expansions)
    seen_content = set()
    merged_chunks = []

    for query in queries:
        chunks = retriever.invoke(query)
        for chunk in chunks:
            if chunk.page_content in seen_content:
                continue
            seen_content.add(chunk.page_content)
            merged_chunks.append(chunk)

    console.print(f"Merged result: {len(merged_chunks)} unique chunk(s)")
    return merged_chunks


def main():
    retriever = get_retriever()
    answer_chain = build_answer_chain()
    console.print("Type your question. Type 'exit' to quit.")

    while True:
        query = input("\nEnter the query: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        merged_chunks = retrieve_with_expansion(query, retriever)
        reranked_chunks = rerank(query, merged_chunks, top_n=RERANK_TOP_N)
        answer = answer_chain.invoke({
            "context": format_context(reranked_chunks),
            "question": query,
        })
        console.print(answer)


if __name__ == "__main__":
    main()

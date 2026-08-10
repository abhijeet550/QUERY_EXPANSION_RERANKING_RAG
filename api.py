"""FastAPI server for the query-expansion + reranking RAG pipeline."""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_service import answer_question, get_pipeline


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Query Expansion Reranking RAG API",
    version="1.0.0",
    description="Answers questions over the indexed quarterly financial reports.",
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    num_expansions: int = Field(1, ge=0, le=1, description="0 or 1 HyDE retrieval query")
    top_n: int = Field(5, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    try:
        return answer_question(
            request.question,
            num_expansions=request.num_expansions,
            top_n=request.top_n,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/warmup")
def warmup() -> dict:
    """Load the FAISS index and answer model before the first user query."""
    try:
        get_pipeline()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

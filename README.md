# Query Expansion + Reranking RAG

This project answers questions over the quarterly PDF reports with:

1. FAISS vector retrieval
2. HyDE query expansion: Grok generates a hypothetical answer used as a second retrieval query
3. FlashRank reranking with the lightweight `rank-T5-flan` model
4. Grounded answer generation
5. FastAPI and Streamlit interfaces

The active web pipeline is implemented in `rag_service.py`, with query
expansion in `query_expansion.py` and reranking in `reranker.py`.

## Setup

```bash
cd /Users/abhijssi/Documents/Codex/query_expansion_reranking_rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in `.env`, then build the index once. The index uses local Hugging Face embeddings, so only an OpenAI key is required:

```env
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

Build the FAISS index from the PDFs in the project directory:

```bash
python ingest.py
```

Start the API:

```bash
uvicorn api:app --reload --port 8000
```

In another terminal, start Streamlit:

```bash
streamlit run streamlit_app.py
```

The API exposes `GET /health`, `POST /query`, and `POST /warmup`. Swagger documentation is available at `/docs`.

"""Streamlit UI for the FastAPI-backed RAG application."""

import os

import requests
import streamlit as st


st.set_page_config(page_title="Financial Reports RAG", page_icon="📚", layout="wide")
st.title("📚 Financial Reports RAG")
st.caption("HyDE expands the query; FlashRank reranking improves precision.")

api_url = st.sidebar.text_input(
    "FastAPI URL",
    value=os.getenv("RAG_API_URL", "http://localhost:8000"),
).rstrip("/")
num_expansions = st.sidebar.slider("Use HyDE expansion", 0, 1, 1)
top_n = st.sidebar.slider("Sources in answer", 1, 10, 5)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['source']} — page {source['page']}**")
                    st.caption(source["text"][:500])

question = st.chat_input("Ask about revenue, EBITDA, profit, or another report metric")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Expanding query, retrieving, reranking, and answering…"):
            try:
                response = requests.post(
                    f"{api_url}/query",
                    json={
                        "question": question,
                        "num_expansions": num_expansions,
                        "top_n": top_n,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
                st.markdown(result["answer"])
                with st.expander("Retrieval details"):
                    st.write(f"{result['candidate_count']} unique candidates after HyDE expansion")
                    st.write(result["expanded_queries"])
                    if result.get("hypothetical_document"):
                        st.caption("Hypothetical document used for semantic retrieval")
                        st.write(result["hypothetical_document"])
                with st.expander("Sources"):
                    for source in result["sources"]:
                        st.markdown(f"**{source['source']} — page {source['page']}**")
                        st.caption(source["text"][:500])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"FastAPI returned an error: {detail}")
            except requests.ConnectionError as exc:
                st.error(f"Could not reach the FastAPI server: {exc}")
            except requests.RequestException as exc:
                st.error(f"Request to FastAPI failed: {exc}")
            except Exception as exc:
                st.error(f"RAG request failed: {exc}")

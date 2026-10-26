# RAG-based LLM Assistant

Portfolio project demonstrating retrieval-augmented generation (RAG): document
chunking, TF-IDF retrieval, grounded prompt construction, and an optional
generative answer step with any OpenAI-compatible LLM.

> **Data note:** All documents in this project are **synthetic** sample docs
> (fictional product help articles). No real company documentation, customer
> data, or production code is used.

## Problem

LLMs hallucinate when asked about facts they were not trained on. RAG fixes
this by retrieving relevant passages from a trusted knowledge base first, then
asking the model to answer **only from that retrieved context**, with sources
cited.

## Approach

1. **Chunking** — Long docs are split into overlapping word windows so no
   answer context is cut mid-thought and each chunk stays topically focused.
2. **Retrieval** — TF-IDF vectors over chunks, ranked by cosine similarity.
   Runs fully offline with no API key. (Swap in embeddings later; the
   retriever interface is deliberately small so that change is one file.)
3. **Grounded answering** — The top chunks are assembled into a prompt that
   instructs the model to answer from the context and say "I don't know" when
   the context does not contain the answer.
4. **Optional LLM step** — If `OPENAI_API_KEY` is set and the `openai`
   package is installed, `qa.py` calls a chat model; otherwise it returns the
   retrieved passages as an extractive answer. The pipeline never pretends an
   extractive fallback is a generated answer.

## How to run

```bash
pip install -r requirements.txt
python src/qa.py "How do I set up SSO?"
python src/qa.py "What is the data retention policy?"
```

## Project structure

```text
rag-llm-assistant/
├── README.md
├── requirements.txt
├── src/
│   ├── retriever.py   # sample docs, chunking, TF-IDF retrieval
│   └── qa.py          # grounded answering (extractive or LLM-backed)
```

## Skills demonstrated

Python, RAG architecture, document chunking, TF-IDF retrieval, prompt
engineering, grounded/guardrailed generation, LangChain-style patterns
(implemented directly here so the retrieval logic is visible, not hidden
behind a framework call).

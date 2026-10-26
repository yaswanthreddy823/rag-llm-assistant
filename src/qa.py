"""Grounded question answering over the sample knowledge base.

Usage:  python src/qa.py "How do I set up SSO?"

Without an LLM API key this returns the retrieved passages (extractive
answer) and says so plainly. With ``OPENAI_API_KEY`` set and the ``openai``
package installed, it asks a chat model to answer strictly from the
retrieved context, and to say it does not know when the context is silent.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import DOCUMENTS, build_index, retrieve

SYSTEM_PROMPT = (
    "Answer the question using ONLY the context below. "
    "If the context does not contain the answer, say you don't know. "
    "Cite the source document title for each fact you use."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{p['title']}] {p['text']}" for p in passages
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"


def llm_answer(prompt: str) -> str | None:
    """Return a generated answer, or None when no LLM is configured."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL) from env
    response = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def answer(question: str) -> str:
    vectorizer, matrix, chunks = build_index(DOCUMENTS)
    passages = retrieve(question, vectorizer, matrix, chunks, top_k=3)
    if not passages or passages[0]["score"] <= 0:
        return "I don't know — nothing in the knowledge base covers this."

    generated = llm_answer(build_prompt(question, passages))
    if generated is not None:
        return generated

    lines = ["(Extractive answer — no LLM key configured; "
             "top retrieved passages:)\n"]
    for p in passages:
        lines.append(f"[{p['title']} (score {p['score']:.2f})] {p['text']}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/qa.py "your question"')
        raise SystemExit(1)
    print(answer(" ".join(sys.argv[1:])))

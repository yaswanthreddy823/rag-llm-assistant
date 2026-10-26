"""Grounded question answering for the banking assistant.

Usage:  python src/qa.py "What documents do I need for KYC?"

Without an LLM API key this returns the retrieved passages (extractive
answer) and says so plainly. With ``OPENAI_API_KEY`` set and the ``openai``
package installed, it asks a chat model to answer strictly from the
retrieved context, and to say it does not know when the context is silent.
This mirrors the production assistant's grounding discipline: the model
never answers banking questions from its own knowledge alone.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import build_index, retrieve

SYSTEM_PROMPT = (
    "You are a banking assistant. Answer the question using ONLY the "
    "context below. If the context does not contain the answer, say you "
    "don't know and suggest contacting support. Cite the source document "
    "title for each fact you use."
)

MIN_SCORE = 0.1


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{p['title']}] {p['text']}" for p in passages
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"


def answer(question: str, top_k: int = 3) -> dict:
    """Answer a question from the knowledge base. Returns answer + sources."""
    index = build_index()
    passages = retrieve(question, index, top_k=top_k)
    passages = [p for p in passages if p["score"] >= MIN_SCORE]
    if not passages:
        return {
            "answer": "I don't know based on the available documents. "
                      "Please contact support for help with this question.",
            "sources": [],
            "mode": "no_context",
        }
    sources = sorted({p["title"] for p in passages})
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, passages)},
                ],
                temperature=0,
            )
            return {
                "answer": response.choices[0].message.content,
                "sources": sources,
                "mode": "generative",
            }
        except Exception as exc:
            print(f"LLM call failed ({exc}); falling back to extractive answer.")
    extractive = "\n\n".join(
        f"From {p['title']}: {p['text']}" for p in passages
    )
    return {
        "answer": extractive,
        "sources": sources,
        "mode": "extractive",
    }


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python src/qa.py "What documents do I need for KYC?"')
        sys.exit(1)
    result = answer(" ".join(sys.argv[1:]))
    print(result["answer"])
    if result["sources"]:
        print(f"\nSources: {', '.join(result['sources'])} "
              f"(mode: {result['mode']})")


if __name__ == "__main__":
    main()

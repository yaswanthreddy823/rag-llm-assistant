"""Document chunking and TF-IDF retrieval for the banking knowledge base.

The knowledge base is a small set of sample banking help articles (KYC,
onboarding, servicing, cards, loans, fraud reporting), written for this
demo in the style of the documents the production assistant retrieved
over. Chunking uses overlapping word windows; retrieval ranks chunks by
cosine similarity against the query.

In production, retrieval used LangChain with embeddings and an LLM
(OpenAI GPT) for generation. This demo keeps the same RAG shape --
retrieve, then ground the answer in the retrieved passages -- with a
dependency-light TF-IDF retriever and an optional LLM step in ``qa.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS = [
    {
        "title": "KYC verification",
        "text": (
            "To verify your identity (KYC), provide a government-issued photo "
            "ID such as a passport or driver's license, proof of address such "
            "as a utility bill or bank statement from the last 90 days, and "
            "your Social Security number or tax ID. Upload clear photos of "
            "each document in online banking under Profile, then Verification. "
            "Most verifications complete within one business day; if a "
            "document is rejected you will see the reason and can re-upload. "
            "Accounts stay limited until verification is complete."
        ),
    },
    {
        "title": "Account onboarding",
        "text": (
            "To open a checking or savings account, complete the application "
            "with your personal details, pass KYC verification, and fund the "
            "account with an initial deposit by transfer, check, or cash at a "
            "branch. Your debit card arrives by mail within 7 to 10 business "
            "days; activate it in online banking or by phone. Enroll in "
            "online banking during onboarding to view statements, set up "
            "direct deposit, and manage alerts from day one."
        ),
    },
    {
        "title": "Account servicing",
        "text": (
            "You can update your mailing address, phone number, and email in "
            "online banking under Profile, then Contact Details; changes take "
            "effect immediately. To add or change a beneficiary, visit a "
            "branch with photo ID. To close an account, withdraw or transfer "
            "the remaining balance first, then submit the closure request in "
            "online banking or at a branch. Joint owners can be added with "
            "both parties present and verified."
        ),
    },
    {
        "title": "Cards and payments",
        "text": (
            "Activate a new debit or credit card in online banking or by "
            "phone before first use. Daily card limits can be raised "
            "temporarily in the app under Cards, then Limits. Set a travel "
            "notice before trips abroad so legitimate transactions are not "
            "declined. To dispute a transaction, open it in your activity "
            "feed and choose Report a problem within 60 days of the statement "
            "date; provisional credit is issued while the dispute is reviewed."
        ),
    },
    {
        "title": "Loans and credit",
        "text": (
            "A personal loan application asks for income, employment, existing "
            "debts, and the loan amount and purpose. Decisions typically take "
            "one to three business days after all documents are received. "
            "Rates depend on credit history and loan term; autopay from a "
            "checking account may reduce the rate. You can check your "
            "application status in online banking under Loans, then "
            "Applications."
        ),
    },
    {
        "title": "Reporting fraud",
        "text": (
            "If you see a transaction you do not recognize, freeze the card "
            "immediately in the app under Cards, then Freeze, and report the "
            "transaction from your activity feed. Fraud claims are reviewed "
            "within 10 business days and provisional credit is issued in most "
            "cases. Never share one-time codes or passwords with anyone "
            "claiming to be from the bank; the bank will never ask for them."
        ),
    },
]

CHUNK_WORDS = 60
CHUNK_OVERLAP = 15


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + CHUNK_WORDS])
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def build_index():
    """Chunk all documents and fit a TF-IDF vectorizer. Returns an index."""
    chunks = []
    for doc in DOCUMENTS:
        for chunk in chunk_text(doc["text"]):
            chunks.append({"title": doc["title"], "text": chunk})
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}


def retrieve(query: str, index: dict, top_k: int = 3) -> list[dict]:
    """Return the top_k chunks most similar to the query, with scores."""
    query_vec = index["vectorizer"].transform([query])
    sims = cosine_similarity(query_vec, index["matrix"])[0]
    order = np.argsort(sims)[::-1][:top_k]
    return [
        {**index["chunks"][i], "score": float(sims[i])}
        for i in order
        if sims[i] > 0
    ]

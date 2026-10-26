"""Document chunking and TF-IDF retrieval for a small knowledge base.

The sample documents are fictional product help articles, written for this
demo. Chunking uses overlapping word windows; retrieval ranks chunks by
cosine similarity against the query.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS = [
    {
        "title": "Setting up SSO",
        "text": (
            "To set up single sign-on, go to Settings, then Security, then "
            "Single Sign-On. Upload your identity provider metadata file and "
            "copy the assertion consumer service URL back into your provider. "
            "Supported providers include Okta, Azure AD, and Google Workspace. "
            "After saving, test the connection with a single user before "
            "enforcing SSO for the whole workspace, because enforcing it "
            "immediately locks out password-only accounts."
        ),
    },
    {
        "title": "Data retention policy",
        "text": (
            "Customer data is retained for 90 days after account closure and "
            "then permanently deleted. Audit logs are kept for one year for "
            "security review. Backups roll over every 30 days, so a deleted "
            "account cannot be restored from backup after the 90-day window. "
            "To request early deletion, contact support with account ownership "
            "verification."
        ),
    },
    {
        "title": "API rate limits",
        "text": (
            "The public API allows 100 requests per minute per API key, with "
            "bursts up to 150. Responses include X-RateLimit-Remaining and "
            "X-RateLimit-Reset headers. When the limit is reached the API "
            "returns HTTP 429; clients should back off exponentially and retry. "
            "Higher limits are available on the Enterprise plan."
        ),
    },
    {
        "title": "Billing and refunds",
        "text": (
            "Plans are billed monthly in advance. You can cancel at any time "
            "and keep access until the end of the current billing period. "
            "Refunds are issued for the unused portion when cancellation happens "
            "within 14 days of a plan change. Annual plans are refunded pro-rata "
            "within the first 30 days."
        ),
    },
    {
        "title": "Exporting your data",
        "text": (
            "Workspace admins can export all workspace data from Settings, then "
            "Data, then Export. Exports are delivered as a ZIP of CSV files "
            "within 24 hours. Exports include tasks, comments, and attachments, "
            "but not deleted items or audit logs."
        ),
    },
]


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_index(documents: list[dict]) -> tuple[TfidfVectorizer, np.ndarray, list[dict]]:
    """Return (vectorizer, chunk matrix, chunk records with title + text)."""
    chunks: list[dict] = []
    for doc in documents:
        for piece in chunk_text(doc["text"]):
            chunks.append({"title": doc["title"], "text": piece})
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return vectorizer, matrix, chunks


def retrieve(query: str, vectorizer: TfidfVectorizer, matrix: np.ndarray,
             chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Return the top_k chunks by cosine similarity to the query."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    order = np.argsort(scores)[::-1][:top_k]
    return [
        {"title": chunks[i]["title"], "text": chunks[i]["text"],
         "score": float(scores[i])}
        for i in order
    ]

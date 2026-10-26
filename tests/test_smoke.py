"""Smoke tests for the banking assistant: retrieval, QA, KYC flow.

Run:  python -m pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flows import KYCFlow, KYC_STEPS
from qa import answer
from retriever import build_index, retrieve


def test_retrieval_finds_kyc_document():
    index = build_index()
    top = retrieve("What documents do I need for KYC verification?",
                   index, top_k=3)
    assert top, "no passages retrieved"
    assert top[0]["title"] == "KYC verification"


def test_retrieval_finds_fraud_document():
    index = build_index()
    top = retrieve("I see a transaction I do not recognize, what do I do?",
                   index, top_k=3)
    assert top[0]["title"] == "Reporting fraud"


def test_qa_answer_cites_source():
    result = answer("How do I activate a new debit card?")
    assert result["sources"], "answer has no sources"
    assert "Cards and payments" in result["sources"]
    assert result["mode"] in ("extractive", "generative")


def test_qa_unknown_question_says_so():
    result = answer("What is the weather on Mars?")
    assert result["mode"] == "no_context"
    assert "don't know" in result["answer"]


def test_kyc_flow_completes():
    flow = KYCFlow()
    assert flow.status()["next_step"] == KYC_STEPS[0]
    for step in KYC_STEPS:
        status = flow.complete(step)
    assert status["complete"] is True
    assert status["remaining"] == []
    assert status["next_step"] is None

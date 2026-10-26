"""Chat API for the banking assistant (demonstration).

Endpoints:
  GET  /health
  POST /ask   {"question": "..."} -> grounded answer + sources
  POST /kyc   {"session_id": "...", "step": "..."} -> checklist status

Run:  python src/app.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request

from flows import KYCFlow
from qa import answer

app = Flask(__name__)
_sessions: dict[str, KYCFlow] = {}


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "missing 'question'"}), 400
    return jsonify(answer(question))


@app.post("/kyc")
def kyc():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "default")
    step = data.get("step")
    flow = _sessions.setdefault(session_id, KYCFlow())
    if step:
        try:
            return jsonify(flow.complete(step))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    return jsonify(flow.status())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

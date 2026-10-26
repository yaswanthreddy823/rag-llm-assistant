"""Guided onboarding flows for the banking assistant.

Production assistants did more than answer questions: they walked
customers through multi-step processes. This module models the KYC
onboarding checklist as a small state machine -- the assistant tracks
which steps are done, reports what remains, and tells the customer the
next step. In production this state lived in the conversation service;
here it is in-memory for demonstration.
"""

from __future__ import annotations

KYC_STEPS = [
    "identity_document",
    "address_proof",
    "tax_id",
    "employment_info",
    "risk_profile",
]

STEP_GUIDANCE = {
    "identity_document": "Upload a government-issued photo ID "
                         "(passport or driver's license).",
    "address_proof": "Upload a utility bill or bank statement "
                     "from the last 90 days.",
    "tax_id": "Provide your Social Security number or tax ID.",
    "employment_info": "Provide your employment status and annual income.",
    "risk_profile": "Complete the short risk-profile questionnaire.",
}


class KYCFlow:
    """Track one customer's progress through KYC onboarding."""

    def __init__(self) -> None:
        self.done: set[str] = set()

    def complete(self, step: str) -> dict:
        if step not in KYC_STEPS:
            raise ValueError(f"unknown KYC step: {step}")
        self.done.add(step)
        return self.status()

    def status(self) -> dict:
        remaining = [s for s in KYC_STEPS if s not in self.done]
        return {
            "completed": [s for s in KYC_STEPS if s in self.done],
            "remaining": remaining,
            "next_step": remaining[0] if remaining else None,
            "next_guidance": STEP_GUIDANCE[remaining[0]] if remaining else None,
            "complete": not remaining,
        }

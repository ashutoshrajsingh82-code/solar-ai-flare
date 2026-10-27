"""Validation helpers for scientific-semantics blocker registers.

This module manages inquiry workflow state only.  It cannot adjudicate a
scientific rule or alter the Phase 3C gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


ALLOWED_STATUSES = {"OPEN", "REQUEST_PREPARED", "RESPONSE_RECEIVED", "RESOLVED", "UNAVAILABLE"}
ALLOWED_TRANSITIONS = {
    "OPEN": {"REQUEST_PREPARED", "UNAVAILABLE"},
    "REQUEST_PREPARED": {"RESPONSE_RECEIVED", "UNAVAILABLE"},
    "RESPONSE_RECEIVED": {"RESOLVED", "REQUEST_PREPARED", "UNAVAILABLE"},
    "RESOLVED": set(),
    "UNAVAILABLE": {"REQUEST_PREPARED"},
}
REQUIRED_FIELDS = {
    "blocker_id", "affected_rule_ids", "question", "existing_evidence", "insufficient_evidence",
    "required_response_or_artifact", "official_contact_or_access_route", "status",
    "adjudication_criteria", "audit_history",
}
REQUEST_STATUSES = {"REQUEST_PREPARED", "REQUEST_SENT", "ACKNOWLEDGED", "RESPONSE_RECEIVED", "UNDER_ADJUDICATION", "RESOLVED", "UNAVAILABLE"}
REQUEST_TRANSITIONS = {
    "REQUEST_PREPARED": {"REQUEST_SENT", "UNAVAILABLE"},
    "REQUEST_SENT": {"ACKNOWLEDGED", "RESPONSE_RECEIVED", "UNAVAILABLE"},
    "ACKNOWLEDGED": {"RESPONSE_RECEIVED", "UNAVAILABLE"},
    "RESPONSE_RECEIVED": {"UNDER_ADJUDICATION", "UNAVAILABLE"},
    "UNDER_ADJUDICATION": {"RESOLVED", "REQUEST_SENT", "UNAVAILABLE"},
    "RESOLVED": set(), "UNAVAILABLE": {"REQUEST_PREPARED"},
}
REQUEST_REQUIRED_FIELDS = {"request_id", "blocker_ids", "rule_ids", "draft_artifact", "draft_sha256", "status", "adjudication_status", "audit_history"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_register(register: dict) -> None:
    if register.get("schema_version") != "phase4k_blocker_register_v1":
        raise ValueError("unsupported blocker-register schema")
    seen = set()
    for row in register.get("blockers", []):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"blocker missing fields: {sorted(missing)}")
        blocker_id = row["blocker_id"]
        if blocker_id in seen:
            raise ValueError(f"duplicate blocker ID: {blocker_id}")
        seen.add(blocker_id)
        if not isinstance(row["affected_rule_ids"], list) or not row["affected_rule_ids"]:
            raise ValueError(f"{blocker_id}: affected_rule_ids must be nonempty")
        if row["status"] not in ALLOWED_STATUSES:
            raise ValueError(f"{blocker_id}: invalid status {row['status']}")
        if row["status"] == "RESOLVED" and not row.get("qualifying_response"):
            raise ValueError(f"{blocker_id}: RESOLVED requires qualifying_response provenance")


def validate_request_register(register: dict, blocker_ids: set[str] | None = None) -> None:
    """Validate Phase 4L operational request tracking, separately from science."""
    if register.get("schema_version") != "phase4l_request_register_v1":
        raise ValueError("unsupported request-register schema")
    seen = set()
    for request in register.get("requests", []):
        missing = REQUEST_REQUIRED_FIELDS - set(request)
        if missing:
            raise ValueError(f"request missing fields: {sorted(missing)}")
        request_id = request["request_id"]
        if request_id in seen:
            raise ValueError(f"duplicate request ID: {request_id}")
        seen.add(request_id)
        if request["status"] not in REQUEST_STATUSES:
            raise ValueError(f"{request_id}: invalid request status")
        if not request["blocker_ids"] or not request["rule_ids"]:
            raise ValueError(f"{request_id}: blocker_ids and rule_ids must be nonempty")
        if blocker_ids is not None and not set(request["blocker_ids"]).issubset(blocker_ids):
            raise ValueError(f"{request_id}: unknown blocker ID")
        digest = request["draft_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"{request_id}: invalid draft SHA-256")
        if request["status"] == "REQUEST_SENT":
            submission = request.get("submission")
            if not submission or not submission.get("recipient_verified") or not submission.get("evidence"):
                raise ValueError(f"{request_id}: REQUEST_SENT requires verified recipient and submission evidence")
        if request["status"] == "RESOLVED":
            resolution = request.get("canonical_adjudication")
            if not resolution or not resolution.get("successful_phase3c_run"):
                raise ValueError(f"{request_id}: RESOLVED requires successful canonical adjudication")


def transition_request(request: dict, target: str, *, submission: dict | None = None,
                       canonical_adjudication: dict | None = None) -> dict:
    current = request["status"]
    if target not in REQUEST_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid request transition: {current} -> {target}")
    if target == "REQUEST_SENT" and (not submission or not submission.get("recipient_verified") or not submission.get("evidence")):
        raise ValueError("REQUEST_SENT requires verified recipient and submission evidence")
    if target == "RESOLVED" and (not canonical_adjudication or not canonical_adjudication.get("successful_phase3c_run")):
        raise ValueError("RESOLVED requires successful canonical adjudication")
    updated = dict(request); updated["status"] = target
    if submission: updated["submission"] = submission
    if canonical_adjudication: updated["canonical_adjudication"] = canonical_adjudication
    return updated


def validate_request_artifacts(register: dict, repo_root: Path) -> None:
    """Fail closed when a tracked draft is missing or has changed after hashing."""
    for request in register["requests"]:
        artifact = repo_root / "backend" / request["draft_artifact"]
        if not artifact.is_file():
            raise FileNotFoundError(f"{request['request_id']}: draft artifact missing: {artifact}")
        if sha256(artifact).lower() != request["draft_sha256"].lower():
            raise ValueError(f"{request['request_id']}: draft artifact hash mismatch")


def transition(blocker: dict, target: str, *, qualifying_response: dict | None = None) -> dict:
    """Return an updated workflow record; never modify a Phase 3C rule."""
    current = blocker["status"]
    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid blocker transition: {current} -> {target}")
    if target == "RESOLVED" and not qualifying_response:
        raise ValueError("RESOLVED requires qualifying response provenance and adjudication")
    updated = dict(blocker)
    updated["status"] = target
    if qualifying_response:
        updated["qualifying_response"] = qualifying_response
    return updated

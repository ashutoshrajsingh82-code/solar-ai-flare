import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "blocker_register.py"
SPEC = importlib.util.spec_from_file_location("blocker_register", MODULE_PATH)
blockers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = blockers
SPEC.loader.exec_module(blockers)


def _register():
    return json.loads((REPO_ROOT / "backend" / "reports" / "phase4k_blocker_register.json").read_text(encoding="utf-8"))


def test_blocker_register_is_valid_and_traceable_to_unresolved_rule_ids():
    register = _register()
    blockers.validate_register(register)
    assert len(register["blockers"]) == 6
    assert {rule for row in register["blockers"] for rule in row["affected_rule_ids"]} == {
        "R03", "R05", "R06", "R07", "R08", "R13", "R14"}
    assert all(row["status"] == "REQUEST_PREPARED" for row in register["blockers"])


def test_blocker_status_transitions_require_provenanced_adjudication_for_resolution():
    row = _register()["blockers"][0]
    received = blockers.transition(row, "RESPONSE_RECEIVED")
    assert received["status"] == "RESPONSE_RECEIVED"
    with pytest.raises(ValueError, match="qualifying response"):
        blockers.transition(received, "RESOLVED")
    resolved = blockers.transition(received, "RESOLVED", qualifying_response={
        "path": "E:/SolarFlareData/source_docs/isro/solexs/reply.pdf",
        "sha256": "a" * 64,
        "issuer_verified": True,
        "citation": "p. 1",
        "adjudication_run": "required",
    })
    assert resolved["status"] == "RESOLVED"


def test_blocker_register_rejects_duplicate_ids_and_unprovenanced_resolved_status():
    register = _register()
    duplicate = dict(register["blockers"][0])
    register["blockers"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate blocker ID"):
        blockers.validate_register(register)
    register = _register()
    register["blockers"][0]["status"] = "RESOLVED"
    with pytest.raises(ValueError, match="qualifying_response"):
        blockers.validate_register(register)


def test_phase4l_request_tracking_rejects_unsent_draft_and_preserves_traceability():
    request_register = json.loads((REPO_ROOT / "backend" / "reports" / "phase4l_request_register.json").read_text(encoding="utf-8"))
    blocker_register = _register()
    blockers.validate_request_register(request_register, {row["blocker_id"] for row in blocker_register["blockers"]})
    request = request_register["requests"][0]
    with pytest.raises(ValueError, match="submission evidence"):
        blockers.transition_request(request, "REQUEST_SENT")
    sent = blockers.transition_request(request, "REQUEST_SENT", submission={"recipient_verified": True, "evidence": "saved confirmation"})
    assert sent["status"] == "REQUEST_SENT"
    assert set(request["rule_ids"]).issubset({"R03", "R05", "R06", "R07", "R08", "R13", "R14"})


def test_phase4l_request_tracking_rejects_unknown_blockers_and_unadjudicated_resolution():
    request_register = json.loads((REPO_ROOT / "backend" / "reports" / "phase4l_request_register.json").read_text(encoding="utf-8"))
    request_register["requests"][0]["blocker_ids"] = ["not-real"]
    with pytest.raises(ValueError, match="unknown blocker ID"):
        blockers.validate_request_register(request_register, set())
    request = json.loads((REPO_ROOT / "backend" / "reports" / "phase4l_request_register.json").read_text(encoding="utf-8"))["requests"][0]
    request["status"] = "UNDER_ADJUDICATION"
    with pytest.raises(ValueError, match="successful canonical adjudication"):
        blockers.transition_request(request, "RESOLVED")


def test_phase4l_request_draft_hash_and_missing_artifact_fail_closed(tmp_path):
    request_register = json.loads((REPO_ROOT / "backend" / "reports" / "phase4l_request_register.json").read_text(encoding="utf-8"))
    blockers.validate_request_artifacts(request_register, REPO_ROOT)
    request_register["requests"][0]["draft_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        blockers.validate_request_artifacts(request_register, REPO_ROOT)
    request_register = json.loads((REPO_ROOT / "backend" / "reports" / "phase4l_request_register.json").read_text(encoding="utf-8"))
    request_register["requests"][0]["draft_artifact"] = "reports/missing.md"
    with pytest.raises(FileNotFoundError, match="draft artifact missing"):
        blockers.validate_request_artifacts(request_register, REPO_ROOT)

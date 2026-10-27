"""Phase 4V: Multi-Epoch Readiness Gate Evaluator.

Evaluates all mandatory prerequisites for the Phase 4V multi-epoch forecasting pipeline by
inspecting actual files, verified provenance, and machine-readable gate reports.  Exits
nonzero when any hard prerequisite fails.

Gate Modes
----------
PHASE4V_BLOCKED_READINESS_GATE
    One or more development-data prerequisites are unsatisfied (archives missing / hash
    mismatch / dataset contract absent).  Training cannot proceed in any mode.

PHASE4V_DEVELOPMENT_TRAINING_AUTHORIZED
    All development prerequisites pass.  The October holdout is still absent (GATE-4V-07
    fails, non-blocking).  Training is authorized in DEVELOPMENT_ONLY_NOT_FINAL_MODEL
    mode per the Phase 4U holdout governance policy (Epoch 1 + Epoch 2 are explicitly
    "Open for training, feature engineering, and cross-validation").

NOTE: GATE-4V-07 (holdout acquired) is NOT a blocker for DEVELOPMENT_ONLY training.
      It is recorded for audit and must be satisfied before any final-evaluation run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

REPORTS_DIR = backend_dir / "reports"
ART_4V_DIR = backend_dir / "artifacts" / "real_data_experiments" / "phase4v"

# ── Archive provenance ────────────────────────────────────────────────────────
ARCHIVE_MAY2024 = Path(r"C:\Users\ASHUTOSH\Downloads\solexs_2026Oct25T200126271.zip")
ARCHIVE_FEB2026 = Path(r"C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip")
SHA256_MAY2024 = "8df85ed75b6f00052d7f45a6764b187c700977f931504fb79fdfe56f10df9db2"
SHA256_FEB2026 = "34c39624194ee2da95bbf8e3dfed5ac002c55a517a31bc4091bfdd0d2b4671c2"

# October holdout: reserved, not yet acquired
HOLDOUT_EPOCH_ID = "EPOCH_3_RESERVED_UNTOUCHED_HOLDOUT"
HOLDOUT_DATE_RANGE = "2024-10-01 to 2024-10-07"

# Required report / contract files (absence is a hard blocker)
REQUIRED_CONTRACT_FILES: List[Tuple[str, Path]] = [
    ("phase4u_dataset_contract.md", REPORTS_DIR / "phase4u_dataset_contract.md"),
    ("phase4u_holdout_governance.md", REPORTS_DIR / "phase4u_holdout_governance.md"),
    ("phase4u_readiness_gate.json", REPORTS_DIR / "phase4u_readiness_gate.json"),
    ("phase4u_observation_inventory.csv", REPORTS_DIR / "phase4u_observation_inventory.csv"),
]

EXPECTED_DEV_PRODUCTS = 14
EXPECTED_MAY_DATES = {
    "20240508", "20240509", "20240510", "20240511",
    "20240512", "20240513", "20240514", "20240515",
}
EXPECTED_FEB_DATES = {
    "20260131", "20260201", "20260202",
    "20260203", "20260204", "20260205",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _zip_ok(path: Path) -> Tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                return False, f"Corrupt ZIP member: {bad}"
        return True, "ZIP integrity OK"
    except Exception as exc:
        return False, f"ZIP open error: {exc}"


def _cond(cid: str, description: str, satisfied: bool, detail: str) -> Dict[str, Any]:
    return {
        "condition_id": cid,
        "description": description,
        "satisfied": satisfied,
        "detail": detail,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# ── Gate evaluation ───────────────────────────────────────────────────────────

def evaluate_phase4v_gate() -> Dict[str, Any]:
    conditions: List[Dict[str, Any]] = []

    # GATE-4V-01: May 2024 archive present and hash-verified
    ok1 = ARCHIVE_MAY2024.is_file()
    if ok1:
        sha1 = _sha256(ARCHIVE_MAY2024)
        ok1b = sha1 == SHA256_MAY2024
        zok1, zmsg1 = _zip_ok(ARCHIVE_MAY2024) if ok1b else (False, "skipped (hash mismatch)")
        sat1 = ok1b and zok1
        d1 = (f"Present ({ARCHIVE_MAY2024.stat().st_size:,} bytes), SHA-256 verified, {zmsg1}."
              if sat1 else f"SHA-256 mismatch or ZIP error: {zmsg1}")
    else:
        sat1, d1 = False, f"Not found: {ARCHIVE_MAY2024}"
    conditions.append(_cond("GATE-4V-01",
        "May 2024 SoLEXS archive present and bit-level SHA-256 verified.", sat1, d1))

    # GATE-4V-02: Feb 2026 archive present and hash-verified
    ok2 = ARCHIVE_FEB2026.is_file()
    if ok2:
        sha2 = _sha256(ARCHIVE_FEB2026)
        ok2b = sha2 == SHA256_FEB2026
        zok2, zmsg2 = _zip_ok(ARCHIVE_FEB2026) if ok2b else (False, "skipped")
        sat2 = ok2b and zok2
        d2 = (f"Present ({ARCHIVE_FEB2026.stat().st_size:,} bytes), SHA-256 verified, {zmsg2}."
              if sat2 else f"SHA-256 mismatch or ZIP error: {zmsg2}")
    else:
        sat2, d2 = False, f"Not found: {ARCHIVE_FEB2026}"
    conditions.append(_cond("GATE-4V-02",
        "Jan-Feb 2026 SoLEXS archive present and bit-level SHA-256 verified.", sat2, d2))

    # GATE-4V-03: Phase 4U contract and governance documents present
    missing = [n for n, p in REQUIRED_CONTRACT_FILES if not p.is_file()]
    sat3 = len(missing) == 0
    conditions.append(_cond("GATE-4V-03",
        "All Phase 4U dataset contract and holdout governance documents present on disk.",
        sat3, "All required contract files present." if sat3 else f"Missing: {missing}"))

    # GATE-4V-04: Observation inventory has 14 deduplicated products
    inv_path = REPORTS_DIR / "phase4u_observation_inventory.csv"
    try:
        import pandas as pd
        df = pd.read_csv(inv_path)
        n = len(df)
        u = df["product_key"].nunique() if "product_key" in df.columns else 0
        no_dup = bool((df["is_duplicate_product"] == False).all()) if "is_duplicate_product" in df.columns else False
        may_found = set(df.loc[df["observation_date"].astype(str).str.startswith("2024"),
                                "observation_date"].astype(str).unique())
        feb_found = set(df.loc[df["observation_date"].astype(str).str.startswith("2026"),
                                "observation_date"].astype(str).unique())
        sat4 = (n == EXPECTED_DEV_PRODUCTS and u == EXPECTED_DEV_PRODUCTS and no_dup
                and EXPECTED_MAY_DATES == may_found and EXPECTED_FEB_DATES == feb_found)
        d4 = (f"Verified: {u} unique products, 0 duplicates."
              if sat4 else
              f"Mismatch: n={n}, unique={u}, dups={not no_dup}, "
              f"may_missing={sorted(EXPECTED_MAY_DATES - may_found)}, "
              f"feb_missing={sorted(EXPECTED_FEB_DATES - feb_found)}")
    except Exception as exc:
        sat4, d4 = False, f"Failed to read inventory: {exc}"
    conditions.append(_cond("GATE-4V-04",
        "Phase 4U inventory contains 14 deduplicated SoLEXS SDD2 products across both epochs.",
        sat4, d4))

    # GATE-4V-05: Prior Phase 4U gate confirms development prerequisites satisfied
    try:
        with open(REPORTS_DIR / "phase4u_readiness_gate.json", "r", encoding="utf-8") as fh:
            pg = json.load(fh)
        dev_ids = {"GATE-4U-01", "GATE-4U-02", "GATE-4U-03", "GATE-4U-04"}
        dev_conds = [c for c in pg.get("conditions", []) if c["condition_id"] in dev_ids]
        sat5 = len(dev_conds) == 4 and all(c["satisfied"] for c in dev_conds)
        d5 = ("Phase 4U GATE-4U-01..04 all satisfied in recorded gate report."
              if sat5 else
              f"Unsatisfied: {[c['condition_id'] for c in dev_conds if not c['satisfied']]}")
    except Exception as exc:
        sat5, d5 = False, f"Cannot read phase4u_readiness_gate.json: {exc}"
    conditions.append(_cond("GATE-4V-05",
        "Phase 4U gate report confirms development prerequisites (GATE-4U-01..04) satisfied.",
        sat5, d5))

    # GATE-4V-06: Causal conditioning and no-lookahead rule documented
    try:
        txt = (REPORTS_DIR / "phase4u_dataset_contract.md").read_text(encoding="utf-8")
        sat6 = ("Causal" in txt and "forecast origin" in txt and
                ("look-ahead" in txt.lower() or "lookahead" in txt.lower() or "informational" in txt.lower()))
        d6 = ("Contract contains causal cohort conditioning and look-ahead prohibition."
              if sat6 else "Contract missing causal conditioning clause.")
    except Exception as exc:
        sat6, d6 = False, f"Cannot read contract: {exc}"
    conditions.append(_cond("GATE-4V-06",
        "Dataset contract specifies causal origin conditioning and prohibits look-ahead bias.",
        sat6, d6))

    # GATE-4V-07: October holdout acquired (INFORMATIONAL — non-blocking for dev training)
    conditions.append(_cond("GATE-4V-07",
        f"{HOLDOUT_EPOCH_ID} ({HOLDOUT_DATE_RANGE}) acquired locally [NON-BLOCKING for dev training].",
        False,
        "BLOCKED (non-blocking): October 2024 holdout not acquired; requires authenticated "
        "ISRO PRADAN download. Final holdout evaluation MUST NOT proceed until acquired."))

    # ── Decision ──────────────────────────────────────────────────────────────
    hard_ids = {"GATE-4V-01", "GATE-4V-02", "GATE-4V-03", "GATE-4V-04", "GATE-4V-05", "GATE-4V-06"}
    hard_conds = [c for c in conditions if c["condition_id"] in hard_ids]
    all_hard = all(c["satisfied"] for c in hard_conds)
    failed_hard = [c["condition_id"] for c in hard_conds if not c["satisfied"]]

    overall_status = ("PHASE4V_DEVELOPMENT_TRAINING_AUTHORIZED" if all_hard
                      else "PHASE4V_BLOCKED_READINESS_GATE")
    training_mode = "DEVELOPMENT_ONLY_NOT_FINAL_MODEL" if all_hard else "BLOCKED"

    return {
        "gate_evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 4V",
        "overall_status": overall_status,
        "training_authorized": all_hard,
        "training_mode": training_mode,
        "holdout_locked": True,
        "holdout_epoch": HOLDOUT_EPOCH_ID,
        "holdout_date_range": HOLDOUT_DATE_RANGE,
        "holdout_locally_available": False,
        "summary": (
            "All 6 hard-prerequisite conditions satisfied. Development-only training authorized on "
            "Epoch 1 (May 2024) and Epoch 2 (Jan-Feb 2026) per Phase 4U holdout governance. "
            "October holdout remains LOCKED."
            if all_hard else
            f"Training BLOCKED. Failed hard conditions: {failed_hard}."
        ),
        "conditions": conditions,
        "failed_hard_conditions": failed_hard,
        "governance_reference": "backend/reports/phase4u_holdout_governance.md",
        "dataset_contract_reference": "backend/reports/phase4u_dataset_contract.md",
        "next_action": (
            "Run phase4v_pipeline.py --mode dev-train to execute DEVELOPMENT_ONLY_NOT_FINAL_MODEL training. "
            "Acquire October 2024 epoch before any final evaluation."
            if all_hard else "Resolve failing gate conditions."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4V Readiness Gate — exits nonzero if any hard prerequisite fails"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    report = evaluate_phase4v_gate()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 80)
        print("Phase 4V: Multi-Epoch Readiness Gate")
        print("=" * 80)
        print(f"Status  : {report['overall_status']}")
        print(f"Mode    : {report['training_mode']}")
        print(f"Holdout : {'LOCKED' if report['holdout_locked'] else 'OPEN'}")
        print()
        for c in report["conditions"]:
            mark = "PASS" if c["satisfied"] else "FAIL"
            nb = " [NON-BLOCKING]" if c["condition_id"] == "GATE-4V-07" else ""
            print(f"  [{mark}]{nb} {c['condition_id']}: {c['detail'][:120]}")
        print()
        print(f"Summary: {report['summary']}")
        print("=" * 80)

    if args.save:
        ART_4V_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "phase4v_readiness_gate.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        (ART_4V_DIR / "phase4v_readiness_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved gate report to {out}")

    return 0 if report["training_authorized"] else 1


if __name__ == "__main__":
    sys.exit(main())

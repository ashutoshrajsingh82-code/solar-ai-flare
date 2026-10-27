"""Build a document/evidence register without changing science data or gates."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[1]
DATA = Path(r"E:\SolarFlareData")
DOCS = DATA / "source_docs"
OUT = REPO / "backend" / "reports"
RULES = DATA / "reports" / "phase3c_instrument_semantics_matrix.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    docs = []
    inventory_roots = (DOCS, DATA / "goes_official")
    for path in sorted(p for root in inventory_roots if root.exists() for p in root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".json"}:
            continue
        item = {"filename": path.name, "path": str(path), "size_bytes": path.stat().st_size,
                "sha256": sha256(path), "modified_local": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()}
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path), strict=False)
            meta = reader.metadata or {}
            item.update({"kind": "PDF", "page_count": len(reader.pages),
                         "pdf_metadata": {str(k).lstrip("/"): str(v) for k, v in meta.items() if v is not None}})
        else:
            item.update({"kind": "JSON", "metadata": json.loads(path.read_text(encoding="utf-8"))})
        docs.append(item)

    matrix = list(csv.DictReader(RULES.open(encoding="utf-8-sig", newline="")))
    evidence = []
    for row in matrix:
        row["evidence_review"] = "No instrument semantics established by the available document set. ALPPS guide p.6 (scope) directs detailed instrument configurability to instrument-specific guides; it is a scope limitation, not evidence for the proposed rule."
        row["evidence_status"] = "UNVERIFIED"
        row["confidence"] = "HIGH that the local registered set does not substantiate this rule; rule itself remains scientifically unresolved"
        row["required_authoritative_evidence"] = "Applicable SoLEXS/HEL1OS L1 product manual or SIS, including version and exact field/quality/calibration semantics."
        evidence.append(row)
    matrix_path = OUT / "phase4d_semantics_evidence_matrix.csv"
    with matrix_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence[0].keys()))
        writer.writeheader(); writer.writerows(evidence)

    write_json(OUT / "phase4d_document_inventory.json", {"audit_time_utc": now, "documents": docs,
              "instrument_document_directories": {"solexs": str(DATA / "source_docs/isro/solexs"),
              "hel1os": str(DATA / "source_docs/isro/hel1os"), "general": str(DATA / "source_docs/isro/general")},
              "instrument_specific_document_count": 0})
    register = {"audit_time_utc": now, "documents": docs,
                "instrument_document_status": "MISSING: no SoLEXS or HEL1OS product manual/SIS present in registered instrument directories",
                "instrument_rules_verified": 0, "instrument_rules_total": len(matrix),
                "goes_evidence": {"GOES_Flare_Report_ReadMe.pdf": "Pages 1-4, 6-7 establish report scope, per-event records/fields, internal cross-satellite deduplication caveat and version history. Does not establish catalog absence as a valid negative or direct identity tolerance against local catalog.",
                                  "GOES-R_XRS_L2_Data_ReadMe.pdf": "Pages 2-4 describe product inputs and science/operational quality distinction; no global completeness guarantee.",
                                  "GOES-R_XRS_L2_Data_Users_Guide.pdf": "Section 8, PDF p.12 describes the flare-report composite."},
                "overall_status": "BLOCKED_DOCUMENTATION_UNAVAILABLE"}
    write_json(OUT / "phase4d_document_register.json", register)

    coverage = """# Phase 4D — Evidence-based semantics and label readiness\n\n""" + f"Audit timestamp: {now}\n\n""" + "## Decision\n\n**BLOCKED — 0/14 SoLEXS/HEL1OS semantics verified. No verified real training dataset exists; no model fitting was run.** The authoritative Phase 3C gate remains `BLOCKED_DOCUMENTATION_UNAVAILABLE`. Instrument folders `source_docs/isro/solexs` and `source_docs/isro/hel1os` contain no files.\n\n## Document inventory\n\nSee `phase4d_document_inventory.json` for every PDF/JSON under `E:\\SolarFlareData\\source_docs` and the GOES metadata directory, including byte size, SHA-256, PDF pages, and embedded metadata. The four available PDFs are the general ALPPS guide and three GOES documents; none is an instrument L1 contract. ALPPS User Guide PDF p.6 defines its scope and points instrument-specific details to instrument guides.\n\n## Rule-by-rule evidence\n\n`phase4d_semantics_evidence_matrix.csv` carries forward all 14 Phase 3C rules, their observed product evidence and unresolved questions. Each remains `UNVERIFIED`; the ALPPS scope statement is cited only to establish absence/scope, never to infer semantics. Required evidence includes SoLEXS RATE epoch/time scale, units, GTI endpoints, quality, calibration and version compatibility; HEL1OS MJD/ISOT authoritative time scale, CTR/STAT_ERR semantics, GTI endpoints, quality, calibration and V111/V211 product precedence. No adapter semantics were changed.\n\n## GOES matching and negative labels (separate assessment)\n\nThe local `GOES_Flare_Report_ReadMe.pdf` documents report fields and composition: PDF pp.1-4, including one record per event, start/peak/end fields, and a four-minute plus irradiance-tolerance rule for reconciling observations between GOES satellites. That rule is internal to NOAA report composition; it does not establish direct identity matching against the local catalog. The local Phase 3B start-time candidate tolerance remains project policy, not a NOAA-authorized match rule. NOAA materials describe available-data/report inputs and caveats, but do not state that an absent record proves no flare. Thus catalog-unlisted intervals remain unknown. Existing reported audit counts remain provisional: 123 official events (C=50, M=67, X=6), 107 local exact start/class matches, 8 same-family magnitude discrepancies, 8 official-only events; 21 official rows lack event end. These are audit reconciliation counts, not independently verified instrument event windows.\n\n## Coverage and training eligibility\n\nPrior structural inventory: SoLEXS 518,400 raw rows across six light curves; HEL1OS 1,047,288 rows across 44 products, with V111/V211 overlap unresolved. These are raw parsed counts, not usable observations. Verified usable observations: **unknown**; verified positive/negative windows: **0 established**; negative completeness: **unverified**; independently verified event windows: **0**. No final chronological split feasibility can be established until time semantics, GTI/quality policy, paired-observation rules, event identity and label-completeness policy are resolved. No checkpoint was produced.\n\n## Tests and scope\n\nExisting focused tests cover provisional event matching, preserving class discrepancies, ambiguous candidates, unknown labels for incomplete coverage/catalog completeness, split-horizon purging and fail-closed instrument readers. Phase 4D adds no tests that assert unverified instrument semantics. See `phase4d_test_results.md`.\n\n## Exact next step\n\nManually obtain the registered SoLEXS and HEL1OS Level-1 product manuals/SIS and applicable calibration/quality documents via authorized ISRO/ISSDC access, then place copies in the prepared instrument folders. After the documents are available, rerun:\n\n```powershell\npython scripts/phase3c_semantics_audit.py --data-root E:\\SolarFlareData --repo-root C:\\Users\\ASHUTOSH\\Downloads\\solar-ai-flare-main\\solar-ai-flare-main --report-root E:\\SolarFlareData\\reports\n```\n"""
    (OUT / "phase4d_coverage_report.md").write_text(coverage, encoding="utf-8")
    (OUT / "phase4d_readiness.md").write_text("# Phase 4D readiness\n\nBLOCKED_DOCUMENTATION_UNAVAILABLE — 0/14 instrument semantics verified. Real-data training remains disabled; no verified training dataset or chronological partitions exist, and model fitting was not executed. See `phase4d_coverage_report.md`, `phase4d_document_register.json`, and `phase4d_semantics_evidence_matrix.csv`.\n", encoding="utf-8")
    (OUT / "phase4d_test_results.md").write_text("# Phase 4D focused regression results\n\nCommand (from `backend`):\n\n```powershell\npython -m pytest tests\\test_phase3a_data_audit.py tests\\test_phase3b_authoritative_audit.py tests\\test_phase3c_semantics_audit.py tests\\test_real_data_sources.py tests\\test_chronological_audit.py tests\\test_feature_engineering_step9.py tests\\test_scientific_readiness.py tests\\test_phase2c_eligibility.py -q --basetemp=.pytest-phase4d-readiness-20261024\n```\n\nResult: **63 passed, 25 warnings, 0 failed** (13.63 s). Warnings: upstream Pydantic/SQLAlchemy deprecations and pandas DataFrame fragmentation performance warnings. Existing tests cover label unknowns, GOES candidate matching and ambiguity, cross-catalog class discrepancies, horizon-tail purging, and fail-closed product readers. No new tests were added for instrument rules because no such rule was verified by authoritative documentation; asserting them would encode unsupported semantics.\n", encoding="utf-8")


if __name__ == "__main__":
    main()

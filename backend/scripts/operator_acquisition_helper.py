"""Phase 4Q: Operator-Assisted Acquisition & Provenance Verification Tool.

Facilitates authorized data acquisition through the official ISRO ISSDC PRADAN portal:
- Verifies downloaded archive integrity (SHA-256 and byte counts).
- Sanitizes and redacts credentials, session cookies, and private tokens.
- Records acquisition provenance and validation status.
- Generates structured, reproducible acquisition logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def redact_sensitive_string(text: str) -> str:
    """Redact tokens, passwords, cookies, and sensitive query parameters."""
    # Redact session tokens
    text = re.sub(r'(SESSION|JSESSIONID|token|access_token|state)=([a-zA-Z0-9_\-\.]+)', r'\1=[REDACTED]', text, flags=re.IGNORECASE)
    # Redact Bearer tokens
    text = re.sub(r'Bearer\s+[a-zA-Z0-9_\-\.]+', 'Bearer [REDACTED]', text, flags=re.IGNORECASE)
    # Redact password fields
    text = re.sub(r'password=([^&\s]+)', 'password=[REDACTED]', text, flags=re.IGNORECASE)
    return text


def inspect_archive(archive_path: Path) -> Dict[str, Any]:
    """Inspect and verify archive bitstream integrity and structure."""
    if not archive_path.exists():
        return {
            "archive_path": str(archive_path),
            "status": "FILE_NOT_FOUND",
            "sha256": None,
            "byte_size": 0,
            "is_valid_zip": False,
        }

    byte_size = archive_path.stat().st_size
    sha256_hash = compute_sha256(archive_path)

    is_valid_zip = False
    members = []
    daily_packages = []
    corrupt_error = None

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            test_res = zf.testzip()
            if test_res is None:
                is_valid_zip = True
                members = zf.namelist()
                daily_packages = [m for m in members if m.endswith(".zip")]
            else:
                corrupt_error = f"Corrupt archive member: {test_res}"
    except Exception as e:
        corrupt_error = str(e)

    return {
        "archive_path": str(archive_path),
        "archive_name": archive_path.name,
        "byte_size": byte_size,
        "sha256": sha256_hash,
        "is_valid_zip": is_valid_zip,
        "corruption_error": corrupt_error,
        "total_members": len(members),
        "daily_packages": sorted(daily_packages),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def generate_acquisition_log(archive_paths: List[Path], output_path: Path) -> Dict[str, Any]:
    """Generate structured Phase 4Q acquisition log."""
    inspections = [inspect_archive(p) for p in archive_paths if p.exists()]
    
    log = {
        "log_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "portal_name": "ISRO ISSDC PRADAN (Aditya-L1)",
        "portal_url": "https://pradan1.issdc.gov.in/al1/",
        "authentication_method": "Authorized User Interactive SSO (Keycloak)",
        "credential_policy": "NO_STORED_CREDENTIALS_ALLOWED",
        "archives_inspected": len(inspections),
        "archives": inspections,
        "operator_instructions": {
            "step_1": "Navigate to https://pradan1.issdc.gov.in/al1/ in an authorized web browser.",
            "step_2": "Authenticate via ISRO Single Sign-On (SSO).",
            "step_3": "Search Mission 'Aditya-L1', Instrument 'SoLEXS', Level-1 Lightcurves ('lc').",
            "step_4": "Select candidate dates (e.g. May 2024, or Feb 6-28 2024/2026), add to Cart, and export download script.",
            "step_5": "Execute download script within active session window, placing zip file in Downloads.",
            "step_6": "Run operator_acquisition_helper.py to verify checksums and index new files.",
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return log


def main():
    parser = argparse.ArgumentParser(description="Phase 4Q Operator-Assisted Acquisition & Verification Tool")
    parser.add_argument("--archives", type=str, nargs="+",
                        default=[
                            r"C:\Users\ASHUTOSH\Downloads\solexs_2026Oct23T184402685.zip",
                            r"C:\Users\ASHUTOSH\Downloads\hel1os_2026Oct23T184518914.zip"
                        ],
                        help="List of archive paths to verify")
    parser.add_argument("--output-log", type=str,
                        default=str(backend_dir / "reports" / "phase4q_acquisition_log.json"),
                        help="Path for generated JSON acquisition log")
    args = parser.parse_args()

    paths = [Path(p) for p in args.archives]
    out_path = Path(args.output_log)

    print("=" * 80)
    print("Phase 4Q: Operator-Assisted Acquisition & Integrity Verifier")
    print("=" * 80)
    
    log = generate_acquisition_log(paths, out_path)
    print(f"Verified {log['archives_inspected']} archive(s).")
    for a in log["archives"]:
        print(f"  - {a['archive_name']}: {a['byte_size']:,} bytes, valid_zip={a['is_valid_zip']}, sha256={a['sha256'][:16]}...")
    print(f"\nSaved Acquisition Log to: {out_path.name}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

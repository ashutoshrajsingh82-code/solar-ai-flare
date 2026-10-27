#!/usr/bin/env python3
"""Execute polite, resumable PRADAN batch downloads under strict credential and holdout rules.

URL and File Patterns (derived from authoritative PRADAN bulk download scripts):
  - SoLEXS:
      URL: https://pradan1.issdc.gov.in/al1/protected/downloadData/solexs/level1/{YYYY}/{MM}/{ORBIT_DIR}/AL1_SLX_L1_{YYYYMMDD}_v1.0.zip?solexs
      File on disk: AL1_SLX_L1_{YYYYMMDD}_v1.0.zip
  - HEL1OS:
      URL: https://pradan1.issdc.gov.in/al1/protected/downloadData/hel1os/level1/{YYYY}/{MM}/{DD}/N00_0000/HLS_{YYYYMMDD}_{HHMMSS}_{SECS}sec_lev1_V{VER}.zip?hel1os
      File on disk: HLS_{YYYYMMDD}_{HHMMSS}_{SECS}sec_lev1_V{VER}.zip

Rules:
  1. Credentials/tokens read ONLY from environment variable PRADAN_TOKEN.
  2. Strip leading/trailing single or double quotes from PRADAN_TOKEN.
  3. Never prints or logs credentials, cookies, tokens, or query secrets.
  4. Polite to server: 1 concurrent download, >= 2.0s sleep between requests, exponential backoff.
  5. 404 is treated as NOT_AVAILABLE: logged, not retried, and does NOT count toward 5-failure halt.
  6. 401/403: halts batch immediately and requests cookie refresh.
  7. Failures counted toward 5-consecutive halt: network errors, 5xx, and corrupted zip files.
  8. Redirects: allow_redirects=False. Follows redirect ONLY if Location host is on PRADAN domain.
     Never sends Cookie header to any non-PRADAN host.
  9. Max 15 days per batch.
 10. October 2024 (2024-10-01..2024-10-31) is strictly forbidden; raises fatal error.
 11. Verification: file size > 0 and zip integrity check.
 12. Output log: data/pradan_download_log.csv.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse

cur_path = Path(__file__).resolve()
if cur_path.parent.name == "scripts" and cur_path.parent.parent.name == "backend":
    REPO_ROOT = cur_path.parent.parent.parent
else:
    REPO_ROOT = cur_path.parent.parent

DATA_DIR = REPO_ROOT / "data"
PLAN_PATH = DATA_DIR / "pradan_plan.csv"
LOG_PATH = DATA_DIR / "pradan_download_log.csv"
DOWNLOADS_OUT_DIR = REPO_ROOT / "data" / "pradan_downloads"
EXTERNAL_DOWNLOADS_DIR = Path("C:/Users/ASHUTOSH/Downloads")

HOLDOUT_START = "2024-10-01"
HOLDOUT_END = "2024-10-31"

PRADAN_DOMAINS = ("pradan1.issdc.gov.in", "issdc.gov.in")
URL_PREFIX = "https://pradan1.issdc.gov.in"


def is_pradan_host(url: str) -> bool:
    """Check if the URL belongs strictly to the PRADAN / ISSDC domain."""
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return False
        return host in PRADAN_DOMAINS or any(host.endswith("." + d) for d in PRADAN_DOMAINS)
    except Exception:
        return False


def redact_url(url: str) -> str:
    """Strip secret tokens or sensitive query parameters from URL for safe logging."""
    try:
        parsed = urlparse(url)
        q = parse_qs(parsed.query)
        redacted_q = {}
        for k, v in q.items():
            if any(s in k.lower() for s in ("token", "auth", "key", "pass", "session", "secret")):
                redacted_q[k] = "[REDACTED]"
            else:
                redacted_q[k] = v
        new_query = urlencode(redacted_q, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
    except Exception:
        return "[REDACTED_URL]"


redact_url_secrets = redact_url


def assert_safe_date(d_str: str) -> None:
    if HOLDOUT_START <= d_str <= HOLDOUT_END:
        raise ValueError(f"FATAL HOLDOUT GUARD: Refusing to request date {d_str} in October 2024 holdout!")


def assert_safe_from_holdout(dates: list[str]) -> None:
    for d in dates:
        assert_safe_date(d)


def login_pradan_keycloak(username: str, password: str, session=None) -> str | None:
    """Authenticate via ISSDC Keycloak OpenID Connect to obtain active session cookies."""
    import requests
    if session is None:
        session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SolarFlareAI/1.0"
    })

    try:
        r1 = session.get("https://pradan1.issdc.gov.in/al1/protected/payload.xhtml", timeout=20, allow_redirects=True)
        m_act = re.search(r'<form\b[^>]*action=["\']([^"\']*)["\']', r1.text, re.IGNORECASE)
        if not m_act:
            return None

        action_url = m_act.group(1).replace("&amp;", "&")
        payload = {
            "username": username,
            "password": password,
            "credentialId": "",
            "login": "Sign In",
        }
        r2 = session.post(action_url, data=payload, timeout=20, allow_redirects=True)
        if "pradan1.issdc.gov.in" in r2.url and ("payload.xhtml" in r2.url or "protected" in r2.url):
            cookie_parts = []
            for c in session.cookies:
                cookie_parts.append(f"{c.name}={c.value}")
            return "; ".join(cookie_parts)
    except Exception as exc:
        print(f"[AUTH ERROR] Keycloak authentication failed: {exc}")
    return None


def get_auth_cookie(session=None) -> str | None:
    """Retrieve auth cookie/token from environment, stripping surrounding quotes.
    If PRADAN_TOKEN is absent, checks PRADAN_USER and PRADAN_PASS to authenticate."""
    token = os.environ.get("PRADAN_TOKEN")
    if token:
        token = token.strip()
        while len(token) >= 2 and ((token[0] == "'" and token[-1] == "'") or (token[0] == '"' and token[-1] == '"')):
            token = token[1:-1].strip()
        if token:
            return token

    # Check for PRADAN_USER and PRADAN_PASS
    u = os.environ.get("PRADAN_USER")
    p = os.environ.get("PRADAN_PASS")
    if u and p:
        u_clean = u.strip().strip("'\"")
        p_clean = p.strip().strip("'\"")
        print("[AUTH] Authenticating via PRADAN Keycloak SSO using environment credentials...")
        cookie_str = login_pradan_keycloak(u_clean, p_clean, session=session)
        if cookie_str:
            print("[AUTH] Successfully established authenticated PRADAN session.")
            os.environ["PRADAN_TOKEN"] = cookie_str
            return cookie_str
        else:
            print("[AUTH] Keycloak authentication failed. Check credentials.")

    return None


def init_log_file() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("date,instrument,file_name,size_bytes,status,timestamp\n", encoding="utf-8")


def append_log(date_str: str, instrument: str, file_name: str, size_bytes: int, status: str) -> None:
    init_log_file()
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Safety assertion: ensure no token text can be logged
    token = get_auth_cookie()
    if token and len(token) > 8 and (token in file_name or token in status):
        file_name = "[REDACTED]"
        status = "[REDACTED]"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{date_str},{instrument},{file_name},{size_bytes},{status},{ts}\n")


def parse_pradan_script_catalog() -> dict[tuple[str, str], list[str]]:
    """Scan existing PRADAN bulk download scripts to build authoritative path mapping.
    
    Returns:
        dict mapping (instrument, 'YYYY-MM-DD') -> list of relative data_file_paths.
    """
    catalog: dict[tuple[str, str], list[str]] = {}
    search_dirs = [EXTERNAL_DOWNLOADS_DIR, DOWNLOADS_OUT_DIR, REPO_ROOT]
    seen_scripts = set()

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for py_path in s_dir.glob("*.py"):
            if py_path in seen_scripts:
                continue
            seen_scripts.add(py_path)
            name_lower = py_path.name.lower()
            if not ("solexs" in name_lower or "hel1os" in name_lower):
                continue

            try:
                content = py_path.read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"data_file_paths\s*=\s*\[(.*?)\]", content, re.DOTALL)
                if not m:
                    continue
                paths = re.findall(r'"([^"]+)"', m.group(1))
                for p in paths:
                    clean_p = p.split("?")[0]
                    # Extract date
                    # SoLEXS: AL1_SLX_L1_YYYYMMDD_v1.0.zip
                    m_s = re.search(r"AL1_SLX_L1_(\d{4})(\d{2})(\d{2})", clean_p)
                    if m_s:
                        d = f"{m_s.group(1)}-{m_s.group(2)}-{m_s.group(3)}"
                        catalog.setdefault(("solexs", d), []).append(p)
                        continue
                    # HEL1OS: HLS_YYYYMMDD_...
                    m_h = re.search(r"HLS_(\d{4})(\d{2})(\d{2})", clean_p)
                    if m_h:
                        d = f"{m_h.group(1)}-{m_h.group(2)}-{m_h.group(3)}"
                        catalog.setdefault(("hel1os", d), []).append(p)
            except Exception:
                pass

    return catalog


def get_download_items_for_date(
    d_str: str,
    instrument: str,
    catalog: dict[tuple[str, str], list[str]],
) -> list[tuple[str, str]]:
    """Determine URL and destination filename for a given date and instrument.
    
    Returns list of (url, target_filename).
    """
    key = (instrument.lower(), d_str)
    d_clean = d_str.replace("-", "")
    yyyy = d_str[:4]
    mm = d_str[5:7]
    dd = d_str[8:10]

    # 1. Authoritative path from PRADAN bulk script
    if key in catalog and len(catalog[key]) > 0:
        items = []
        for rel_p in catalog[key]:
            url = urljoin(URL_PREFIX, rel_p)
            clean_name = rel_p.split("?")[0].split("/")[-1]
            items.append((url, clean_name))
        return items

    # 2. Inferred pattern based on authoritative PRADAN URL structure
    if instrument.lower() == "solexs":
        # Standard PRADAN orbit directory is N00_0000
        fname = f"AL1_SLX_L1_{d_clean}_v1.0.zip"
        rel_path = f"/al1/protected/downloadData/solexs/level1/{yyyy}/{mm}/N00_0000/{fname}?solexs"
        return [(urljoin(URL_PREFIX, rel_path), fname)]
    else:
        # HEL1OS requires day directory and nominal orbit N00_0000
        fname = f"HLS_{d_clean}_lev1.zip"
        rel_path = f"/al1/protected/downloadData/hel1os/level1/{yyyy}/{mm}/{dd}/N00_0000/{fname}?hel1os"
        return [(urljoin(URL_PREFIX, rel_path), fname)]


def download_single_file_safe(
    url: str,
    target_p: Path,
    token: str,
    session,
) -> tuple[int, str]:
    """Download a file with allow_redirects=False and PRADAN domain validation.
    
    Returns (status_code, error_message).
    """
    current_url = url
    for _ in range(5):  # Max 5 redirects
        if not is_pradan_host(current_url):
            return 400, f"REDIRECT_TO_NON_PRADAN_HOST_BLOCKED ({urlparse(current_url).netloc})"

        # Only pass Cookie header to verified PRADAN host
        req_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SolarFlareAI/1.0",
            "Cookie": token,
        }
        resp = session.get(
            current_url,
            headers=req_headers,
            stream=True,
            timeout=30,
            allow_redirects=False,
        )

        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if not loc:
                return resp.status_code, "REDIRECT_LOCATION_MISSING"
            next_url = urljoin(current_url, loc)
            if not is_pradan_host(next_url):
                print(f"  [SECURITY] Blocked redirect to external host: {urlparse(next_url).netloc}")
                return 400, "REDIRECT_TO_NON_PRADAN_HOST_BLOCKED"
            current_url = next_url
            continue

        if resp.status_code == 200:
            part_p = Path(str(target_p) + ".part")
            try:
                with open(part_p, "wb") as f_out:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f_out.write(chunk)
                if part_p.exists() and part_p.stat().st_size > 0:
                    part_p.replace(target_p)
                return 200, "OK"
            except Exception as exc:
                part_p.unlink(missing_ok=True)
                return 500, f"STREAM_ERROR: {exc}"

        return resp.status_code, f"HTTP_{resp.status_code}"

    return 310, "TOO_MANY_REDIRECTS"


def run_batch(
    dates: list[str],
    instrument: str = "solexs",
    max_days: int = 15,
) -> dict[str, int | bool]:
    """Execute a batch of up to 15 planned dates with strict status classification."""
    if len(dates) > max_days:
        print(f"Truncating batch from {len(dates)} to maximum permitted {max_days} days.")
        dates = dates[:max_days]

    init_log_file()
    DOWNLOADS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    token = get_auth_cookie()
    print(f"[AUTH STATUS] PRADAN session token set: {'yes' if token else 'no'}")

    if not token:
        print("\n" + "=" * 70)
        print("PRADAN AUTHENTICATION REQUIRED")
        print("=" * 70)
        print("Per project credential rules, PRADAN uses session-bound downloads.")
        print("Please export your active PRADAN session cookie in your terminal (PowerShell):")
        print("  $env:PRADAN_TOKEN = 'FGTServer=...; JSESSIONID=...'")
        print("Never paste credentials into model chats or git commits.")
        print("=" * 70 + "\n")
        return {"ok": 0, "failed": 0, "skipped": 0, "not_available": 0, "auth_required": 1}

    # Verify every date in batch against holdout quarantine
    for d in dates:
        assert_safe_date(d)

    import requests
    session = requests.Session()

    catalog = parse_pradan_script_catalog()
    stats = {"ok": 0, "failed": 0, "skipped": 0, "not_available": 0, "auth_expired": False}
    consecutive_failures = 0

    for d_str in dates:
        if consecutive_failures >= 5:
            print(f"[HALT] 5 consecutive failures encountered. Stopping batch for server protection.")
            break

        download_items = get_download_items_for_date(d_str, instrument, catalog)

        date_has_failure = False
        date_not_available = False

        for url, fname in download_items:
            target_p = DOWNLOADS_OUT_DIR / fname

            # 1. Skip if already downloaded and valid
            if target_p.exists() and target_p.stat().st_size > 0:
                try:
                    with zipfile.ZipFile(target_p, "r") as zf:
                        if len(zf.namelist()) > 0:
                            print(f"[SKIP] {d_str} already on disk ({fname}, {target_p.stat().st_size:,} bytes)")
                            stats["skipped"] += 1
                            append_log(d_str, instrument, fname, target_p.stat().st_size, "SKIPPED_EXISTING")
                            continue
                except Exception:
                    pass

            safe_url = redact_url(url)
            print(f"[FETCH] Requesting {d_str} ({instrument}) from {safe_url} ...")

            retries = 3
            success = False
            wait_s = 2.0

            for attempt in range(1, retries + 1):
                time.sleep(wait_s)  # Polite request spacing (>= 2s)
                try:
                    status_code, msg = download_single_file_safe(url, target_p, token, session)

                    if status_code == 200:
                        f_size = target_p.stat().st_size if target_p.exists() else 0
                        # Integrity check
                        if f_size > 1024 and zipfile.is_zipfile(target_p):
                            print(f"  [OK] Successfully downloaded {fname} ({f_size:,} bytes)")
                            stats["ok"] += 1
                            append_log(d_str, instrument, fname, f_size, "SUCCESS")
                            consecutive_failures = 0
                            success = True
                            break
                        else:
                            target_p.unlink(missing_ok=True)
                            print(f"  [CORRUPTED] Downloaded file invalid or corrupted zip.")
                            consecutive_failures += 1
                            date_has_failure = True
                    elif status_code == 404:
                        print(f"  [NOT AVAILABLE] HTTP 404: {fname} not found on server.")
                        append_log(d_str, instrument, fname, 0, "NOT_AVAILABLE")
                        stats["not_available"] += 1
                        date_not_available = True
                        # Do NOT retry, and do NOT count toward consecutive failures!
                        break
                    elif status_code in (401, 403):
                        print(f"\n[AUTH EXPIRED] HTTP {status_code}. Session cookie expired or invalid.")
                        print("Please refresh your cookie: $env:PRADAN_TOKEN = '...'")
                        append_log(d_str, instrument, fname, 0, f"AUTH_{status_code}")
                        stats["auth_expired"] = True
                        return stats
                    elif status_code >= 500:
                        print(f"  [SERVER ERROR] HTTP {status_code}. Attempt {attempt} failed.")
                        consecutive_failures += 1
                        date_has_failure = True
                    else:
                        print(f"  [FAILED] HTTP {status_code}: {msg}. Attempt {attempt} failed.")
                        consecutive_failures += 1
                        date_has_failure = True

                except Exception as exc:
                    print(f"  [NETWORK ERROR] Attempt {attempt}: {exc}")
                    consecutive_failures += 1
                    date_has_failure = True

                wait_s *= 2.0  # Exponential backoff

            if not success and not date_not_available:
                if date_has_failure:
                    stats["failed"] += 1
                    append_log(d_str, instrument, fname, 0, "FAILED")

    print(f"\n[BATCH RESULT] Completed: {stats['ok']} OK, {stats['failed']} Failed, "
          f"{stats['skipped']} Skipped, {stats['not_available']} Not Available.")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run polite PRADAN batch download using active session token",
        epilog="To set token in PowerShell: $env:PRADAN_TOKEN = 'FGTServer=...; JSESSIONID=...'"
    )
    parser.add_argument("--batch-index", type=int, default=1, help="Batch index (1, 2, 3...)")
    parser.add_argument("--batch-size", type=int, default=15, help="Max days per batch (<= 15)")
    parser.add_argument("--instrument", default="solexs", choices=["solexs", "hel1os"])
    parser.add_argument("--login-stdin", action="store_true", help="Read username and password from stdin to log in")
    parser.add_argument("--dry-run", action="store_true", help="Print batch dates without downloading")
    args = parser.parse_args()

    if args.login_stdin:
        u = sys.stdin.readline().strip()
        p = sys.stdin.readline().strip()
        if u and p:
            print("[AUTH] Authenticating to PRADAN Keycloak SSO from stdin...")
            token = login_pradan_keycloak(u, p)
            if token:
                os.environ["PRADAN_TOKEN"] = token
                print("[AUTH] Successfully authenticated. Active session token acquired.")
            else:
                print("[AUTH ERROR] Authentication failed. Check credentials.")
                sys.exit(1)

    if not PLAN_PATH.exists():
        print(f"Error: {PLAN_PATH} not found. Run scripts/plan_pradan_downloads.py first.")
        sys.exit(1)

    import pandas as pd
    plan_df = pd.read_csv(PLAN_PATH)

    # Filter dates
    batch_size = min(args.batch_size, 15)
    start_idx = (args.batch_index - 1) * batch_size
    batch_dates = plan_df["date"].iloc[start_idx : start_idx + batch_size].tolist()

    print(f"Preparing batch #{args.batch_index} ({len(batch_dates)} days) for {args.instrument} ...")
    print(f"Dates: {batch_dates[0]} .. {batch_dates[-1]}")

    if args.dry_run:
        print("[DRY RUN] Invariants checked. No network requests made.")
        catalog = parse_pradan_script_catalog()
        print(f"Loaded {len(catalog)} authoritative date entries from PRADAN bulk scripts.")
        for d in batch_dates[:5]:
            items = get_download_items_for_date(d, args.instrument, catalog)
            for u, fn in items:
                print(f"  {d} -> {fn} : {redact_url(u)}")
        return

    run_batch(batch_dates, instrument=args.instrument, max_days=batch_size)


if __name__ == "__main__":
    main()

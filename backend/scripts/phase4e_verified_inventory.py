"""Verify Phase 3A/4B historical paths and inventory current local solar inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[2]
DOWNLOADS = Path(r"C:\Users\ASHUTOSH\Downloads")
DATA = Path(r"E:\SolarFlareData")
OLD = REPO / "backend" / "reports" / "phase4b_real_data_inventory.json"


def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def classify(path: Path) -> tuple[str,str]:
    name=path.name.lower()
    if name.startswith("solexs_"): return "SoLEXS","original_nested_zip_archive"
    if name.startswith("hel1os_"): return "HEL1OS","original_nested_zip_archive"
    if "goes" in name and path.suffix.lower() in {".csv",".nc",".json"}: return "GOES","observational_catalog_or_measurements"
    if path.suffix.lower()==".pdf": return "documentation","source_document"
    return "unresolved","historical_report_or_other_input"


def inspect(path: Path, expected_hash: str | None = None) -> dict:
    instrument,kind=classify(path)
    row={"path":str(path),"filename":path.name,"exists_now":path.is_file(),"instrument_or_source":instrument,
         "file_class":kind,"expected_historical_sha256":expected_hash,"readability":"NOT_TESTED","error":None,
         "size_bytes":None,"sha256":None,"historical_hash_matches":None,"archive_status":"not_archive",
         "extraction_status":"not_applicable","extracted_destinations":[],"metadata":{}}
    if not path.is_file():
        row["readability"]="MISSING"
        return row
    try:
        row["size_bytes"]=path.stat().st_size; row["sha256"]=file_hash(path)
        row["historical_hash_matches"]=None if expected_hash is None else row["sha256"]==expected_hash
        row["readability"]="READABLE"
        if path.suffix.lower()==".zip":
            row["archive_status"]="original_archive_not_extracted"
            nested=[]
            with zipfile.ZipFile(path) as outer:
                row["metadata"].update({"outer_member_count":len(outer.infolist()),"outer_uncompressed_bytes":sum(i.file_size for i in outer.infolist()),
                                        "outer_member_names":[i.filename for i in outer.infolist()]})
                for info in outer.infolist():
                    if info.filename.lower().endswith(".zip"):
                        raw=outer.read(info)
                        inner_record={"filename":info.filename,"size_bytes":info.file_size,"compressed_size_bytes":info.compress_size,
                                      "sha256":hashlib.sha256(raw).hexdigest()}
                        with zipfile.ZipFile(__import__("io").BytesIO(raw)) as inner:
                            files=[i for i in inner.infolist() if not i.is_dir()]
                            inner_record["member_count"]=len(files)
                            inner_record["fits_members"]=[i.filename for i in files if i.filename.lower().endswith((".fits",".fit",".fits.gz",".fit.gz",".lc.gz",".gti.gz"))]
                            inner_record["fits_like_count"]=len(inner_record["fits_members"])
                        nested.append(inner_record)
            row["metadata"]["nested_archives"]=nested
            row["metadata"]["nested_archive_count"]=len(nested)
            row["extraction_status"]="NOT_EXTRACTED_NESTED_CONTENTS_READ_IN_MEMORY_ONLY"
        elif path.suffix.lower()==".pdf":
            pdf=PdfReader(str(path),strict=False); meta=pdf.metadata or {}; first=(pdf.pages[0].extract_text() or "") if pdf.pages else ""
            version=re.search(r"\bVersion\s+[\w.]+",first,re.I)
            row["metadata"]={"page_count":len(pdf.pages),"pdf_metadata":{str(k).lstrip("/"):str(v) for k,v in meta.items() if v is not None},
                              "first_page_text":re.sub(r"\s+"," ",first)[:1000],"version_text":version.group(0) if version else None}
        elif path.suffix.lower()==".csv":
            with path.open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.reader(stream); header=next(reader,None); rows=sum(1 for _ in reader)
            row["metadata"]={"data_rows_excluding_header":rows,"column_names":header}
    except Exception as exc:
        row["readability"]="READ_ERROR"; row["error"]=f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    manifest=json.loads(OLD.read_text(encoding="utf-8"))
    records=[]
    for item in manifest.get("provenance_files",[]):
        records.append(inspect(Path(item["path"]),item.get("sha256")))
    # Include the historical GOES NetCDF files which Phase 3A read from Downloads.
    records.extend(inspect(p) for p in sorted((DOWNLOADS/"data"/"goes"/"raw").glob("*.nc")))
    # Add the two previously unregistered user manuals plus currently registered PDFs.
    doc_paths=[Path(r"C:\Users\ASHUTOSH\Downloads\SoLEXS-UserManual.pdf"),
               Path(r"C:\Users\ASHUTOSH\Downloads\HEL1OS_UserManual.pdf")]
    doc_paths.extend(sorted((DATA/"source_docs").rglob("*.pdf")))
    records.extend(inspect(p) for p in doc_paths)
    # Record acquisition/report provenance entries as current path checks too.
    report_paths=[REPO/"backend"/"reports"/"phase3a_data_audit.json",
                  DATA/"reports"/"phase3a_data_audit.json",OLD,
                  DATA/"reports"/"phase3c_instrument_semantics_matrix.csv"]
    records.extend(inspect(p) for p in report_paths)
    out=REPO/"backend"/"reports"/"phase4e_verified_data_inventory.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"audit_time_utc":datetime.now(timezone.utc).isoformat(),"configured_data_root":str(DATA),
       "verified_paths_scope":[str(DOWNLOADS),str(DOWNLOADS/"data"),str(DATA),str(REPO)],
       "extracted_destination_checks":{"E_raw_exists":(DATA/"raw").exists(),"E_extracted_exists":(DATA/"extracted").exists(),
          "configured_root_direct_fits_count":sum(1 for p in DATA.rglob("*") if p.is_file() and p.name.lower().endswith((".fits",".fit",".fits.gz",".fit.gz")))},
       "inventory":records},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"verified_readable":sum(x["readability"]=="READABLE" for x in records),
       "missing":sum(x["readability"]=="MISSING" for x in records),"read_errors":sum(x["readability"]=="READ_ERROR" for x in records)}))


if __name__=="__main__":main()

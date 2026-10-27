"""Create additive Phase 4E document/evidence records from verified local manuals."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

REPO=Path(__file__).resolve().parents[2]
DATA=Path(r"E:\SolarFlareData")
DOWNLOADS=Path(r"C:\Users\ASHUTOSH\Downloads")
SOURCE_MATRIX=DATA/"reports"/"phase3c_instrument_semantics_matrix.csv"
OLD_REGISTER=DATA/"reports"/"phase3c_document_register.json"
OUT=REPO/"backend"/"reports"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def doc(path, document, version, citeable_sections, limits):
    reader=PdfReader(str(path),strict=False);metadata=reader.metadata or {}
    first=(reader.pages[0].extract_text() or "") if reader.pages else ""
    return {"document":document,"local_path":str(path),"exists":path.is_file(),"sha256":digest(path),
            "size_bytes":path.stat().st_size,"page_count":len(reader.pages),"version_or_date":version,
            "pdf_metadata":{str(k).lstrip("/"):str(v) for k,v in metadata.items() if v is not None},
            "first_page_text": " ".join(first.split())[:1000],"source_provenance":"local downloaded copy; file hash verified; retrieval origin not independently authenticated",
            "relevant_sections":citeable_sections,"scope_limits":limits}


def main():
    manuals=[doc(DOWNLOADS/"SoLEXS-UserManual.pdf","SoLEXS Data Analysis Guide","Version 1.0",
                 {"PDF p.5; §§2.1.1-2.1.2":"GTI purpose/criteria; Level-1 RATE.TIME Unix Time and COUNTS counts/s over 2-22 keV",
                  "PDF p.6; §2.2 and §2.3":"Unix seconds since 1970-01-01 00:00:00 UTC; daywise product and detector paths/files",
                  "PDF p.7; §2.4":"SoLEXS CALDB provides ARF and RMF response files for spectral analysis",
                  "PDF p.10; §4.1":"Quality flag 1/0 and detector recommendation values apply to Type-I PI spectrum and download filter context"},
                 ["No GTI START/STOP endpoint closure rule stated in these sections.","No rule converts RATE counts/s to physical irradiance.","The Type-I PI QUALITY flag must not be transferred to the RATE light curve.","No creator/OBS_ID/version-precedence semantics established."]),
            doc(DOWNLOADS/"HEL1OS_UserManual.pdf","HEL1OS Data Analysis User Manual","Version 1.2; Jan 2025 (change record through Jan 31, 2025)",
                 {"PDF p.4; §§2-2.1":"Level-1 and Level-2 distinction; Level-2 CdTe pile-up correction/CZT saturation filtering; daily dump naming/version field",
                  "PDF p.5; §2.1":"Archive layout, four detector identities, GTI/HK/event products and product roles",
                  "PDF p.6; §§2.2-2.3":"One-second light curves; CALDB response epoch is selected by observation date and detector family",
                  "PDF p.13; Level-1 product inventory (printed p.13)":"Level-1 GTIs span the full data dump and are relevant only to Level-2; may be ignored for Level-1 analysis",
                  "PDF pp.51-52; Appendix B.3":"MJD, ISOT labelled UTC ISO-T corresponding to MJD; CTR and STAT_ERR in cts/sec; detector/band examples",
                  "PDF pp.53-55; Appendix B.4":"Housekeeping field names/types/descriptions including status and pile-up/saturation counters"},
                 ["MJD's independent time-scale convention/authority is not separately named beyond ISOT being UTC.","GTI endpoint semantics are not needed for the documented L1 ignore rule but remain unspecified for other uses.","No V111/V211 precedence rule found.","HK field names do not define model-quality thresholds or row-rejection logic."])]
    original=list(csv.DictReader(SOURCE_MATRIX.open(encoding="utf-8-sig",newline="")))
    mappings=[
      ("VERIFIED_FROM_AUTHORITATIVE_DOCUMENT","SoLEXS manual PDF p.6 §2.2","RATE.TIME is Unix seconds from 1970-01-01 00:00:00 UTC.","No ambiguity for described SoLEXS Level-1 time coordinate; FITS values are not converted in this audit."),
      ("VERIFIED_FROM_AUTHORITATIVE_DOCUMENT","SoLEXS manual PDF p.5 §2.1.2","RATE.COUNTS is counts per second in 2-22 keV.","FITS column unit is absent in this sample; manual supplies interpretation, not a physical flux conversion."),
      ("PARTIALLY_RESOLVED","SoLEXS manual PDF p.5 §2.1.1","GTI identifies valid collection intervals based on Sun FOV, detector operation, and health criteria.","START/STOP units, numeric column contract, and interval endpoint closure remain unspecified."),
      ("VERIFIED_FROM_AUTHORITATIVE_DOCUMENT","SoLEXS manual PDF pp.5-6 §§2.1.2-2.3","L1 light curve RATE is associated with day-wise SDD1/SDD2 product directories and named detector products.","Does not establish processing-version precedence beyond naming a version token."),
      ("PARTIALLY_RESOLVED","SoLEXS manual PDF pp.5-6 §§2.1.2-2.3","Light-curve cadence is one second; products are day-wise.","CREATOR, DATE, OBS_ID field semantics and version compatibility are not fully documented."),
      ("PARTIALLY_RESOLVED","SoLEXS manual PDF p.7 §2.4","ARF/RMF response files are required for spectral analysis and physical interpretation.","No direct conversion from the audited light-curve counts/s to irradiance/flux is given."),
      ("PARTIALLY_RESOLVED","HEL1OS manual PDF p.51 Appendix B.3","MJD column is identified as Modified Julian Date.","The independent scale/reference convention for MJD is not explicitly stated; do not select it solely by comparing ISOT."),
      ("PARTIALLY_RESOLVED","HEL1OS manual PDF p.51 Appendix B.3; p.46 Appendix B.1","ISOT/utc-isot is labelled UTC ISO-T corresponding to MJD.","Manual does not state which of the paired MJD/ISOT columns must control in every product."),
      ("VERIFIED_FROM_AUTHORITATIVE_DOCUMENT","HEL1OS manual PDF pp.51-52 Appendix B.3","CTR is count rate in cts/sec for documented light-curve extensions.","No conversion to irradiance is implied."),
      ("VERIFIED_FROM_AUTHORITATIVE_DOCUMENT","HEL1OS manual PDF p.51 Appendix B.3","STAT_ERR is uncertainty on count rate with cts/sec unit.","Manual excerpt does not define a more specific confidence/propagation policy."),
      ("VERIFIED_FROM_AUTHORITATIVE_DOCUMENT","HEL1OS manual PDF pp.5, 13, 51-52 §§2.1, Level-1 inventory, Appendix B.3","Detector paths and LC extensions identify CdTe1/2 and CZT1/2; default broad bands are documented.","Sample headers are examples; current file values are independently listed in FITS audit."),
      ("VERIFIED_FROM_AUTHORITATIVE_DOCUMENT","HEL1OS manual PDF p.13, Level-1 product inventory","The four Level-1 GTIs span the entire dump and are relevant only to Level-2; the manual says they can be ignored for Level-1 analysis.","This verifies the Level-1 policy only; no endpoint rule is inferred for other products/levels."),
      ("STILL_UNVERIFIED","No applicable passage found in HEL1OS manual Version 1.2","Manual sample data uses V111 naming but does not define V111/V211 precedence or compatibility.","Need a versioned release/change note or registered product specification defining overlap resolution."),
      ("PARTIALLY_RESOLVED","HEL1OS manual PDF pp.4-5, 13, 53-55 §§2, 2.1, Level-1 inventory, Appendix B.4","Manual distinguishes some Level-2 pile-up/saturation processing, describes L1 HK fields and says HK/event/GTI can refine higher-level good intervals.","No Level-1 HK threshold/bit interpretation or sample rejection rule established; candidate counters remain unclassified."),
    ]
    matrix=[]
    for index,row in enumerate(original):
        status,citation,interpretation,limits=mappings[index]
        row["verification_status"]=status
        row["document_filename_version_section_page"]=citation
        row["phase4e_supported_interpretation"]=interpretation
        row["phase4e_unresolved_limits"]=limits
        row["evidence_document_path"]=str(DOWNLOADS/("SoLEXS-UserManual.pdf" if row["instrument"]=="SoLEXS" else "HEL1OS_UserManual.pdf"))
        row["evidence_sha256"]=manuals[0 if row["instrument"]=="SoLEXS" else 1]["sha256"]
        matrix.append(row)
    matrix_path=OUT/"phase4e_semantics_evidence_matrix.csv"
    with matrix_path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
    old=json.loads(OLD_REGISTER.read_text(encoding="utf-8"))
    register={"audit_time_utc":datetime.now(timezone.utc).isoformat(),"supersedes_no_prior_register":False,
              "previous_phase3c_register_path":str(OLD_REGISTER),"previous_register_preserved":True,
              "actual_downloaded_manuals":manuals,"existing_registered_documents":old.get("documents",[]),
              "manual_evidence_disposition":"See phase4e_semantics_evidence_matrix.csv; only documentary scopes supported above are promoted.",
              "phase3c_training_gate":"BLOCKED: remaining partial/unverified critical rules and Phase 3C canonical input directories still need reconciliation.",
              "rules_verified":sum(x[0]=="VERIFIED_FROM_AUTHORITATIVE_DOCUMENT" for x in mappings),
              "rules_partially_resolved":sum(x[0]=="PARTIALLY_RESOLVED" for x in mappings),
              "rules_still_unverified":sum(x[0]=="STILL_UNVERIFIED" for x in mappings)}
    (OUT/"phase4e_document_register.json").write_text(json.dumps(register,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"manuals":len(manuals),"rules_verified":register["rules_verified"],"partially_resolved":register["rules_partially_resolved"],"still_unverified":register["rules_still_unverified"],"matrix":str(matrix_path)}))


if __name__=="__main__":main()

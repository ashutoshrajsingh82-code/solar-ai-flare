<!-- phase4i_authenticated_acquisition_addendum_v1 -->
# Phase 4I authenticated evidence acquisition addendum

Audit date: 2026-10-25 UTC. This addendum preserves the prior Phase 4I attempt and records the later authenticated, normal-use PRADAN acquisition. The official route was `https://pradan.issdc.gov.in/al1/protected/miscDownloads.xhtml` (ISRO Science Data Archive / Aditya-L1 / Other Downloads). No access control was bypassed. Downloads were preserved unmodified only after confirming that the canonical target did not already exist.

## Acquired authoritative artifacts

| Canonical path | Portal title / issued metadata | Route | SHA-256 |
|---|---|---|---|
| `E:\SolarFlareData\source_docs\isro\solexs\SoLEXS_Release_Updates.txt` | `SoLEXS_Release_Updates.txt`; updated 2026-06-09 | Other Downloads / SoLEXS | `7fc0229985fef2bd60ab23bc554c09e64f526719f11825e44a60eed12f5396ce` |
| `E:\SolarFlareData\source_docs\isro\solexs\solexs_tools-1.1.tar.gz` | `solexs_tools-1.1.tar.gz`; portal size 19,730,838 bytes | Other Downloads / SoLEXS | `348014aefa50ade556ec4a160a3cfe0b08095bde84b4650a27d6b5d3bf7fa2d3` |
| `E:\SolarFlareData\source_docs\isro\hel1os\HEL1OS_Level2DataProcessingPackage.pdf` | `HEL1OS: User Guide for Level-2 Data Processing`; 2026-05-29; V07 title metadata | Other Downloads / HEL1OS | `8039a4fcb4dff8d34c06904431036f19752f8cd4f9c4143038c17478d996c127` |
| `E:\SolarFlareData\source_docs\isro\hel1os\HEL1OS_ReleaseNotes.pdf` | `Release Notes: Aditya-L1/HEL1OS Software, Product & Data Releases`; PDF creation 2026-09-03 | Other Downloads / HEL1OS | `f9955c56e66f0465a1dda77ca4ed416e35771a71911cf2a24a4937d2c61167a7` |
| `E:\SolarFlareData\source_docs\isro\hel1os\hel1os-level2proc.zip` | HEL1OS Level-2 Processing software, V07; guide dated 2026-05-29 | Other Downloads / HEL1OS | `4a6a0f3853acaf1b4bad242b620110ede545a81dac01e0bee9fd68bc218e518d` |
| `E:\SolarFlareData\source_docs\isro\hel1os\CAL_epoch20231001_CZTResponseReader.zip` | CZT CALDB Response Files epoch 20231001 and OSPEX Reader Code; portal date 2025-01-06 | Other Downloads / HEL1OS | `aece8c94880c3c75737d55d4afb374ae0343fb70723d5d8722d1619b00f3a18a` |
| `E:\SolarFlareData\source_docs\isro\hel1os\CAL_epoch20231001_CdTeResponseReader.zip` | CdTe CALDB Response Files epoch 20231001 and OSPEX Reader Code; portal date 2025-01-06 | Other Downloads / HEL1OS | `66f3862f832d501de6a3a515d3176a20e8b7432aedc57b473dda5d6f94d9396c` |
| `E:\SolarFlareData\source_docs\isro\hel1os\Level2-Product_HEL1OS_HardXRay_40-150keV_FlareCatalog.pdf` | HEL1OS-HXR-CAT-2026-V1.0; release 2026-08-18; catalog period 2024-07-14 to 2026-07-04 | Other Downloads / HEL1OS | `a9f86bdbf61d32810355824babe676cf1d9f5b88e9970cff49935d23fec83a80` |
| `E:\SolarFlareData\source_docs\isro\hel1os\Level2-Product_HEL1OS_SEP_ContaminationTimes_Catalog.pdf` | HEL1OS-SEP-CAT-2026-V1.0; release 2026-08-18; coverage 2024-07-31 to 2025-12-31 | Other Downloads / HEL1OS | `e9cf89a202b4aafde9ebd577f36456556a569c276c92b6b757339e306c274c0a` |

The pre-existing canonical manuals remain hash-valid and are retained: SoLEXS User Manual `d97c83618c85a0984e7dd0300ce5c16c4748ed5facde525949f902fe9f72abda`; HEL1OS User Manual `8efdf3aaf48d8a72cd9d85b567a8c0a6e2655ecf52c5a8d87f36915b741acb4d`.

## Adjudication outcome

The exact 14-rule adjudication is in `phase4i_authenticated_semantics_adjudication.csv`, and the canonical scanner accepted every source path/hash in `phase4i_authenticated_phase3c`. Status counts are unchanged: 7 verified, 6 partially resolved, and 1 still unverified. This is intentional: the new evidence narrows the gaps but does not supply the missing Level-1 SIS contracts.

The official V07 code treats MJD and ISOT inputs as UTC (`validate_time_format`, `isot_to_mjd`, `mjd_to_isot`), while the guide ties that code to Level-1 event and housekeeping input. This supports the current interpretation but does not specify which stored Level-1 column controls if they disagree. The release notes document a historical re-generation/supersession for July-September 2024, but do not map V111/V211 identifiers or define a rule for the observed February 2026 counterpart products. The V111/V211 rule therefore remains `STILL_UNVERIFIED`.

For quality, the guide/release notes and V07 package describe CdTe pile-up counters, CZT bunched-pixel counters, and proxy live-time correction. The release notes also state that peer-review recommendations and high-count-rate testing are still being incorporated. These are not a Level-1 flag dictionary, saturation threshold, or approved rejection policy; no filter is authorized. The SEP catalogue is explicitly diagnostic rather than exclusionary. The flare catalogue is a selected hard-X-ray subset and does not establish complete positive labels or any negative labels.

## Gates and safety

The semantics gate is `BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED`. The separate Phase 4H-derived readiness report has 2 pass and 9 blocked checks: provenance and structural readability pass; timestamps, GTI applicability, quality/saturation, version precedence, calibration application, valid coverage, positive-label sufficiency, negative-label completeness, and chronological split feasibility remain blocked.

No training, tuning, evaluation, checkpoint generation/promotion, production activation, GTI/quality filtering, V111/V211 selection/merge, or label promotion occurred.

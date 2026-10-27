# Phase 4G - Scientific semantics closure review

Audit date: 2026-10-25 UTC

## Baseline and decision

The Phase 4F canonical adjudication was rechecked before this review: 14 rules, 7 `VERIFIED_FROM_AUTHORITATIVE_DOCUMENT`, 6 `PARTIALLY_RESOLVED`, and 1 `STILL_UNVERIFIED`.  The canonical readiness status remains `BLOCKED_SCIENTIFIC_SEMANTICS_UNRESOLVED`.

No rule status is changed in Phase 4G.  The two registered manuals are still the only local instrument documents.  They establish the scoped claims already recorded in the Phase 4E adjudication but do not supply a GTI endpoint contract, a HEL1OS time-authority rule, Level-1 quality rejection criteria, or V111/V211 precedence.

## Documentation search record

The official PRADAN/ISSDC FAQ was reviewed on 2026-10-25:

- https://pradan1.issdc.gov.in/al1/faq.xhtml, HEL1OS FAQ: specifications and technical details are directed to the Aditya-L1 User Handbook in `OtherDownloads`; HEL1OS is stated to operate in Event mode with 100% duty cycle.  This is not a field-level Level-1 contract.
- https://alpps.issdc.gov.in/web/extensions/Aditya-l1/resources/docs/ALPPS_UserGuide_V1.0.pdf, pp. 9-10: describes payload/POC operation and public availability.  It does not define SoLEXS GTI columns, HEL1OS time fields, quality bits, or version precedence.
- https://www.ursc.gov.in/hel1os.jsp: instrument-science description only; it does not define FITS semantics or a selection policy.

No new document was downloaded, copied, or registered.  The public sources are preserved as discovery evidence only and are not used to promote a scientific rule.

## Rule review and required next artifact

1. **SoLEXS RATE.TIME - verified.** `SoLEXS-UserManual.pdf`, p. 6, section 2.2 establishes Unix seconds from 1970-01-01 UTC for SoLEXS time. No further action for this scoped Level-1 claim.
2. **SoLEXS RATE.COUNTS - verified.** Manual p. 5, section 2.1.2 establishes the documented L1 light-curve count-rate scope. It does not establish irradiance.
3. **SoLEXS GTI START/STOP - partially resolved.** Manual p. 5, section 2.1.1 describes valid-data criteria but not the FITS column units, relation to RATE.TIME, endpoint closure, or malformed/overlap behavior. Obtain the SoLEXS L1 FITS/SIS field dictionary or GTI specification.
4. **SoLEXS detector/path identity - verified.** Manual pp. 5-6 identifies L1 detector directories and product naming. A release/change record is still needed for cross-version precedence.
5. **SoLEXS provenance/cadence metadata - partially resolved.** Manual pp. 5-6 supports day-wise products and one-second light curves, not `CREATOR`, `DATE`, `OBS_ID`, or compatibility semantics. Obtain the L1 schema/provenance specification.
6. **SoLEXS physical-flux conversion - partially resolved.** Manual p. 7, section 2.4 requires ARF/RMF for spectral analysis but does not give an applicable RATE-to-irradiance procedure. Obtain CALDB release notes plus response-selection/conversion documentation.
7. **HEL1OS MJD - partially resolved.** `HEL1OS_UserManual.pdf`, p. 51, Appendix B.3 identifies Modified Julian Date, but does not state the authoritative time scale/reference or a disagreement policy. Obtain the HEL1OS L1 SIS/time-correlation contract.
8. **HEL1OS ISOT - partially resolved.** Manual pp. 46 and 51 label UTC ISO-T fields; it does not establish which of MJD/ISOT controls every product or mismatch handling. Obtain the same L1 time-field contract.
9. **HEL1OS CTR - verified.** Manual pp. 51-52, Appendix B.3 defines documented light-curve CTR as cts/sec. No flux conversion is implied.
10. **HEL1OS STAT_ERR - verified.** Manual p. 51 defines count-rate uncertainty units. Obtain an uncertainty-propagation policy only if downstream use requires one.
11. **HEL1OS detector/band extensions - verified.** Manual pp. 5, 13, and 51-52 documents detector identities and light-curve extensions; current values remain inventory evidence.
12. **HEL1OS Level-1 GTI use - verified.** Manual p. 13 says the four Level-1 GTIs span the dump and may be ignored for Level-1 analysis. This is not an endpoint rule or a Level-2 rule.
13. **HEL1OS V111/V211 precedence - still unverified.** The manuals and official public pages provide no supersession/release criterion. Obtain a POC/ISSDC versioned release note or product-change notice covering V111 and V211; retain all 14 counterpart groups separately until then.
14. **HEL1OS quality/HK/saturation - partially resolved.** Manual pp. 4-5, 13, and 53-55 describe products and HK names but no row/detector/band rejection thresholds or quality-bit meaning. Obtain the L1 quality/SIS and calibration/operations note.

## Coverage and safety

The 178-product FITS inventory is structural only.  No complete verified time, GTI, quality, calibration, version, event-identity, and label-completeness rule set exists.  Therefore verified observational coverage, positive windows, and negative windows remain unknown/null; uncatalogued or insufficiently observed periods are not negative examples.

No GTI/quality filter, version merge, training, tuning, evaluation, checkpoint operation, or production activation was performed.

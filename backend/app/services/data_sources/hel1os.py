"""Fail-closed HEL1OS adapter until its product schema is verified."""


def read_hel1os_product(path):
    raise NotImplementedError(
        "HEL1OS raw-product schema is not enabled: obtain the matching registered PRADAN user manual "
        "and representative product files before ingestion can be validated. No demo fallback is performed."
    )

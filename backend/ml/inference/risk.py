"""Canonical provisional score bands. These are not calibrated probabilities."""

RISK_BANDS = ((0.85, "CRITICAL"), (0.70, "HIGH"), (0.50, "MODERATE"), (0.30, "WATCH"), (0.0, "LOW"))


def risk_band_for_score(score):
    if isinstance(score, bool):
        raise ValueError("Risk score must be finite and between zero and one.")
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("Risk score must be finite and between zero and one.") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError("Risk score must be finite and between zero and one.")
    for lower_bound, label in RISK_BANDS:
        if value >= lower_bound:
            return label
    return "LOW"

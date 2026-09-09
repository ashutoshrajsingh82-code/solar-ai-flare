"""
Label generation module.

NOWCAST labels (already present as `state` from the synthetic generator /
GOES-catalogue-style mapping for real data): 0=Quiet 1=Pre-Flare 2=Flare 3=Decay

FORECAST labels: for each timestamp t and horizon h in {1h,3h,6h,12h,24h},
forecast_h(t) = 1 if an M/X-class flare ONSET occurs within (t, t+h], else 0.

Pre-flare duration used to define "onset" is configurable.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

HORIZONS_HOURS = [1, 3, 6, 12, 24]
SIGNIFICANT_CLASSES = {"M", "X"}


@dataclass
class LabelConfig:
    cadence_seconds: int = 60
    preflare_minutes: int = 30          # configurable pre-flare window duration
    horizons_hours: List[int] = None

    def __post_init__(self):
        if self.horizons_hours is None:
            self.horizons_hours = HORIZONS_HOURS


def generate_forecast_labels(df: pd.DataFrame, config: LabelConfig = None) -> pd.DataFrame:
    """
    Adds binary columns forecast_1h, forecast_3h, ... to df, based on whether
    a significant (M/X) flare onset (state transitions into 'Flare'=2 with
    flare_class in {M, X}) occurs within the forward-looking horizon window.
    """
    cfg = config or LabelConfig()
    out = df.copy().reset_index(drop=True)

    is_flare = (out["state"] == 2)
    flare_class = out.get("flare_class", pd.Series(["-"] * len(out)))
    is_significant = is_flare & flare_class.isin(SIGNIFICANT_CLASSES)

    # Onset = first step of a contiguous significant-flare run
    onset_mask = is_significant & (~is_significant.shift(1, fill_value=False))
    onset_indices = np.where(onset_mask.values)[0]

    n = len(out)
    for h in cfg.horizons_hours:
        steps = int(h * 3600 / cfg.cadence_seconds)
        target = np.zeros(n, dtype=int)
        if len(onset_indices) > 0:
            for idx in onset_indices:
                lo = max(0, idx - steps)
                target[lo:idx] = 1  # any point within `steps` before onset is positive
        out[f"forecast_{h}h"] = target

    return out


def apply_preflare_window(df: pd.DataFrame, config: LabelConfig = None) -> pd.DataFrame:
    """
    Recompute/override the 'Pre-Flare' portion of `state` using a configurable
    duration, counted backward from the first impulsive/flare index of each event.
    Useful when re-labelling with a different pre-flare definition than the
    generator default.
    """
    cfg = config or LabelConfig()
    out = df.copy().reset_index(drop=True)
    preflare_steps = int(cfg.preflare_minutes * 60 / cfg.cadence_seconds)

    is_flare = (out["state"] == 2)
    onset_mask = is_flare & (~is_flare.shift(1, fill_value=False))
    onset_indices = np.where(onset_mask.values)[0]

    # reset any previous pre-flare labelling that isn't already inside flare/decay
    for idx in onset_indices:
        lo = max(0, idx - preflare_steps)
        segment = out.loc[lo:idx - 1, "state"]
        out.loc[lo:idx - 1, "state"] = np.where(segment.isin([2, 3]), segment, 1)

    return out

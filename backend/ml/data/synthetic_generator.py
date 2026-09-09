"""
Synthetic Aditya-L1 SoLEXS + HEL1OS Data Generator
====================================================
THIS DATA IS SYNTHETIC / DEMO DATA. It is NOT real Aditya-L1 telemetry.

Simulates a physically-motivated two-channel X-ray time series:
  - SoLEXS  (soft/thermal X-ray, 1-30 keV)   -> gradual rise, delayed peak, slow decay
  - HEL1OS  (hard/non-thermal X-ray, 10-150 keV) -> impulsive rise, earlier peak, fast decay

Physical motivation (Neupert effect style behaviour):
  Hard X-rays (non-thermal electron acceleration) tend to peak BEFORE
  soft/thermal X-rays, whose gradual rise reflects plasma heating that
  integrates the hard X-ray/particle energy input over time. We simulate
  this by making the HEL1OS impulsive component lead the SoLEXS thermal
  component, and by making SoLEXS behave similarly to a "running integral"
  of the HEL1OS impulsive energy input plus its own slower cooling curve.

States simulated (per-timestamp ground truth label):
  0 = Quiet
  1 = Pre-Flare   (slow build-up before impulsive onset)
  2 = Flare       (impulsive rise through peak)
  3 = Decay       (post-peak decline back to quiet)

Flare classes (GOES-like magnitude label): B, C, M, X
Also injects: noise, missing values (NaN gaps), telemetry dropouts,
non-uniform cadence artifacts (resolved later by resampling).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import os

RNG_SEED_DEFAULT = 42

# GOES-like class -> approximate peak SoLEXS flux (W/m^2 equivalent, log-scale surrogate)
FLARE_CLASS_PEAK = {
    "B": 5e-7,
    "C": 5e-6,
    "M": 5e-5,
    "X": 5e-4,
}
FLARE_CLASS_ORDER = ["B", "C", "M", "X"]


@dataclass
class FlareEvent:
    onset_idx: int          # index of pre-flare start
    impulsive_idx: int      # index of impulsive (HEL1OS) rise start
    peak_idx: int            # index of SoLEXS peak
    decay_end_idx: int       # index where decay returns to quiet
    flare_class: str         # B/C/M/X
    peak_flux: float


def _quiet_background(n: int, rng: np.random.Generator, base=2e-8) -> np.ndarray:
    """Slow random-walk quiet-sun background flux (SoLEXS-scale)."""
    walk = rng.normal(0, 0.02, size=n).cumsum()
    walk = walk - walk.mean()
    return base * (1.0 + 0.15 * np.tanh(walk))


def _flare_shape_soft(t: np.ndarray, t_peak: float, rise_tau: float, decay_tau: float) -> np.ndarray:
    """Asymmetric rise/decay pulse used for the SoLEXS (soft, thermal) channel.
    Rise is slower/gradual; decay is slower still (thermal cooling)."""
    shape = np.where(
        t <= t_peak,
        np.exp(-((t_peak - t) ** 2) / (2 * rise_tau ** 2)),
        np.exp(-(t - t_peak) / decay_tau),
    )
    return np.clip(shape, 0, None)


def _flare_shape_hard(t: np.ndarray, t_peak: float, rise_tau: float, decay_tau: float) -> np.ndarray:
    """Sharper, more impulsive pulse used for HEL1OS (hard, non-thermal) channel."""
    shape = np.where(
        t <= t_peak,
        np.exp(-((t_peak - t) ** 2) / (2 * rise_tau ** 2)),
        np.exp(-(t - t_peak) / decay_tau),
    )
    return np.clip(shape, 0, None)


def generate_synthetic_dataset(
    n_days: float = 10.0,
    cadence_seconds: int = 60,
    n_flares: int = 14,
    seed: int = RNG_SEED_DEFAULT,
    inject_gaps: bool = True,
    inject_noise: bool = True,
) -> Tuple[pd.DataFrame, List[FlareEvent]]:
    """
    Generate a synthetic SoLEXS + HEL1OS dual-channel time series.

    Returns
    -------
    df : pd.DataFrame
        Columns: timestamp, solexs_flux, helios_flux, flare_class, state
    events : List[FlareEvent]
        Ground-truth flare event metadata (for label generation / evaluation).
    """
    rng = np.random.default_rng(seed)
    n = int(n_days * 24 * 3600 / cadence_seconds)
    timestamps = pd.date_range("2026-08-01", periods=n, freq=f"{cadence_seconds}s", tz="UTC")

    solexs = _quiet_background(n, rng, base=3e-8)
    helios = _quiet_background(n, rng, base=1e-8) * 0.6

    state = np.zeros(n, dtype=int)  # 0 quiet
    flare_class_arr = np.array(["-"] * n, dtype=object)

    events: List[FlareEvent] = []

    # Choose non-overlapping flare center windows
    min_gap = int(4 * 3600 / cadence_seconds)  # at least 4h between flare peaks
    usable_span = n - 2 * min_gap
    if usable_span <= 0:
        n_flares = 0
    centers = []
    attempts = 0
    while len(centers) < n_flares and attempts < n_flares * 50:
        c = rng.integers(min_gap, n - min_gap)
        if all(abs(c - existing) > min_gap for existing in centers):
            centers.append(c)
        attempts += 1
    centers.sort()

    for c in centers:
        flare_class = rng.choice(FLARE_CLASS_ORDER, p=[0.40, 0.35, 0.20, 0.05])
        peak_flux = FLARE_CLASS_PEAK[flare_class] * rng.uniform(0.7, 1.4)

        # Pre-flare build-up duration: 15-60 minutes
        preflare_minutes = rng.integers(15, 61)
        preflare_steps = int(preflare_minutes * 60 / cadence_seconds)

        # Impulsive (HEL1OS) rise leads SoLEXS peak by 2-8 minutes (Neupert-like lead)
        lead_minutes = rng.uniform(2, 8)
        lead_steps = int(lead_minutes * 60 / cadence_seconds)

        # Rise/decay time constants scale loosely with flare class magnitude
        class_scale = {"B": 1.0, "C": 1.3, "M": 1.8, "X": 2.5}[flare_class]
        soft_rise_tau = max(2, int(6 * class_scale))       # steps
        soft_decay_tau = max(5, int(20 * class_scale))     # steps
        hard_rise_tau = max(1, int(2 * class_scale))
        hard_decay_tau = max(2, int(6 * class_scale))

        window_before = preflare_steps + lead_steps + 5
        window_after = int(soft_decay_tau * 6)
        lo = max(0, c - window_before)
        hi = min(n, c + window_after)
        local_t = np.arange(lo, hi)

        soft_peak_idx = c
        hard_peak_idx = c - lead_steps

        soft_pulse = _flare_shape_soft(local_t, soft_peak_idx, soft_rise_tau, soft_decay_tau)
        hard_pulse = _flare_shape_hard(local_t, hard_peak_idx, hard_rise_tau, hard_decay_tau)

        solexs[lo:hi] += peak_flux * soft_pulse
        helios[lo:hi] += (peak_flux * 1.8) * hard_pulse  # hard channel relatively stronger at peak

        # Label states
        onset_idx = max(0, c - window_before)
        impulsive_idx = max(0, hard_peak_idx - hard_rise_tau)
        # find decay end: where soft pulse drops below 5% of peak
        decay_end_local = np.where(soft_pulse[np.searchsorted(local_t, c) - lo if False else 0:] < 0.05)[0]
        decay_end_idx = min(n - 1, c + soft_decay_tau * 4)

        state[onset_idx:impulsive_idx] = np.maximum(state[onset_idx:impulsive_idx], 1)   # pre-flare
        state[impulsive_idx:soft_peak_idx + max(1, soft_rise_tau)] = 2                    # flare (impulsive->peak)
        state[soft_peak_idx + max(1, soft_rise_tau):decay_end_idx] = np.where(
            state[soft_peak_idx + max(1, soft_rise_tau):decay_end_idx] == 0, 3, 3
        )
        flare_class_arr[onset_idx:decay_end_idx] = flare_class

        events.append(FlareEvent(onset_idx, impulsive_idx, soft_peak_idx, decay_end_idx, flare_class, peak_flux))

    # Instrument + photon noise
    if inject_noise:
        solexs = solexs * (1 + rng.normal(0, 0.06, n)) + rng.normal(0, solexs.std() * 0.05, n)
        helios = helios * (1 + rng.normal(0, 0.09, n)) + rng.normal(0, helios.std() * 0.05, n)
        solexs = np.clip(solexs, 1e-10, None)
        helios = np.clip(helios, 1e-10, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "solexs_flux": solexs,
        "helios_flux": helios,
        "flare_class": flare_class_arr,
        "state": state,
    })

    # Telemetry gaps (missing data) + isolated spikes, to exercise cleaning pipeline
    if inject_gaps:
        n_gaps = max(1, n // 4000)
        for _ in range(n_gaps):
            gap_start = rng.integers(0, n - 30)
            gap_len = rng.integers(3, 25)
            df.loc[gap_start:gap_start + gap_len, ["solexs_flux", "helios_flux"]] = np.nan

        # isolated spikes
        n_spikes = max(1, n // 3000)
        spike_idx = rng.choice(n, size=n_spikes, replace=False)
        df.loc[spike_idx, "solexs_flux"] *= rng.uniform(8, 20, size=n_spikes)

        # occasional duplicate timestamp (telemetry retransmission artifact)
        if n > 100:
            dup_idx = rng.integers(50, n - 50)
            dup_row = df.iloc[[dup_idx]].copy()
            df = pd.concat([df.iloc[:dup_idx + 1], dup_row, df.iloc[dup_idx + 1:]], ignore_index=True)

    return df, events


def save_demo_dataset(output_dir: str, n_days: float = 10.0, cadence_seconds: int = 60, seed: int = RNG_SEED_DEFAULT):
    os.makedirs(output_dir, exist_ok=True)
    df, events = generate_synthetic_dataset(n_days=n_days, cadence_seconds=cadence_seconds, seed=seed)
    path = os.path.join(output_dir, "demo_aditya_l1_solexs_helios.csv")
    df.to_csv(path, index=False)

    events_df = pd.DataFrame([e.__dict__ for e in events])
    events_path = os.path.join(output_dir, "demo_flare_events.csv")
    events_df.to_csv(events_path, index=False)

    return path, events_path, df, events


if __name__ == "__main__":
    p, ep, df, events = save_demo_dataset(os.path.dirname(__file__) + "/../../sample_data")
    print(f"Saved demo dataset: {p} ({len(df)} rows)")
    print(f"Saved event log: {ep} ({len(events)} events)")
    print(df["state"].value_counts())

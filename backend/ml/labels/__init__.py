"""GOES-based labeling package for solar flare nowcast and forecast targets."""

from ml.labels.goes_labels import (
    GoesLabelConfig,
    load_goes_events,
    make_labels,
    split_by_blocks,
    save_labels,
)

__all__ = [
    "GoesLabelConfig",
    "load_goes_events",
    "make_labels",
    "split_by_blocks",
    "save_labels",
]

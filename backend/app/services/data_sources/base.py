from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class IngestionResult:
    """Validated source samples and an auditable, JSON-safe report."""

    data: pd.DataFrame
    report: dict[str, Any]

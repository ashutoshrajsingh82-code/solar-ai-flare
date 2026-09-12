"""
In-memory pipeline state.

For a prototype single-process demo server, dataframes at each pipeline
stage are kept in memory (module-level singleton) so the dashboard can
visualize every stage (raw -> cleaned -> aligned -> transformed ->
featured -> labeled -> windowed) without recomputing on every request.

A production system would persist intermediate artifacts (e.g., parquet
files on disk or a feature store) instead of holding everything in RAM.
"""
import time
from typing import Optional, Dict, Any
import pandas as pd


class PipelineState:
    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.aligned_df: Optional[pd.DataFrame] = None
        self.transformed_df: Optional[pd.DataFrame] = None
        self.featured_df: Optional[pd.DataFrame] = None
        self.labeled_df: Optional[pd.DataFrame] = None

        self.dataset_meta: Dict[str, Any] = {}
        self.stage_reports: Dict[str, Any] = {}
        self.feature_columns: list = []
        self.transformer = None
        self.cadence_seconds: int = 60

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def set_stage(self, stage_name: str, df: pd.DataFrame, report: dict, duration_sec: float):
        setattr(self, f"{stage_name}_df", df)
        self.stage_reports[stage_name] = {
            **report,
            "duration_sec": round(duration_sec, 4),
            "status": "complete",
            "updated_at": time.time(),
        }

    def pipeline_status(self):
        """Returns per-stage status for the Data Pipeline dashboard page."""
        stages = ["raw", "cleaned", "aligned", "transformed", "featured", "labeled"]
        out = []
        for s in stages:
            report = self.stage_reports.get(s)
            df = getattr(self, f"{s}_df")
            out.append({
                "stage": s,
                "status": "complete" if report else "pending",
                "input_rows": report.get("input_rows") if report else None,
                "output_rows": (len(df) if df is not None else report.get("output_rows") if report else None),
                "duration_sec": report.get("duration_sec") if report else None,
                "warnings": report.get("warnings", []) if report else [],
                "missing_values": report.get("nan_count_final", report.get("missing_after_resample")) if report else None,
                "details": report or {},
            })
        return out

    def is_ready_for_features(self) -> bool:
        return self.transformed_df is not None

    def is_ready_for_training(self) -> bool:
        return self.labeled_df is not None and len(self.feature_columns) > 0


pipeline_state = PipelineState()

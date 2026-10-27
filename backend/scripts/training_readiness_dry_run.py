"""Assess pipeline eligibility without fitting, evaluating, or writing checkpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.pipeline_state import pipeline_state
from ml.training.dataset_eligibility import assess_dataset_eligibility


def main():
    report = assess_dataset_eligibility(pipeline_state.dataset_meta, {
        "train": pipeline_state.train_df, "validation": pipeline_state.val_df, "test": pipeline_state.test_df,
    })
    report["dry_run"] = {"trainer_invoked": False, "parameters_fitted": False, "checkpoint_created": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

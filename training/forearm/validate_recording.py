"""Check whether a Nexus recording is dense enough for CNN preparation.

This checker accepts the existing combined Nexus CSV/XLSX format. It does not
train a model and it never creates labels from sensor values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Timestamp",
    "Seq",
    "STM32 ms",
    "sEMG raw",
    "Quat W",
    "Quat X",
    "Quat Y",
    "Quat Z",
]
QUATERNION_COLUMNS = ["Quat W", "Quat X", "Quat Y", "Quat Z"]


def load_recording(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError("Use a CSV or Excel recording file.")


def add_recording_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sequence = pd.to_numeric(result["Seq"], errors="coerce")
    stm32_ms = pd.to_numeric(result["STM32 ms"], errors="coerce")
    reset = sequence.diff().lt(0) | stm32_ms.diff().lt(0)
    result["recording_group"] = reset.fillna(False).cumsum() + 1
    return result


def validate(frame: pd.DataFrame) -> dict:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        return {
            "status": "not_ready",
            "reason": "missing_columns",
            "missing_columns": missing_columns,
        }

    working = add_recording_groups(frame)
    for column in ["Seq", "STM32 ms", "sEMG raw", *QUATERNION_COLUMNS]:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    valid_rows = working.dropna(subset=REQUIRED_COLUMNS).copy()
    intervals = (
        valid_rows.groupby("recording_group")["STM32 ms"]
        .diff()
        .loc[lambda values: values.gt(0)]
    )
    median_interval_ms = (
        float(intervals.median()) if not intervals.empty else None
    )
    estimated_rate_hz = (
        1000.0 / median_interval_ms
        if median_interval_ms is not None and median_interval_ms > 0
        else None
    )

    quaternion = valid_rows[QUATERNION_COLUMNS].to_numpy(dtype=float)
    quaternion_norm_error = (
        float(np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0)))
        if len(quaternion)
        else None
    )

    group_sizes = {
        str(int(group)): int(size)
        for group, size in valid_rows.groupby("recording_group").size().items()
    }
    enough_rows = any(size >= 200 for size in group_sizes.values())
    cadence_ready = bool(
        estimated_rate_hz is not None and estimated_rate_hz >= 80.0
    )
    quaternion_ready = bool(
        quaternion_norm_error is not None and quaternion_norm_error <= 0.02
    )
    trainable = bool(enough_rows and cadence_ready and quaternion_ready)

    reasons = []
    if not enough_rows:
        reasons.append("No recording group contains at least 200 rows.")
    if not cadence_ready:
        reasons.append("The fused rows are slower than the 80 Hz minimum.")
    if not quaternion_ready:
        reasons.append("The quaternion values require review.")

    return {
        "status": "ready_for_preparation" if trainable else "not_ready",
        "rows_loaded": int(len(frame)),
        "valid_rows": int(len(valid_rows)),
        "recording_groups": group_sizes,
        "median_interval_ms": median_interval_ms,
        "estimated_fused_rate_hz": estimated_rate_hz,
        "maximum_quaternion_norm_error": quaternion_norm_error,
        "has_training_labels": all(
            column in frame.columns
            for column in ["movement_label", "fatigue_label"]
        ),
        "reasons": reasons,
        "note": (
            "This checks recording density and structure only. Separate "
            "high-rate sEMG logging and reviewed labels are still required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--sheet", default=0)
    args = parser.parse_args()

    frame = load_recording(args.recording, args.sheet)
    print(json.dumps(validate(frame), indent=2))


if __name__ == "__main__":
    main()


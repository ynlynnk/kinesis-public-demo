from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


REQUIRED_SENSOR_COLUMNS = [
    "Timestamp",
    "Seq",
    "STM32 ms",
    "sEMG raw",
    "Quat W",
    "Quat X",
    "Quat Y",
    "Quat Z",
    "Calib",
]

NUMERIC_SENSOR_COLUMNS = [
    "Seq",
    "STM32 ms",
    "sEMG raw",
    "Quat W",
    "Quat X",
    "Quat Y",
    "Quat Z",
    "Calib",
]

NEXUS_WINDOW_ROWS = 256
MINIMUM_SEMG_RATE_HZ = 180.0


def load_sensor_frame(
    source_url: str,
    local_path: str | Path,
) -> tuple[pd.DataFrame, str, bool]:
    """Load the configured CSV feed or the bundled connectivity-test data."""

    if source_url:
        frame = pd.read_csv(source_url)
        return frame, "Configured live CSV feed", True

    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor test data was not found: {path}")

    frame = pd.read_csv(path)
    return frame, "Bundled Nexus connectivity-test snapshot", False


def prepare_sensor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and derive monitoring fields without changing raw values."""

    missing_columns = [
        column
        for column in REQUIRED_SENSOR_COLUMNS
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "The sensor feed is missing required columns: "
            + ", ".join(missing_columns)
        )

    prepared = frame.copy()
    prepared["Timestamp"] = pd.to_datetime(
        prepared["Timestamp"],
        errors="coerce",
        format="mixed",
    )
    for column in NUMERIC_SENSOR_COLUMNS:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared = prepared.sort_values("Timestamp", kind="stable").reset_index(
        drop=True
    )

    sequence_reset = prepared["Seq"].diff().fillna(0) < 0
    clock_reset = prepared["STM32 ms"].diff().fillna(0) < 0
    prepared["Session"] = (sequence_reset | clock_reset).cumsum() + 1

    same_session = prepared["Session"].eq(prepared["Session"].shift())
    prepared["Server interval (s)"] = (
        prepared["Timestamp"].diff().dt.total_seconds().where(same_session)
    )
    prepared["Device interval (s)"] = (
        prepared["STM32 ms"].diff().div(1000).where(same_session)
    )

    quaternion_columns = ["Quat W", "Quat X", "Quat Y", "Quat Z"]
    prepared["Quaternion norm"] = np.sqrt(
        prepared[quaternion_columns].pow(2).sum(axis=1)
    )

    w = prepared["Quat W"]
    x = prepared["Quat X"]
    y = prepared["Quat Y"]
    z = prepared["Quat Z"]

    roll_radians = np.arctan2(
        2 * (w * x + y * z),
        1 - 2 * (x.pow(2) + y.pow(2)),
    )
    pitch_input = np.clip(2 * (w * y - z * x), -1, 1)
    pitch_radians = np.arcsin(pitch_input)
    yaw_radians = np.arctan2(
        2 * (w * z + x * y),
        1 - 2 * (y.pow(2) + z.pow(2)),
    )

    prepared["Roll (deg)"] = np.degrees(roll_radians)
    prepared["Pitch (deg)"] = np.degrees(pitch_radians)
    prepared["Yaw (deg)"] = np.degrees(yaw_radians)

    return prepared


def summarize_sensor_frame(
    frame: pd.DataFrame,
    checkpoint_file: str | Path | None = None,
) -> dict:
    """Return connection and Model 1-readiness measurements."""

    if frame.empty:
        raise ValueError("The sensor feed does not contain any rows.")

    valid_frame = frame.dropna(
        subset=["Timestamp", "Seq", "STM32 ms", "sEMG raw"]
    )
    if valid_frame.empty:
        raise ValueError("The sensor feed has no complete timestamped rows.")

    intervals = valid_frame["Device interval (s)"].dropna()
    positive_intervals = intervals[intervals > 0]
    median_interval = (
        float(positive_intervals.median())
        if not positive_intervals.empty
        else None
    )
    effective_rate = (
        1 / median_interval
        if median_interval is not None and median_interval > 0
        else None
    )

    same_session = valid_frame["Session"].eq(
        valid_frame["Session"].shift()
    )
    sequence_step = valid_frame["Seq"].diff()
    sequence_gaps = int(((sequence_step != 1) & same_session).sum())

    quaternion_error = (
        valid_frame["Quaternion norm"].sub(1).abs().max()
    )
    quaternion_valid = bool(
        pd.notna(quaternion_error) and quaternion_error <= 0.02
    )

    latest = valid_frame.iloc[-1]
    has_training_labels = all(
        column in valid_frame.columns
        for column in [
            "fatigue_label",
            "movement_label",
        ]
    )

    session_sizes = valid_frame.groupby("Session", sort=False).size()
    complete_nexus_windows = int(
        sum(max(0, int(size) - NEXUS_WINDOW_ROWS + 1) for size in session_sizes)
    )
    checkpoint_available = bool(
        checkpoint_file is not None and Path(checkpoint_file).is_file()
    )
    cadence_ready = bool(
        effective_rate is not None
        and effective_rate >= MINIMUM_SEMG_RATE_HZ
    )
    nexus_input_ready = bool(
        sequence_gaps == 0
        and quaternion_valid
        and cadence_ready
        and complete_nexus_windows > 0
    )

    readiness_rows = [
        {
            "Requirement": "Feed loaded",
            "Observed": f"{len(valid_frame)} timestamped rows",
            "Status": "Ready",
        },
        {
            "Requirement": "Sequence continuity",
            "Observed": f"{sequence_gaps} missing sequence steps",
            "Status": "Ready" if sequence_gaps == 0 else "Review",
        },
        {
            "Requirement": "Quaternion validity",
            "Observed": (
                f"Maximum norm error {float(quaternion_error):.4f}"
                if pd.notna(quaternion_error)
                else "Unavailable"
            ),
            "Status": "Ready" if quaternion_valid else "Review",
        },
        {
            "Requirement": "sEMG input",
            "Observed": "1 raw channel; matches the Nexus CNN input contract",
            "Status": "Ready",
        },
        {
            "Requirement": "Recorded-row cadence",
            "Observed": (
                f"{median_interval:.4f} s between logged rows "
                f"({effective_rate:.2f} Hz)"
                if effective_rate is not None
                else "Unable to calculate"
            ),
            "Status": "Ready" if cadence_ready else "Not ready",
        },
        {
            "Requirement": "Orientation input",
            "Observed": "4 quaternion channels; matches the Nexus CNN contract",
            "Status": "Ready" if quaternion_valid else "Review",
        },
        {
            "Requirement": "CNN input window",
            "Observed": (
                f"{complete_nexus_windows} complete contiguous "
                f"{NEXUS_WINDOW_ROWS}-row window(s)"
            ),
            "Status": "Ready" if complete_nexus_windows else "Not ready",
        },
        {
            "Requirement": "CNN checkpoint file",
            "Observed": (
                "Selected real-data checkpoint installed"
                if checkpoint_available
                else "No checkpoint installed"
            ),
            "Status": "Available" if checkpoint_available else "Not ready",
        },
        {
            "Requirement": "Training labels",
            "Observed": (
                "Available"
                if has_training_labels
                else "No real fatigue or movement labels"
            ),
            "Status": (
                "Available"
                if has_training_labels
                else "Needed for validation"
            ),
        },
    ]

    return {
        "row_count": int(len(valid_frame)),
        "session_count": int(valid_frame["Session"].nunique()),
        "latest_timestamp": latest["Timestamp"],
        "latest_sequence": int(latest["Seq"]),
        "latest_semg": float(latest["sEMG raw"]),
        "latest_calibration": int(latest["Calib"]),
        "latest_roll": float(latest["Roll (deg)"]),
        "latest_pitch": float(latest["Pitch (deg)"]),
        "latest_yaw": float(latest["Yaw (deg)"]),
        "median_interval_seconds": median_interval,
        "effective_rate_hz": effective_rate,
        "minimum_semg_rate_hz": MINIMUM_SEMG_RATE_HZ,
        "cadence_ready": cadence_ready,
        "sequence_gaps": sequence_gaps,
        "quaternion_valid": quaternion_valid,
        "nexus_window_rows": NEXUS_WINDOW_ROWS,
        "complete_nexus_windows": complete_nexus_windows,
        "nexus_input_ready": nexus_input_ready,
        "checkpoint_available": checkpoint_available,
        "readiness": readiness_rows,
    }


def calculate_feed_age_seconds(
    latest_timestamp: pd.Timestamp,
    timezone_name: str,
) -> float:
    """Calculate feed age while treating device timestamps as local time."""

    timezone = ZoneInfo(timezone_name)
    timestamp = pd.Timestamp(latest_timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)

    now = pd.Timestamp.now(tz=timezone)
    return float((now - timestamp).total_seconds())

"""Build the stable end-of-session packet used by storage and Model 2.

The packet deliberately keeps physical repetition counts separate from CNN
analysis-window counts.  A repetition count is included only when the edge
device or a reviewed annotation supplies it.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


SESSION_PACKET_SCHEMA_VERSION = "1.0"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _safe_identifier(value: str, label: str) -> str:
    clean = str(value).strip()
    if not IDENTIFIER_PATTERN.fullmatch(clean):
        raise ValueError(
            f"{label} must contain only letters, numbers, underscores or hyphens."
        )
    return clean


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def build_repetition_summary(
    total_repetitions: int | None = None,
    correct_repetitions: int | None = None,
    source: str = "edge_counter",
) -> dict[str, Any]:
    """Return honest repetition fields without substituting CNN windows."""

    if total_repetitions is None and correct_repetitions is None:
        return {
            "available": False,
            "status": "Awaiting edge repetition counter",
            "source": source,
            "total_repetitions": None,
            "correct_repetitions": None,
            "incorrect_repetitions": None,
            "correct_repetition_percent": None,
        }
    if total_repetitions is None or correct_repetitions is None:
        raise ValueError(
            "Total and correct repetitions must be supplied together."
        )
    total = int(total_repetitions)
    correct = int(correct_repetitions)
    if total < 0 or correct < 0 or correct > total:
        raise ValueError(
            "Repetition counts must satisfy 0 <= correct <= total."
        )
    incorrect = total - correct
    return {
        "available": True,
        "status": "Repetition count received",
        "source": source,
        "total_repetitions": total,
        "correct_repetitions": correct,
        "incorrect_repetitions": incorrect,
        "correct_repetition_percent": (
            round(100.0 * correct / total, 1) if total else None
        ),
    }


def _movement_window_summary(model1_result: dict[str, Any] | None) -> dict:
    windows = list((model1_result or {}).get("window_predictions", []))
    if not windows:
        return {
            "available": False,
            "analysed_windows": 0,
            "matching_windows": None,
            "movement_consistency_percent": None,
            "note": "No per-window movement results were supplied.",
        }
    incorrect = sum(
        float(window.get("movement_incorrect_probability", 1.0)) >= 0.5
        for window in windows
    )
    matching = len(windows) - incorrect
    return {
        "available": True,
        "analysed_windows": len(windows),
        "matching_windows": matching,
        "movement_consistency_percent": round(
            100.0 * matching / len(windows), 1
        ),
        "note": (
            "These are CNN analysis windows, not counted exercise repetitions."
        ),
    }


def _model1_packet(model1_result: dict[str, Any] | None) -> dict:
    if not model1_result:
        return {
            "status": "not_run",
            "release_status": "withheld",
            "movement": None,
            "fatigue": None,
            "release_reasons": ["No Model 1 result was supplied."],
        }
    released = bool(model1_result.get("prediction_released", False))
    return {
        "status": "completed",
        "release_status": "released" if released else "preview_withheld",
        "movement": model1_result.get("movement"),
        "fatigue": model1_result.get("fatigue"),
        "release_reasons": list(model1_result.get("release_reasons", [])),
        "source_rate_hz": model1_result.get("source_rate_hz"),
        "model_rate_hz": model1_result.get("model_rate_hz"),
    }


def build_session_packet(
    sensor_data: pd.DataFrame,
    assessment: dict[str, Any],
    model1_result: dict[str, Any] | None,
    *,
    patient_id: str,
    session_id: str,
    muscle_group: str,
    exercise: str,
    total_repetitions: int | None = None,
    correct_repetitions: int | None = None,
    repetition_source: str = "edge_counter",
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Create one JSON-safe session packet from the latest recording."""

    patient = _safe_identifier(patient_id, "Patient ID")
    session = _safe_identifier(session_id, "Session ID")
    if sensor_data.empty:
        raise ValueError("A session packet cannot be built from an empty feed.")

    latest_session_number = sensor_data["Session"].iloc[-1]
    latest = sensor_data[
        sensor_data["Session"] == latest_session_number
    ].sort_values(["Seq", "STM32 ms"], kind="stable")
    timestamps = pd.to_datetime(latest["Timestamp"], errors="coerce").dropna()
    recording_id = (
        str(latest["Recording ID"].dropna().iloc[-1])
        if "Recording ID" in latest.columns
        and not latest["Recording ID"].dropna().empty
        else f"{patient}-{session}"
    )

    packet = {
        "schema_version": SESSION_PACKET_SCHEMA_VERSION,
        "packet_type": "kinesis.completed_session",
        "patient_id": patient,
        "session_id": session,
        "recording_id": recording_id,
        "session_status": "completed",
        "session_started_at": (
            timestamps.iloc[0].isoformat() if not timestamps.empty else None
        ),
        "session_completed_at": (
            completed_at
            or (
                timestamps.iloc[-1].isoformat()
                if not timestamps.empty
                else datetime.now(timezone.utc).isoformat()
            )
        ),
        "exercise": {
            "muscle_group": str(muscle_group),
            "name": str(exercise),
        },
        "sampling": {
            "semg_rate_hz": assessment.get("observed_semg_rate_hz"),
            "expected_semg_rate_hz": assessment.get("expected_semg_rate_hz"),
            "expected_imu_rate_hz": assessment.get("expected_imu_rate_hz"),
            "sensor_row_count": int(len(latest)),
        },
        "signal_quality": assessment.get(
            "signal_quality",
            {
                "status": "Unavailable",
                "level": "yellow",
                "blocking_issues": ["No signal-quality assessment supplied."],
                "warnings": [],
                "metrics": {},
            },
        ),
        "repetition_summary": build_repetition_summary(
            total_repetitions,
            correct_repetitions,
            repetition_source,
        ),
        "movement_window_summary": _movement_window_summary(model1_result),
        "model1": _model1_packet(model1_result),
        "storage": {
            "current_mode": "local_packet",
            "future_transport": "firebase",
            "raw_sensor_delivery": "dense_time-stamped_batches",
        },
    }
    return _json_value(packet)


def packet_to_json(packet: dict[str, Any]) -> str:
    return json.dumps(_json_value(packet), indent=2, ensure_ascii=False)


def save_session_packet(
    packet: dict[str, Any],
    root: str | Path,
) -> Path:
    """Save a completed packet locally until Firebase persistence is enabled."""

    patient_id = _safe_identifier(packet["patient_id"], "Patient ID")
    session_id = _safe_identifier(packet["session_id"], "Session ID")
    destination = Path(root) / patient_id / session_id / "session_packet.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(packet_to_json(packet), encoding="utf-8")
    return destination

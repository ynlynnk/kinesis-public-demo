"""Convert dense Firebase recording batches into the KINESIS sensor frame.

Firebase is the transport and storage layer.  A two-second upload must contain
all 1,000 sEMG samples and 100 IMU samples; it must never contain only one
summary value for the whole interval.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from realtime_input_contract import (
    EXPECTED_IMU_RATE_HZ,
    EXPECTED_SEMG_RATE_HZ,
)


def _as_batches(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("The Firebase payload must be an object or a list.")
    if "batches" in payload:
        batches = payload["batches"]
        if isinstance(batches, dict):
            return list(batches.values())
        if isinstance(batches, list):
            return batches
    if "semg_samples" in payload and "imu_samples" in payload:
        return [payload]
    values = list(payload.values())
    if values and all(isinstance(value, dict) for value in values):
        return values
    raise ValueError("No Firebase sensor batches were found in the payload.")


def _require_fields(item: dict, fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if field not in item]
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(missing)}")


def firebase_payload_to_frame(payload) -> pd.DataFrame:
    """Return rows compatible with :func:`sensor_feed.prepare_sensor_frame`."""

    rows = []
    batches = _as_batches(payload)
    recording_ids = {
        str(batch.get("recording_id", "")).strip() for batch in batches
    }
    if len(recording_ids) > 1:
        raise ValueError(
            "One payload may contain batches from only one recording."
        )

    for batch_index, batch in enumerate(batches, start=1):
        _require_fields(
            batch,
            (
                "recording_id",
                "server_time_iso",
                "semg_rate_hz",
                "imu_rate_hz",
                "semg_samples",
                "imu_samples",
            ),
            f"Batch {batch_index}",
        )
        if float(batch["semg_rate_hz"]) != EXPECTED_SEMG_RATE_HZ:
            raise ValueError(
                f"Batch {batch_index} declares {batch['semg_rate_hz']} Hz sEMG; "
                f"{EXPECTED_SEMG_RATE_HZ:.0f} Hz is required."
            )
        if float(batch["imu_rate_hz"]) != EXPECTED_IMU_RATE_HZ:
            raise ValueError(
                f"Batch {batch_index} declares {batch['imu_rate_hz']} Hz IMU; "
                f"{EXPECTED_IMU_RATE_HZ:.0f} Hz is required."
            )

        semg_samples = list(batch["semg_samples"])
        imu_samples = list(batch["imu_samples"])
        if len(semg_samples) < 2 or len(imu_samples) < 2:
            raise ValueError(
                f"Batch {batch_index} is a sparse summary, not a dense sensor batch."
            )
        for sample in semg_samples:
            _require_fields(
                sample,
                ("seq", "device_time_us", "raw"),
                f"Batch {batch_index} sEMG sample",
            )
        for sample in imu_samples:
            _require_fields(
                sample,
                (
                    "device_time_us",
                    "quat_w",
                    "quat_x",
                    "quat_y",
                    "quat_z",
                    "calib",
                ),
                f"Batch {batch_index} IMU sample",
            )

        semg_times = np.asarray(
            [sample["device_time_us"] for sample in semg_samples],
            dtype=np.int64,
        )
        imu_times = np.asarray(
            [sample["device_time_us"] for sample in imu_samples],
            dtype=np.int64,
        )
        if np.any(np.diff(semg_times) <= 0) or np.any(np.diff(imu_times) <= 0):
            raise ValueError(
                f"Batch {batch_index} device timestamps must increase strictly."
            )

        semg_rate = 1_000_000.0 / float(np.median(np.diff(semg_times)))
        imu_rate = 1_000_000.0 / float(np.median(np.diff(imu_times)))
        if not np.isclose(semg_rate, EXPECTED_SEMG_RATE_HZ, rtol=0.10):
            raise ValueError(
                f"Batch {batch_index} measured sEMG cadence is {semg_rate:.1f} Hz."
            )
        if not np.isclose(imu_rate, EXPECTED_IMU_RATE_HZ, rtol=0.10):
            raise ValueError(
                f"Batch {batch_index} measured IMU cadence is {imu_rate:.1f} Hz."
            )

        start_time = pd.Timestamp(batch["server_time_iso"])
        if start_time.tzinfo is None:
            raise ValueError(
                f"Batch {batch_index} server_time_iso must include a timezone."
            )
        first_device_time = int(semg_times[0])
        imu_indices = np.searchsorted(imu_times, semg_times, side="right") - 1
        imu_indices = np.clip(imu_indices, 0, len(imu_samples) - 1)

        for sample, device_time, imu_index in zip(
            semg_samples, semg_times, imu_indices
        ):
            imu = imu_samples[int(imu_index)]
            timestamp = start_time + pd.to_timedelta(
                int(device_time) - first_device_time, unit="us"
            )
            rows.append(
                {
                    "Timestamp": timestamp,
                    "Seq": sample["seq"],
                    "STM32 ms": float(device_time) / 1000.0,
                    "sEMG raw": sample["raw"],
                    "sEMG filtered": sample.get("filtered"),
                    "Quat W": imu["quat_w"],
                    "Quat X": imu["quat_x"],
                    "Quat Y": imu["quat_y"],
                    "Quat Z": imu["quat_z"],
                    "Calib": imu["calib"],
                    "Recording ID": batch["recording_id"],
                    "Participant ID": batch.get("participant_id", ""),
                    "Patient ID": batch.get(
                        "patient_id", batch.get("participant_id", "")
                    ),
                    "Session ID": batch.get("session_id", ""),
                    "Batch ID": batch.get("batch_id", batch_index),
                }
            )

    if not rows:
        raise ValueError("The Firebase payload did not contain sensor samples.")
    return pd.DataFrame(rows)


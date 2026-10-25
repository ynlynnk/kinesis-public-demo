"""End-to-end inference for dense KINESIS device recordings.

The public-dataset CNNs are allowed to execute for engineering verification,
but a result is not released as validated until the checkpoint has also been
validated on labelled KINESIS-device recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forearm_fatigue_runtime import (
    load_forearm_fatigue_model,
    predict_normalized_fatigue_window,
)
from forearm_movement_runtime import (
    load_forearm_movement_model,
    predict_normalized_forearm_window,
)
from model1_signal_pipeline import (
    MODEL_SEMG_RATE_HZ,
    prepare_semg_for_model,
    resample_signal,
)
from quadriceps_runtime import (
    load_quadriceps_fatigue_model,
    load_quadriceps_movement_model,
    predict_quadriceps_fatigue_window,
    predict_quadriceps_movement_window,
)
from realtime_input_contract import (
    BASELINE_SECONDS,
    EXPECTED_IMU_RATE_HZ,
    assess_realtime_sensor_frame,
)


MINIMUM_DECISION_CONFIDENCE = 0.60
MINIMUM_DECISION_MARGIN = 0.15


class RealtimeInputNotReady(ValueError):
    """Raised when a live window fails one or more sensor-quality gates."""

    def __init__(self, assessment: dict):
        self.assessment = assessment
        failed = [
            item["check"]
            for item in assessment["checks"]
            if not item["passed"]
        ]
        super().__init__("Input quality checks failed: " + ", ".join(failed))


def _checkpoint_is_device_validated(checkpoint: dict) -> bool:
    return bool(checkpoint.get("validated_on_kinesis_data", False))


def _decision_gate(prediction: dict) -> dict:
    probabilities = sorted(
        (float(value) for value in prediction["probabilities"].values()),
        reverse=True,
    )
    confidence = probabilities[0]
    margin = confidence - probabilities[1] if len(probabilities) > 1 else 1.0
    accepted = bool(
        confidence >= MINIMUM_DECISION_CONFIDENCE
        and margin >= MINIMUM_DECISION_MARGIN
    )
    result = dict(prediction)
    result.update(
        {
            "decision_accepted": accepted,
            "decision_margin": margin,
            "display_label": prediction["label"] if accepted else "Uncertain",
        }
    )
    return result


def _resampled_quaternion_window(
    session: pd.DataFrame,
    source_rate_hz: int,
    target_samples: int,
) -> np.ndarray:
    duration_seconds = target_samples / EXPECTED_IMU_RATE_HZ
    source_samples = int(np.ceil(duration_seconds * source_rate_hz))
    source = session[["Quat W", "Quat X", "Quat Y", "Quat Z"]].tail(
        source_samples
    )
    if len(source) < source_samples:
        raise ValueError("The latest session does not contain enough IMU data.")
    values = source.to_numpy(dtype=np.float32)
    resampled = np.stack(
        [
            resample_signal(values[:, index], source_rate_hz, int(EXPECTED_IMU_RATE_HZ))
            for index in range(4)
        ],
        axis=1,
    )
    if len(resampled) < target_samples:
        raise ValueError("The resampled IMU window is incomplete.")
    resampled = resampled[-target_samples:]
    norms = np.linalg.norm(resampled, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("The resampled IMU window contains an invalid quaternion.")
    return (resampled / norms).astype(np.float32)


def predict_live_model1_session(
    sensor_data: pd.DataFrame,
    muscle_group: str,
    movement_model_path: str | Path,
    fatigue_model_path: str | Path,
) -> dict:
    """Run both trained heads after enforcing the live input contract."""

    assessment = assess_realtime_sensor_frame(sensor_data)
    if not assessment["ready"]:
        raise RealtimeInputNotReady(assessment)

    session_number = sensor_data["Session"].iloc[-1]
    session = sensor_data[sensor_data["Session"] == session_number].copy()
    session = session.sort_values(["Seq", "STM32 ms"], kind="stable")
    source_rate_hz = int(round(assessment["observed_semg_rate_hz"]))
    raw_signal = session["sEMG raw"].to_numpy(dtype=np.float32)
    baseline_count = int(round(BASELINE_SECONDS * source_rate_hz))
    baseline = raw_signal[:baseline_count]
    baseline_mean = float(np.mean(baseline))
    baseline_standard_deviation = float(np.std(baseline))
    model_signal = prepare_semg_for_model(
        raw_signal,
        baseline_mean,
        baseline_standard_deviation,
        source_rate_hz=source_rate_hz,
    )
    baseline_model_samples = int(round(BASELINE_SECONDS * MODEL_SEMG_RATE_HZ))
    post_baseline_signal = model_signal[baseline_model_samples:]
    if len(post_baseline_signal) < 512:
        raise ValueError("The post-baseline Model 1 window is incomplete.")

    movement_path = str(Path(movement_model_path))
    fatigue_path = str(Path(fatigue_model_path))
    normalized_group = muscle_group.strip().casefold()
    if normalized_group == "forearm":
        _, movement_checkpoint = load_forearm_movement_model(movement_path)
        _, fatigue_checkpoint = load_forearm_fatigue_model(fatigue_path)
        movement = predict_normalized_forearm_window(
            post_baseline_signal[-256:], movement_path
        )
        fatigue = predict_normalized_fatigue_window(
            post_baseline_signal[-512:], fatigue_path
        )
    elif normalized_group == "quadriceps":
        _, movement_checkpoint = load_quadriceps_movement_model(movement_path)
        _, fatigue_checkpoint = load_quadriceps_fatigue_model(fatigue_path)
        imu_samples = int(movement_checkpoint["imu_window_samples"])
        quaternions = _resampled_quaternion_window(
            session, source_rate_hz, imu_samples
        )
        movement = predict_quadriceps_movement_window(
            post_baseline_signal[-int(movement_checkpoint["window_samples"]):],
            quaternions,
            movement_path,
        )
        fatigue = predict_quadriceps_fatigue_window(
            post_baseline_signal[-int(fatigue_checkpoint["window_samples"]):],
            fatigue_path,
        )
    else:
        raise ValueError(f"Unsupported muscle group: {muscle_group}")

    movement = _decision_gate(movement)
    fatigue = _decision_gate(fatigue)
    device_validated = bool(
        _checkpoint_is_device_validated(movement_checkpoint)
        and _checkpoint_is_device_validated(fatigue_checkpoint)
    )
    decisions_accepted = bool(
        movement["decision_accepted"] and fatigue["decision_accepted"]
    )
    prediction_released = bool(device_validated and decisions_accepted)
    reasons = []
    if not device_validated:
        reasons.append(
            "the public-dataset checkpoints have not yet been validated on "
            "labelled KINESIS-device recordings"
        )
    if not decisions_accepted:
        reasons.append("one or more outputs failed the uncertainty gate")

    return {
        "input_assessment": assessment,
        "movement": movement,
        "fatigue": fatigue,
        "device_validated": device_validated,
        "prediction_released": prediction_released,
        "release_reasons": reasons,
        "source_rate_hz": source_rate_hz,
        "baseline_samples": baseline_count,
        "model_rate_hz": MODEL_SEMG_RATE_HZ,
    }


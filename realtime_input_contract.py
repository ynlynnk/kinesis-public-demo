"""Quality gates for real-time KINESIS sensor recordings.

These checks run before Model 1 inference.  They deliberately reject sparse,
mis-timed, incomplete, or flat sensor windows instead of allowing the CNN to
produce an apparently confident result from data unlike its input contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPECTED_SEMG_RATE_HZ = 500.0
EXPECTED_IMU_RATE_HZ = 50.0
RATE_TOLERANCE_FRACTION = 0.10
BASELINE_SECONDS = 10.0
POST_BASELINE_SECONDS = 2.56
MAX_QUATERNION_NORM_ERROR = 0.02
MIN_SIGNAL_STANDARD_DEVIATION = 1e-3
MAX_INTERVAL_OUTLIER_FRACTION = 0.02
MAX_EMG_SPIKE_FRACTION = 0.02
ROBUST_SPIKE_THRESHOLD = 12.0


def _check(name: str, passed: bool, observed: str) -> dict:
    return {
        "check": name,
        "passed": bool(passed),
        "observed": observed,
    }


def _robust_spike_fraction(signal: np.ndarray) -> float:
    """Estimate abrupt sEMG artefacts without assuming an ADC voltage range."""

    if len(signal) < 3:
        return 0.0
    differences = np.diff(signal.astype(np.float64))
    median = float(np.median(differences))
    absolute_deviation = np.abs(differences - median)
    mad = float(np.median(absolute_deviation))
    if not np.isfinite(mad) or mad <= 1e-12:
        return float(np.mean(absolute_deviation > 0.0))
    robust_scale = 1.4826 * mad
    return float(
        np.mean(absolute_deviation > ROBUST_SPIKE_THRESHOLD * robust_scale)
    )


def assess_realtime_sensor_frame(frame: pd.DataFrame) -> dict:
    """Assess the latest recording against the device-to-model contract."""

    required = {
        "Session",
        "Seq",
        "STM32 ms",
        "sEMG raw",
        "Quat W",
        "Quat X",
        "Quat Y",
        "Quat Z",
        "Quaternion norm",
        "Device interval (s)",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "The prepared sensor frame is missing: " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("The prepared sensor frame is empty.")

    latest_session = frame[frame["Session"] == frame["Session"].iloc[-1]].copy()
    latest_session = latest_session.sort_values(
        ["Seq", "STM32 ms"], kind="stable"
    )

    numeric_columns = [
        "Seq",
        "STM32 ms",
        "sEMG raw",
        "Quat W",
        "Quat X",
        "Quat Y",
        "Quat Z",
    ]
    numeric_values = latest_session[numeric_columns].to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(numeric_values).all())

    sequence_steps = latest_session["Seq"].diff().dropna()
    missing_sequence_count = int(
        sum(max(int(round(step)) - 1, 0) for step in sequence_steps)
    )
    duplicate_sequence_count = int(sequence_steps.eq(0).sum())
    backward_sequence_count = int(sequence_steps.lt(0).sum())
    sequence_contiguous = bool(
        not sequence_steps.empty and sequence_steps.eq(1).all()
    )

    intervals = latest_session["Device interval (s)"].dropna()
    intervals = intervals[intervals > 0]
    median_interval = float(intervals.median()) if not intervals.empty else None
    observed_rate = (
        1.0 / median_interval
        if median_interval is not None and median_interval > 0
        else None
    )
    minimum_rate = EXPECTED_SEMG_RATE_HZ * (1.0 - RATE_TOLERANCE_FRACTION)
    maximum_rate = EXPECTED_SEMG_RATE_HZ * (1.0 + RATE_TOLERANCE_FRACTION)
    rate_valid = bool(
        observed_rate is not None
        and minimum_rate <= observed_rate <= maximum_rate
    )
    expected_interval = 1.0 / EXPECTED_SEMG_RATE_HZ
    interval_outlier_fraction = (
        float(
            np.mean(
                np.abs(intervals.to_numpy(dtype=np.float64) - expected_interval)
                > expected_interval * RATE_TOLERANCE_FRACTION
            )
        )
        if not intervals.empty
        else 1.0
    )
    cadence_stable = bool(
        interval_outlier_fraction <= MAX_INTERVAL_OUTLIER_FRACTION
    )

    quaternion_error = latest_session["Quaternion norm"].sub(1.0).abs()
    maximum_quaternion_error = (
        float(quaternion_error.max()) if not quaternion_error.empty else None
    )
    quaternion_valid = bool(
        maximum_quaternion_error is not None
        and maximum_quaternion_error <= MAX_QUATERNION_NORM_ERROR
    )

    signal = latest_session["sEMG raw"].to_numpy(dtype=np.float64)
    signal_standard_deviation = (
        float(np.std(signal)) if len(signal) else 0.0
    )
    signal_varies = bool(
        np.isfinite(signal_standard_deviation)
        and signal_standard_deviation >= MIN_SIGNAL_STANDARD_DEVIATION
    )
    emg_spike_fraction = _robust_spike_fraction(signal)
    emg_spikes_limited = bool(emg_spike_fraction <= MAX_EMG_SPIKE_FRACTION)

    required_seconds = BASELINE_SECONDS + POST_BASELINE_SECONDS
    required_samples = int(round(EXPECTED_SEMG_RATE_HZ * required_seconds))
    duration_ready = len(latest_session) >= required_samples

    baseline_samples = int(round(EXPECTED_SEMG_RATE_HZ * BASELINE_SECONDS))
    baseline = signal[:baseline_samples]
    baseline_standard_deviation = (
        float(np.std(baseline)) if len(baseline) == baseline_samples else None
    )
    baseline_valid = bool(
        baseline_standard_deviation is not None
        and np.isfinite(baseline_standard_deviation)
        and baseline_standard_deviation >= MIN_SIGNAL_STANDARD_DEVIATION
    )

    checks = [
        _check(
            "Finite sensor values",
            finite,
            "All required values are finite" if finite else "Missing or non-finite values found",
        ),
        _check(
            "Sequence continuity",
            sequence_contiguous,
            "No missing sequence steps"
            if sequence_contiguous
            else (
                f"Missing {missing_sequence_count}; duplicates "
                f"{duplicate_sequence_count}; backward steps "
                f"{backward_sequence_count}"
            ),
        ),
        _check(
            "500 Hz sEMG cadence",
            rate_valid,
            (
                f"{observed_rate:.1f} Hz"
                if observed_rate is not None
                else "Unable to calculate"
            ),
        ),
        _check(
            "Stable device cadence",
            cadence_stable,
            f"{interval_outlier_fraction:.2%} timing outliers",
        ),
        _check(
            "Quaternion validity",
            quaternion_valid,
            (
                f"Maximum norm error {maximum_quaternion_error:.4f}"
                if maximum_quaternion_error is not None
                else "Unable to calculate"
            ),
        ),
        _check(
            "Non-flat sEMG",
            signal_varies,
            f"Signal standard deviation {signal_standard_deviation:.4f}",
        ),
        _check(
            "Limited abrupt sEMG spikes",
            emg_spikes_limited,
            f"{emg_spike_fraction:.2%} abrupt differences",
        ),
        _check(
            "Ten-second rest baseline",
            baseline_valid,
            (
                f"{len(baseline)} samples; standard deviation "
                f"{baseline_standard_deviation:.4f}"
                if baseline_standard_deviation is not None
                else f"{len(baseline)} of {baseline_samples} samples"
            ),
        ),
        _check(
            "Complete analysis duration",
            duration_ready,
            f"{len(latest_session)} of {required_samples} required samples",
        ),
    ]

    blocking_issues = [
        item["check"] for item in checks if not item["passed"]
    ]
    warnings = []
    if "Calib" in latest_session.columns:
        calibration = pd.to_numeric(
            latest_session["Calib"], errors="coerce"
        ).dropna()
        low_calibration_fraction = (
            float((calibration < 2).mean()) if not calibration.empty else 1.0
        )
        if low_calibration_fraction > 0:
            warnings.append(
                "IMU calibration was below level 2 for "
                f"{low_calibration_fraction:.1%} of samples."
            )
    else:
        low_calibration_fraction = None
        warnings.append("IMU calibration metadata was not supplied.")

    if blocking_issues:
        quality_status = "Invalid"
        quality_level = "red"
    elif warnings:
        quality_status = "Needs review"
        quality_level = "yellow"
    else:
        quality_status = "Good"
        quality_level = "green"

    return {
        "ready": not blocking_issues,
        "checks": checks,
        "signal_quality": {
            "status": quality_status,
            "level": quality_level,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "metrics": {
                "missing_sequence_count": missing_sequence_count,
                "duplicate_sequence_count": duplicate_sequence_count,
                "backward_sequence_count": backward_sequence_count,
                "timing_outlier_fraction": interval_outlier_fraction,
                "emg_spike_fraction": emg_spike_fraction,
                "maximum_quaternion_norm_error": maximum_quaternion_error,
                "low_imu_calibration_fraction": low_calibration_fraction,
            },
        },
        "latest_session": int(latest_session["Session"].iloc[-1]),
        "row_count": int(len(latest_session)),
        "observed_semg_rate_hz": observed_rate,
        "expected_semg_rate_hz": EXPECTED_SEMG_RATE_HZ,
        "expected_imu_rate_hz": EXPECTED_IMU_RATE_HZ,
        "baseline_samples": baseline_samples,
        "required_samples": required_samples,
        "required_seconds": required_seconds,
    }



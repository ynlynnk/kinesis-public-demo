"""Deterministic in-memory sensor stream for exercising Model 1.

The stream follows the proposed device timing contract: sEMG is generated at
500 Hz, filtered with the firmware EMA, baseline-normalised, and resampled to
the trained CNN rate of 200 Hz. Quaternion IMU samples are generated at 50 Hz.
The forearm checkpoints use sEMG, while the quadriceps movement checkpoint
uses the matching sEMG and IMU windows.
"""

from __future__ import annotations

import numpy as np

from model1_signal_pipeline import (
    DEVICE_IMU_RATE_HZ,
    DEVICE_SEMG_RATE_HZ,
    EMA_ALPHA,
    MODEL_SEMG_RATE_HZ,
    ema_filter,
    normalize_to_baseline,
    resample_signal,
)


SAMPLE_RATE_HZ = MODEL_SEMG_RATE_HZ
BUFFER_SAMPLES = 512
DEVICE_BUFFER_SAMPLES = int(
    round(BUFFER_SAMPLES * DEVICE_SEMG_RATE_HZ / MODEL_SEMG_RATE_HZ)
)
IMU_BUFFER_SAMPLES = int(
    round(BUFFER_SAMPLES * DEVICE_IMU_RATE_HZ / MODEL_SEMG_RATE_HZ)
)
STEP_SAMPLES = DEVICE_SEMG_RATE_HZ
PATTERNS = (
    "Low activity",
    "Rhythmic contraction",
    "Sustained contraction",
)


def generate_synthetic_semg(
    start_sample: int,
    sample_count: int,
    pattern: str,
    sample_rate_hz: int = DEVICE_SEMG_RATE_HZ,
) -> np.ndarray:
    """Generate a repeatable raw sEMG-like chunk at the device rate."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if start_sample < 0:
        raise ValueError("start_sample cannot be negative")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown signal pattern: {pattern}")

    sample_numbers = start_sample + np.arange(sample_count, dtype=np.float64)
    time_seconds = sample_numbers / sample_rate_hz
    carrier = (
        0.62 * np.sin(2 * np.pi * 37.0 * time_seconds + 0.2)
        + 0.41 * np.sin(2 * np.pi * 61.0 * time_seconds + 1.1)
        + 0.24 * np.sin(2 * np.pi * 83.0 * time_seconds + 2.0)
        + 0.18 * np.sin(2 * np.pi * 17.3 * time_seconds + 0.7)
    )

    if pattern == "Low activity":
        envelope = 0.35 + 0.05 * np.sin(2 * np.pi * 0.35 * time_seconds)
    elif pattern == "Rhythmic contraction":
        envelope = 0.45 + 0.85 * (
            0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * time_seconds - 0.8)
        )
    else:
        elapsed_in_cycle = np.mod(time_seconds, 60.0)
        envelope = 0.75 + 0.018 * elapsed_in_cycle

    drift = 0.08 * np.sin(2 * np.pi * 0.12 * time_seconds)
    return (envelope * carrier + drift).astype(np.float32)


def generate_synthetic_imu(
    start_sample: int,
    sample_count: int,
    sample_rate_hz: int = DEVICE_IMU_RATE_HZ,
) -> np.ndarray:
    """Generate unit quaternions describing a gentle joint rotation."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if start_sample < 0:
        raise ValueError("start_sample cannot be negative")
    sample_numbers = start_sample + np.arange(sample_count, dtype=np.float64)
    time_seconds = sample_numbers / sample_rate_hz
    angle_degrees = 18.0 * np.sin(2 * np.pi * 0.22 * time_seconds)
    angle_radians = np.deg2rad(angle_degrees)
    quaternions = np.zeros((sample_count, 4), dtype=np.float32)
    quaternions[:, 0] = np.cos(angle_radians / 2.0)
    quaternions[:, 1] = np.sin(angle_radians / 2.0)
    return quaternions


def _calibration_statistics() -> tuple[float, float]:
    calibration = generate_synthetic_semg(
        0,
        10 * DEVICE_SEMG_RATE_HZ,
        "Low activity",
    )
    filtered, _ = ema_filter(calibration, EMA_ALPHA)
    return float(filtered.mean()), max(float(filtered.std()), 1e-3)


def _model_signal(
    filtered_signal: np.ndarray,
    baseline_mean: float,
    baseline_standard_deviation: float,
) -> np.ndarray:
    normalized = normalize_to_baseline(
        filtered_signal,
        baseline_mean,
        baseline_standard_deviation,
    )
    resampled = resample_signal(
        normalized,
        DEVICE_SEMG_RATE_HZ,
        MODEL_SEMG_RATE_HZ,
    )
    return resampled[-BUFFER_SAMPLES:].astype(np.float32)


def create_demo_buffer(pattern: str) -> dict:
    """Create complete sEMG and IMU buffers so both CNN heads can execute."""

    baseline_mean, baseline_standard_deviation = _calibration_statistics()
    raw_semg = generate_synthetic_semg(0, DEVICE_BUFFER_SAMPLES, pattern)
    filtered_semg, ema_last = ema_filter(raw_semg, EMA_ALPHA)
    imu_quaternion = generate_synthetic_imu(0, IMU_BUFFER_SAMPLES)
    return {
        "pattern": pattern,
        "raw_semg": raw_semg,
        "filtered_semg": filtered_semg,
        "signal": _model_signal(
            filtered_semg,
            baseline_mean,
            baseline_standard_deviation,
        ),
        "imu_quaternion": imu_quaternion,
        "next_device_sample": DEVICE_BUFFER_SAMPLES,
        "next_imu_sample": IMU_BUFFER_SAMPLES,
        "next_sample": BUFFER_SAMPLES,
        "elapsed_seconds": 0.0,
        "ema_last": ema_last,
        "baseline_mean": baseline_mean,
        "baseline_standard_deviation": baseline_standard_deviation,
    }


def advance_demo_buffer(state: dict, sample_count: int = STEP_SAMPLES) -> dict:
    """Advance the stream by device samples while retaining CNN-ready data."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    pattern = str(state["pattern"])
    start_device_sample = int(state["next_device_sample"])
    new_raw = generate_synthetic_semg(
        start_device_sample,
        sample_count,
        pattern,
    )
    new_filtered, ema_last = ema_filter(
        new_raw,
        EMA_ALPHA,
        initial_value=float(state["ema_last"]),
    )
    raw_semg = np.concatenate([state["raw_semg"], new_raw])[
        -DEVICE_BUFFER_SAMPLES:
    ].astype(np.float32)
    filtered_semg = np.concatenate([
        state["filtered_semg"],
        new_filtered,
    ])[-DEVICE_BUFFER_SAMPLES:].astype(np.float32)

    duration_seconds = sample_count / DEVICE_SEMG_RATE_HZ
    imu_sample_count = int(round(duration_seconds * DEVICE_IMU_RATE_HZ))
    start_imu_sample = int(state["next_imu_sample"])
    new_imu = generate_synthetic_imu(start_imu_sample, imu_sample_count)
    imu_quaternion = np.concatenate([
        state["imu_quaternion"],
        new_imu,
    ])[-IMU_BUFFER_SAMPLES:].astype(np.float32)

    next_device_sample = start_device_sample + sample_count
    next_imu_sample = start_imu_sample + imu_sample_count
    return {
        **state,
        "raw_semg": raw_semg,
        "filtered_semg": filtered_semg,
        "signal": _model_signal(
            filtered_semg,
            float(state["baseline_mean"]),
            float(state["baseline_standard_deviation"]),
        ),
        "imu_quaternion": imu_quaternion,
        "next_device_sample": next_device_sample,
        "next_imu_sample": next_imu_sample,
        "next_sample": int(
            round(next_device_sample * MODEL_SEMG_RATE_HZ / DEVICE_SEMG_RATE_HZ)
        ),
        "elapsed_seconds": float(state["elapsed_seconds"]) + duration_seconds,
        "ema_last": ema_last,
    }

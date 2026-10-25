"""Signal preparation contract between the KINESIS device and Model 1.

The current trained CNN checkpoints consume one baseline-normalised sEMG
channel at 200 Hz.  The proposed device acquires sEMG more densely and applies
an exponential moving average (EMA) on the ESP32.  This module keeps those two
rates explicit so a 500 Hz device stream is never mistaken for a 200 Hz model
window.
"""

from __future__ import annotations

import numpy as np


DEVICE_SEMG_RATE_HZ = 500
DEVICE_IMU_RATE_HZ = 50
MODEL_SEMG_RATE_HZ = 200
EMA_ALPHA = 0.25
UPLOAD_INTERVAL_SECONDS = 2
MAX_SESSION_SECONDS = 20 * 60


def ema_filter(
    values,
    alpha: float = EMA_ALPHA,
    initial_value: float | None = None,
) -> tuple[np.ndarray, float]:
    """Apply the firmware EMA and return the filtered signal and final state."""

    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(signal) == 0:
        raise ValueError("The sEMG signal cannot be empty.")
    if not np.isfinite(signal).all():
        raise ValueError("The sEMG signal contains a non-finite value.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be greater than 0 and at most 1.")

    filtered = np.empty_like(signal)
    previous = float(signal[0] if initial_value is None else initial_value)
    for index, raw_value in enumerate(signal):
        previous = alpha * float(raw_value) + (1.0 - alpha) * previous
        filtered[index] = previous
    return filtered, previous


def resample_signal(
    values,
    source_rate_hz: int,
    target_rate_hz: int = MODEL_SEMG_RATE_HZ,
) -> np.ndarray:
    """Linearly resample a continuous signal while preserving its duration."""

    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(signal) == 0:
        raise ValueError("The signal cannot be empty.")
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("Sampling rates must be positive.")

    target_count = max(
        1,
        int(round(len(signal) * target_rate_hz / source_rate_hz)),
    )
    if target_count == len(signal):
        return signal.copy()

    source_time = np.arange(len(signal), dtype=np.float64) / source_rate_hz
    target_time = np.arange(target_count, dtype=np.float64) / target_rate_hz
    target_time = np.minimum(target_time, source_time[-1])
    return np.interp(target_time, source_time, signal).astype(np.float32)


def normalize_to_baseline(
    values,
    baseline_mean: float,
    baseline_standard_deviation: float,
) -> np.ndarray:
    """Apply the same neutral-baseline scaling used during CNN training."""

    if not np.isfinite(baseline_mean):
        raise ValueError("Baseline mean must be finite.")
    if not np.isfinite(baseline_standard_deviation):
        raise ValueError("Baseline standard deviation must be finite.")
    scale = max(float(baseline_standard_deviation), 1e-3)
    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    normalized = (signal - float(baseline_mean)) / scale
    return np.clip(normalized, -12.0, 12.0).astype(np.float32)


def prepare_semg_for_model(
    raw_values,
    baseline_mean: float,
    baseline_standard_deviation: float,
    source_rate_hz: int = DEVICE_SEMG_RATE_HZ,
) -> np.ndarray:
    """Filter, baseline-normalise, and resample device sEMG for Model 1."""

    filtered, _ = ema_filter(raw_values)
    normalized = normalize_to_baseline(
        filtered,
        baseline_mean,
        baseline_standard_deviation,
    )
    return resample_signal(normalized, source_rate_hz, MODEL_SEMG_RATE_HZ)

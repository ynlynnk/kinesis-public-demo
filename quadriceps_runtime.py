"""Runtime helpers for the real-data quadriceps CNN checkpoints."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


MOVEMENT_CLASS_NAMES = (
    "Correct extension",
    "Incomplete extension",
    "Compensatory leg lift",
)
FATIGUE_CLASS_NAMES = (
    "Low-fatigue phase",
    "Fatigue-like phase",
)


class QuadricepsMovementCNN(nn.Module):
    """Dual-branch CNN for rectus-femoris sEMG and thigh motion."""

    def __init__(self, class_count: int = len(MOVEMENT_CLASS_NAMES)):
        super().__init__()
        self.semg_features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.imu_features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, class_count),
        )

    def forward(self, semg_window, imu_window):
        envelope = functional.avg_pool1d(
            semg_window.abs(),
            kernel_size=21,
            stride=1,
            padding=10,
        )
        semg_features = self.semg_features(
            torch.cat([semg_window, envelope], dim=1)
        ).flatten(1)
        imu_features = self.imu_features(imu_window).flatten(1)
        return self.classifier(torch.cat([semg_features, imu_features], dim=1))


class QuadricepsFatigueCNN(nn.Module):
    """Small 1D CNN for a single quadriceps sEMG channel."""

    def __init__(self, class_count: int = len(FATIGUE_CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, class_count),
        )

    def forward(self, semg_window):
        envelope = functional.avg_pool1d(
            semg_window.abs(),
            kernel_size=31,
            stride=1,
            padding=15,
        )
        return self.classifier(
            self.features(torch.cat([semg_window, envelope], dim=1))
        )


def _load_checkpoint(model_path: str, expected_classes: tuple[str, ...]) -> dict:
    checkpoint = torch.load(
        Path(model_path),
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("training_data_kind") != "real_public_semg":
        raise ValueError("This is not a real public-sEMG checkpoint.")
    if checkpoint.get("trained_on_labelled_data") is not True:
        raise ValueError("The checkpoint is not marked as trained.")
    if checkpoint.get("muscle_group") != "Quadriceps":
        raise ValueError("The checkpoint is not a quadriceps model.")
    if tuple(checkpoint.get("class_names", ())) != expected_classes:
        raise ValueError("The checkpoint has unexpected classes.")
    return checkpoint


@lru_cache(maxsize=2)
def load_quadriceps_movement_model(model_path: str):
    checkpoint = _load_checkpoint(model_path, MOVEMENT_CLASS_NAMES)
    model = QuadricepsMovementCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@lru_cache(maxsize=2)
def load_quadriceps_fatigue_model(model_path: str):
    checkpoint = _load_checkpoint(model_path, FATIGUE_CLASS_NAMES)
    model = QuadricepsFatigueCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _validated_window(values, expected_samples: int) -> np.ndarray:
    window = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(window) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} sEMG samples, received {len(window)}."
        )
    if not np.isfinite(window).all():
        raise ValueError("The sEMG window contains a non-finite value.")
    return window


def _probabilities(model, checkpoint: dict, window: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(window[None, None, :])
    temperature = max(float(checkpoint.get("temperature", 1.0)), 1e-3)
    with torch.no_grad():
        logits = model(tensor) / temperature
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


def _quaternion_angular_speed(quaternions: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float32)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("IMU input must contain four quaternion channels.")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / np.maximum(norms, 1e-6)
    dots = np.sum(quaternions[1:] * quaternions[:-1], axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    angular_speed = 2.0 * np.arccos(dots) * float(sample_rate_hz)
    return np.concatenate(([angular_speed[0]], angular_speed)).astype(np.float32)


def predict_quadriceps_movement_window(
    values,
    imu_quaternions,
    model_path: str,
) -> dict:
    model, checkpoint = load_quadriceps_movement_model(model_path)
    window = _validated_window(values, int(checkpoint["window_samples"]))
    if checkpoint.get("runtime_window_standardization", False):
        window = (window - window.mean()) / max(float(window.std()), 1e-3)
        window = np.clip(window, -12.0, 12.0)
    imu_expected = int(checkpoint["imu_window_samples"])
    quaternions = np.asarray(imu_quaternions, dtype=np.float32)
    if len(quaternions) != imu_expected:
        raise ValueError(
            f"Expected {imu_expected} quaternion samples, received {len(quaternions)}."
        )
    if not np.isfinite(quaternions).all():
        raise ValueError("The quaternion window contains a non-finite value.")
    imu_window = _quaternion_angular_speed(
        quaternions,
        int(checkpoint["imu_sample_rate_hz"]),
    )
    imu_window = (imu_window - imu_window.mean()) / max(float(imu_window.std()), 1e-3)
    imu_window = np.clip(imu_window, -12.0, 12.0)
    temperature = max(float(checkpoint.get("temperature", 1.0)), 1e-3)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(window[None, None, :]),
            torch.from_numpy(imu_window[None, None, :]),
        ) / temperature
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_index = int(np.argmax(probabilities))
    return {
        "label": MOVEMENT_CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(MOVEMENT_CLASS_NAMES)
        },
        "model_version": checkpoint["model_version"],
    }


def predict_quadriceps_fatigue_window(values, model_path: str) -> dict:
    model, checkpoint = load_quadriceps_fatigue_model(model_path)
    window = _validated_window(values, int(checkpoint["window_samples"]))
    probabilities = _probabilities(model, checkpoint, window)
    predicted_index = int(np.argmax(probabilities))
    return {
        "label": FATIGUE_CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "confidence": float(probabilities[predicted_index]),
        "fatigue_score": float(probabilities[1]),
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(FATIGUE_CLASS_NAMES)
        },
        "model_version": checkpoint["model_version"],
    }

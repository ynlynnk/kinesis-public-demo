"""Runtime for the synthetic-trained Nexus-format Model 1 preview."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn


QUATERNION_COLUMNS = ["Quat W", "Quat X", "Quat Y", "Quat Z"]
INPUT_Z_SCORE_LIMIT = 3.0


@lru_cache(maxsize=8)
def get_nexus_checkpoint_metadata(model_path: str) -> dict:
    """Read checkpoint provenance without executing the CNN."""

    checkpoint = torch.load(
        Path(model_path),
        map_location="cpu",
        weights_only=False,
    )
    return {
        "model_version": checkpoint.get("model_version", "unknown"),
        "training_data_kind": checkpoint.get(
            "training_data_kind",
            "unknown",
        ),
        "trained_on_labelled_data": bool(
            checkpoint.get("trained_on_labelled_data", False)
        ),
        "validated_on_real_data": bool(
            checkpoint.get("validated_on_real_data", False)
        ),
        "muscle_group": checkpoint.get("muscle_group"),
        "exercise": checkpoint.get("exercise"),
    }


class NexusSensorBranch(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, sensor_data):
        return self.features(sensor_data).flatten(start_dim=1)


class NexusKinesisCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.semg_branch = NexusSensorBranch(1)
        self.orientation_branch = NexusSensorBranch(4)
        self.fusion = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fatigue_head = nn.Linear(32, 2)
        self.movement_head = nn.Linear(32, 2)

    def forward(self, semg_data, quaternion_data):
        features = torch.cat(
            [
                self.semg_branch(semg_data),
                self.orientation_branch(quaternion_data),
            ],
            dim=1,
        )
        fused = self.fusion(features)
        return self.fatigue_head(fused), self.movement_head(fused)


@lru_cache(maxsize=2)
def load_nexus_model(model_path: str):
    path = Path(model_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("trained_on_labelled_data") is not True:
        raise ValueError("The Nexus checkpoint is not marked as trained.")
    expected = {
        "semg": ["sEMG raw"],
        "quaternion": QUATERNION_COLUMNS,
    }
    if checkpoint.get("input_columns") != expected:
        raise ValueError("The Nexus checkpoint has a different input contract.")
    model = NexusKinesisCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_latest_nexus_window(sensor_data, model_path: str) -> dict:
    """Predict the latest complete window without crossing a recording reset."""

    torch.set_num_threads(1)
    model, checkpoint = load_nexus_model(model_path)
    window_rows = int(checkpoint["window_rows"])
    latest_session = sensor_data["Session"].iloc[-1]
    session = sensor_data[sensor_data["Session"] == latest_session]
    session = session.sort_values(["Seq", "STM32 ms"], kind="stable")
    if len(session) < window_rows:
        raise ValueError(
            f"The latest recording needs {window_rows} contiguous rows; "
            f"only {len(session)} are available."
        )
    selected = session.iloc[-window_rows:]
    if not selected["Seq"].diff().dropna().eq(1).all():
        raise ValueError("The latest Nexus window has a sequence gap.")

    semg = selected[["sEMG raw"]].to_numpy(dtype=np.float32).T[None, :, :]
    quaternion = selected[QUATERNION_COLUMNS].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("The latest Nexus window has an invalid quaternion.")
    quaternion = (quaternion / norms).T[None, :, :]
    semg = (
        semg - checkpoint["semg_mean"].cpu().numpy()
    ) / checkpoint["semg_standard_deviation"].cpu().numpy()
    quaternion = (
        quaternion - checkpoint["quaternion_mean"].cpu().numpy()
    ) / checkpoint["quaternion_standard_deviation"].cpu().numpy()

    semg_max_absolute_z_score = float(np.max(np.abs(semg)))
    quaternion_max_absolute_z_score = float(np.max(np.abs(quaternion)))
    input_max_absolute_z_score = max(
        semg_max_absolute_z_score,
        quaternion_max_absolute_z_score,
    )
    input_within_training_range = bool(
        input_max_absolute_z_score <= INPUT_Z_SCORE_LIMIT
    )
    validated_on_real_data = bool(
        checkpoint.get("validated_on_real_data", False)
    )

    with torch.no_grad():
        fatigue_output, movement_output = model(
            torch.from_numpy(semg),
            torch.from_numpy(quaternion),
        )
        fatigue_probability = float(
            torch.softmax(fatigue_output, dim=1)[0, 1]
        )
        movement_incorrect_probability = float(
            torch.softmax(movement_output, dim=1)[0, 1]
        )

    output_saturated = bool(
        fatigue_probability <= 0.005
        or fatigue_probability >= 0.995
        or movement_incorrect_probability <= 0.005
        or movement_incorrect_probability >= 0.995
    )
    prediction_released = bool(
        validated_on_real_data and input_within_training_range
    )

    return {
        "model_version": checkpoint["model_version"],
        "training_data_kind": checkpoint.get("training_data_kind", "unknown"),
        "validated_on_real_data": validated_on_real_data,
        "window_rows": window_rows,
        "window_end_sequence": int(selected["Seq"].iloc[-1]),
        "window_end_timestamp": selected["Timestamp"].iloc[-1],
        "fatigue_probability": fatigue_probability,
        "fatigue_label": (
            "Fatigued" if fatigue_probability >= 0.5 else "Non-fatigued"
        ),
        "movement_incorrect_probability": movement_incorrect_probability,
        "movement_label": (
            "Incorrect"
            if movement_incorrect_probability >= 0.5
            else "Correct"
        ),
        "semg_max_absolute_z_score": semg_max_absolute_z_score,
        "quaternion_max_absolute_z_score": (
            quaternion_max_absolute_z_score
        ),
        "input_max_absolute_z_score": input_max_absolute_z_score,
        "input_z_score_limit": INPUT_Z_SCORE_LIMIT,
        "input_within_training_range": input_within_training_range,
        "output_saturated": output_saturated,
        "prediction_released": prediction_released,
    }

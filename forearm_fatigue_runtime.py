"""Runtime helpers for the real-data upper-limb fatigue-phase CNN."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


CLASS_NAMES = ("Low-fatigue phase", "Fatigue-like phase")


class ForearmFatigueCNN(nn.Module):
    """Small 1D CNN for one baseline-normalised sEMG channel."""

    def __init__(self, class_count: int = len(CLASS_NAMES)):
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
        features = torch.cat([semg_window, envelope], dim=1)
        return self.classifier(self.features(features))


@lru_cache(maxsize=2)
def load_forearm_fatigue_model(model_path: str):
    """Load the labelled-real-data fatigue model and model card."""

    checkpoint = torch.load(
        Path(model_path),
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("training_data_kind") != "real_public_semg":
        raise ValueError("This is not a real public-sEMG checkpoint.")
    if checkpoint.get("trained_on_labelled_data") is not True:
        raise ValueError("The fatigue checkpoint is not marked as trained.")
    if tuple(checkpoint.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("The fatigue checkpoint has unexpected classes.")

    model = ForearmFatigueCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_normalized_fatigue_window(
    normalized_window,
    model_path: str,
) -> dict:
    """Classify one already baseline-normalised sEMG window."""

    model, checkpoint = load_forearm_fatigue_model(model_path)
    window = np.asarray(normalized_window, dtype=np.float32).reshape(-1)
    expected_samples = int(checkpoint["window_samples"])
    if len(window) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} sEMG samples, received {len(window)}."
        )
    if not np.isfinite(window).all():
        raise ValueError("The sEMG window contains a non-finite value.")

    tensor = torch.from_numpy(window[None, None, :])
    temperature = max(float(checkpoint.get("temperature", 1.0)), 1e-3)
    with torch.no_grad():
        logits = model(tensor) / temperature
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    return {
        "label": CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "confidence": float(probabilities[predicted_index]),
        "fatigue_score": float(probabilities[1]),
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "model_version": checkpoint["model_version"],
    }


@lru_cache(maxsize=2)
def load_fatigue_verification_samples(sample_path: str) -> dict:
    """Load small held-out real-data examples bundled with the dashboard."""

    with np.load(Path(sample_path), allow_pickle=False) as bundle:
        return {
            "windows": bundle["windows"].astype(np.float32),
            "labels": bundle["labels"].astype(np.int64),
            "subjects": bundle["subjects"].astype(np.int64),
            "channels": bundle["channels"].astype(np.int64),
            "start_seconds": bundle["start_seconds"].astype(np.float32),
        }

"""Evaluate repeated-window decisions for the two deployed forearm CNNs.

The base checkpoints and held-out participants are unchanged. This script
measures the decision unit used by a rehabilitation session: several adjacent
window probabilities are averaged before assigning one label. It reports both
the original single-window result and the repeated-window result so the two
conditions cannot be confused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from forearm_fatigue_runtime import ForearmFatigueCNN  # noqa: E402
from forearm_movement_runtime import (  # noqa: E402
    CLASS_NAMES as MOVEMENT_CLASSES,
    ForearmMovementCNN,
)
from training.forearm import train_real_fatigue_cnn as fatigue_training  # noqa: E402
from training.forearm import train_real_forearm_cnn as movement_training  # noqa: E402


OUTPUT_FILE = (
    PUBLIC_ROOT / "outputs" / "evaluation" / "forearm_session_metrics.json"
)


def multiclass_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: tuple[str, ...],
) -> dict:
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names))),
    )
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": {
            name: float(recalls[index])
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "decision_count": int(len(targets)),
    }


def movement_session_metrics() -> dict:
    checkpoint_path = (
        PUBLIC_ROOT / "artifacts" / "models" / "kinesis_forearm_movement_cnn.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = ForearmMovementCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    frame = movement_training.load_source_data(
        PUBLIC_ROOT
        / "data"
        / "raw"
        / "forearm"
        / "mendeley_p77jn92bzg_v1"
        / "Multi-Channel Electromyography Signal Acquisition of Forearm"
        / "signal_dataset.dat"
    )
    windows, labels, subjects, channels = movement_training.build_examples(
        frame,
        checkpoint["test_metrics"]["test_subjects"],
        movement_training.EVALUATION_JITTER_SECONDS,
    )
    with torch.no_grad():
        logits = model(torch.from_numpy(windows[:, None, :]))
        probabilities = torch.softmax(
            logits / float(checkpoint.get("temperature", 1.0)),
            dim=1,
        ).numpy()

    targets, predictions = [], []
    for subject in np.unique(subjects):
        for channel in np.unique(channels):
            for class_index in range(len(MOVEMENT_CLASSES)):
                indices = np.flatnonzero(
                    (subjects == subject)
                    & (channels == channel)
                    & (labels == class_index)
                )
                targets.append(class_index)
                predictions.append(int(probabilities[indices].mean(axis=0).argmax()))
    result = multiclass_metrics(
        np.asarray(targets),
        np.asarray(predictions),
        MOVEMENT_CLASSES,
    )
    result.update(
        {
            "evaluation_kind": "held-out-participant repeated-window session decision",
            "windows_per_decision": 10,
            "held_out_subjects": checkpoint["test_metrics"]["test_subjects"],
            "single_window_reference": checkpoint["test_metrics"],
        }
    )
    return result


def fatigue_session_metrics() -> dict:
    checkpoint_path = (
        PUBLIC_ROOT / "artifacts" / "models" / "kinesis_forearm_fatigue_cnn.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = ForearmFatigueCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    windows, labels, subjects, channels, start_seconds = fatigue_training.build_examples(
        PUBLIC_ROOT
        / "data"
        / "raw"
        / "forearm_fatigue"
        / "zenodo_5189275"
        / "extracted"
        / "Dataset EMG Fatigue"
        / "Data as txt Files",
        checkpoint["test_metrics"]["test_subjects"],
    )
    with torch.no_grad():
        logits = model(torch.from_numpy(windows[:, None, :]))
        probabilities = torch.softmax(
            logits / float(checkpoint.get("temperature", 1.0)),
            dim=1,
        ).numpy()

    block_size = 9
    targets, predictions, fatigue_scores = [], [], []
    for subject in np.unique(subjects):
        for channel in np.unique(channels):
            for class_index in range(2):
                indices = np.flatnonzero(
                    (subjects == subject)
                    & (channels == channel)
                    & (labels == class_index)
                )
                indices = indices[np.argsort(start_seconds[indices])]
                for first in range(0, len(indices) - block_size + 1, block_size):
                    block = probabilities[indices[first : first + block_size]].mean(
                        axis=0
                    )
                    targets.append(class_index)
                    predictions.append(int(block.argmax()))
                    fatigue_scores.append(float(block[1]))
    class_names = ("Low-fatigue phase", "Fatigue-like phase")
    result = multiclass_metrics(
        np.asarray(targets),
        np.asarray(predictions),
        class_names,
    )
    result.update(
        {
            "roc_auc": float(roc_auc_score(targets, fatigue_scores)),
            "evaluation_kind": "held-out-participant sustained-window decision",
            "windows_per_decision": block_size,
            "evidence_span_seconds": (
                fatigue_training.WINDOW_SAMPLES
                + (block_size - 1) * fatigue_training.WINDOW_STRIDE
            )
            / fatigue_training.SAMPLE_RATE_HZ,
            "held_out_subjects": checkpoint["test_metrics"]["test_subjects"],
            "single_window_reference": checkpoint["test_metrics"],
        }
    )
    return result


def main() -> None:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    report = {
        "forearm_movement": movement_session_metrics(),
        "forearm_fatigue": fatigue_session_metrics(),
        "reporting_rule": (
            "Repeated-window metrics are operational session decisions and must "
            "not be relabelled as single-window checkpoint accuracy."
        ),
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

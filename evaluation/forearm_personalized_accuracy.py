"""Evaluate optional guided calibration for the forearm movement CNN.

Each held-out participant/electrode combination is treated as one possible
single-electrode device placement. Three labelled examples per class are used
to adapt only the classifier head. Metrics are calculated exclusively on the
remaining examples and must be labelled "after supervised calibration".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_ROOT))

from forearm_movement_runtime import CLASS_NAMES, ForearmMovementCNN  # noqa: E402
from training.forearm import train_real_forearm_cnn as training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PUBLIC_ROOT
            / "artifacts"
            / "models"
            / "kinesis_forearm_movement_cnn.pt"
        ),
    )
    parser.add_argument("--calibration-examples-per-class", type=int, default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=(
            PUBLIC_ROOT
            / "outputs"
            / "evaluation"
            / "forearm_movement_personalized_metrics.json"
        ),
    )
    return parser.parse_args()


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = confusion_matrix(targets, predictions, labels=[0, 1, 2])
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": {
            name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(targets)),
    }


def adapt_classifier(
    base_model: ForearmMovementCNN,
    windows: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
) -> ForearmMovementCNN:
    model = deepcopy(base_model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    optimiser = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=6e-4,
        weight_decay=5e-3,
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.04)
    model.train()
    for _ in range(steps):
        optimiser.zero_grad()
        augmented = windows * (0.94 + 0.12 * torch.rand(len(windows), 1, 1))
        augmented = augmented + 0.01 * torch.randn_like(augmented)
        loss = loss_function(model(augmented), labels)
        loss.backward()
        optimiser.step()
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    random.seed(training.SEED)
    np.random.seed(training.SEED)
    torch.manual_seed(training.SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model = ForearmMovementCNN()
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()
    test_subjects = checkpoint["test_metrics"]["test_subjects"]
    frame = training.load_source_data(
        PUBLIC_ROOT
        / "data"
        / "raw"
        / "forearm"
        / "mendeley_p77jn92bzg_v1"
        / "Multi-Channel Electromyography Signal Acquisition of Forearm"
        / "signal_dataset.dat"
    )
    windows, labels, subjects, channels = training.build_examples(
        frame,
        test_subjects,
        training.EVALUATION_JITTER_SECONDS,
    )

    all_targets, all_predictions, placement_results = [], [], []
    selection_rng = np.random.default_rng(training.SEED)
    for subject in test_subjects:
        for channel in range(1, 9):
            placement_indices = np.flatnonzero(
                (subjects == subject) & (channels == channel)
            )
            calibration_indices, evaluation_indices = [], []
            for class_index in range(len(CLASS_NAMES)):
                class_indices = placement_indices[
                    labels[placement_indices] == class_index
                ].copy()
                selection_rng.shuffle(class_indices)
                count = min(
                    args.calibration_examples_per_class,
                    len(class_indices) - 1,
                )
                calibration_indices.extend(class_indices[:count].tolist())
                evaluation_indices.extend(class_indices[count:].tolist())
            calibration_indices = np.asarray(calibration_indices, dtype=np.int64)
            evaluation_indices = np.asarray(evaluation_indices, dtype=np.int64)

            calibration_windows = torch.from_numpy(
                windows[calibration_indices, None, :]
            )
            calibration_labels = torch.from_numpy(labels[calibration_indices])
            model = adapt_classifier(
                base_model,
                calibration_windows,
                calibration_labels,
                args.steps,
            )
            evaluation_windows = torch.from_numpy(
                windows[evaluation_indices, None, :]
            )
            with torch.no_grad():
                predictions = model(evaluation_windows).argmax(dim=1).numpy()
            targets = labels[evaluation_indices]
            result = metrics(targets, predictions)
            result.update(
                {
                    "subject": int(subject),
                    "channel": channel,
                    "calibration_sample_count": int(len(calibration_indices)),
                }
            )
            placement_results.append(result)
            all_targets.extend(targets.tolist())
            all_predictions.extend(predictions.tolist())

    aggregate = metrics(
        np.asarray(all_targets, dtype=np.int64),
        np.asarray(all_predictions, dtype=np.int64),
    )
    aggregate.update(
        {
            "evaluation_kind": (
                "held-out-participant-and-electrode-after-supervised-calibration"
            ),
            "calibration_examples_per_class": args.calibration_examples_per_class,
            "held_out_subject_count": len(test_subjects),
            "evaluated_device_placements": len(placement_results),
            "test_subjects": test_subjects,
            "placement_results": placement_results,
            "zero_calibration_reference": checkpoint["test_metrics"],
        }
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

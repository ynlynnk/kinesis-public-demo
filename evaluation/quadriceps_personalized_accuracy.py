"""Evaluate optional patient calibration for the quadriceps movement CNN.

The base model remains participant-separated.  For each held-out participant,
three labelled repetitions per class are reserved as a supervised fitting-set
calibration; accuracy is calculated only on that participant's remaining
repetitions.  This metric must be labelled "after calibration" and must not be
reported as zero-calibration held-out accuracy.
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
from torch import nn


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIRECTORY = PUBLIC_ROOT / "training" / "quadriceps"
if str(TRAINING_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIRECTORY))

from train_quadriceps_movement_repetition_cnn import (  # noqa: E402
    CLASS_NAMES,
    QuadricepsMovementRepetitionCNN,
    SEED,
    build_examples,
    classification_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=(
            PUBLIC_ROOT / "data" / "raw" / "quadriceps_movement"
            / "kneepad" / "dataset.zip"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PUBLIC_ROOT / "artifacts" / "models" / "candidates"
            / "kinesis_quadriceps_movement_repetition_signed_cnn_v2.pt"
        ),
    )
    parser.add_argument("--calibration-repetitions-per-class", type=int, default=3)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=(
            PUBLIC_ROOT / "outputs" / "evaluation"
            / "quadriceps_movement_personalized_metrics.json"
        ),
    )
    return parser.parse_args()


def standardise_imu(values, checkpoint):
    mean = np.asarray(
        checkpoint["imu_training_mean"], dtype=np.float32
    ).reshape(1, -1, 1)
    standard_deviation = np.asarray(
        checkpoint["imu_training_standard_deviation"], dtype=np.float32
    ).reshape(1, -1, 1)
    return np.clip((values - mean) / standard_deviation, -8, 8)


def tensors(data, indices):
    return (
        torch.from_numpy(data[0][indices, None, :]),
        torch.from_numpy(data[1][indices]),
        torch.from_numpy(data[2][indices]),
    )


def adapt_model(base_model, calibration, steps):
    model = deepcopy(base_model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    optimiser = torch.optim.AdamW(
        model.classifier.parameters(), lr=6e-4, weight_decay=5e-3
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.04)
    semg, imu, labels = calibration
    model.train()
    for _ in range(steps):
        optimiser.zero_grad()
        noisy_semg = semg * (0.94 + 0.12 * torch.rand(len(semg), 1, 1))
        noisy_semg = noisy_semg + 0.01 * torch.randn_like(noisy_semg)
        noisy_imu = imu + 0.01 * torch.randn_like(imu)
        loss = loss_function(model(noisy_semg, noisy_imu), labels)
        loss.backward()
        optimiser.step()
    model.eval()
    return model


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model = QuadricepsMovementRepetitionCNN()
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()
    test_subjects = checkpoint["test_metrics"]["test_subjects"]
    data = list(build_examples(args.dataset_zip, test_subjects))
    data[1] = standardise_imu(data[1], checkpoint)
    data = tuple(data)

    all_targets, all_predictions, subject_results = [], [], []
    for subject in test_subjects:
        subject_indices = np.flatnonzero(data[3] == subject)
        calibration_indices, evaluation_indices = [], []
        for class_index in range(len(CLASS_NAMES)):
            class_indices = subject_indices[data[2][subject_indices] == class_index]
            count = min(args.calibration_repetitions_per_class, len(class_indices) - 1)
            calibration_indices.extend(class_indices[:count].tolist())
            evaluation_indices.extend(class_indices[count:].tolist())
        calibration_indices = np.asarray(calibration_indices, dtype=np.int64)
        evaluation_indices = np.asarray(evaluation_indices, dtype=np.int64)
        model = adapt_model(
            base_model, tensors(data, calibration_indices), args.steps
        )
        semg, imu, labels = tensors(data, evaluation_indices)
        with torch.no_grad():
            predictions = model(semg, imu).argmax(dim=1).numpy()
        targets = labels.numpy()
        result = classification_metrics(targets, predictions, CLASS_NAMES)
        result.update({
            "subject": int(subject),
            "calibration_sample_count": int(len(calibration_indices)),
        })
        subject_results.append(result)
        all_targets.extend(targets.tolist())
        all_predictions.extend(predictions.tolist())

    aggregate = classification_metrics(
        np.asarray(all_targets), np.asarray(all_predictions), CLASS_NAMES
    )
    aggregate.update({
        "evaluation_kind": "held-out-participant-after-supervised-calibration",
        "calibration_repetitions_per_class": args.calibration_repetitions_per_class,
        "held_out_subject_count": len(test_subjects),
        "test_subjects": test_subjects,
        "subject_results": subject_results,
        "zero_calibration_reference": checkpoint["test_metrics"],
    })
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

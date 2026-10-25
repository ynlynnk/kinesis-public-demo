"""Train a participant-separated hierarchical quadriceps movement CNN.

Stage 1 detects whether a repetition matches the correct-extension pattern.
Stage 2 distinguishes the two review categories.  Both stages use the same
single rectus-femoris sEMG channel and co-located six-axis IMU information.
Model selection and the Stage-1 threshold use validation participants only.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_quadriceps_movement_repetition_cnn import (
    CLASS_NAMES,
    PROJECT_ROOT,
    SEED,
    build_examples,
    classification_metrics,
    standardise_imu,
)


QUALITY_NAMES = ("Correct extension", "Movement requiring review")
REVIEW_NAMES = ("Incomplete extension", "Compensatory leg lift")


class RepetitionCNN(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.semg_features = nn.Sequential(
            nn.Conv1d(2, 24, 15, stride=2, padding=7),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, 11, stride=2, padding=5),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.imu_features = nn.Sequential(
            nn.Conv1d(8, 32, 9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.45),
            nn.Linear(128 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, class_count),
        )

    def forward(self, semg, imu):
        envelope = F.avg_pool1d(semg.abs(), 41, stride=1, padding=20)
        semg_features = self.semg_features(torch.cat([semg, envelope], dim=1))
        imu_features = self.imu_features(imu)
        return self.classifier(
            torch.cat([semg_features.flatten(1), imu_features.flatten(1)], dim=1)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "raw" / "quadriceps_movement"
            / "kneepad" / "dataset.zip"
        ),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "models" / "candidates"
            / "kinesis_quadriceps_movement_hierarchical_cnn_v2.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def remap_data(data, labels, keep=None):
    indices = np.arange(len(data[2])) if keep is None else np.flatnonzero(keep)
    return data[0][indices], data[1][indices], labels[indices], data[3][indices]


def make_loader(data, batch_size, shuffle):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(data[0][:, None, :]),
            torch.from_numpy(data[1]),
            torch.from_numpy(data[2]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def collect_logits(model, loader):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for semg, imu, target in loader:
            logits.append(model(semg, imu))
            labels.append(target)
    return torch.cat(logits), torch.cat(labels)


def train_stage(train_data, validation_data, class_names, epochs, batch_size):
    train_loader = make_loader(train_data, batch_size, True)
    validation_loader = make_loader(validation_data, batch_size, False)
    model = RepetitionCNN(len(class_names))
    counts = torch.bincount(
        train_loader.dataset.tensors[2], minlength=len(class_names)
    ).float()
    weights = counts.sum() / torch.clamp(counts * len(class_names), min=1.0)
    loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.04)
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=4, min_lr=2e-5
    )
    best_state, best_f1, remaining, history = None, -1.0, 12, []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, sample_count = 0.0, 0
        for semg, imu, target in train_loader:
            optimiser.zero_grad()
            if epoch > 1:
                semg = semg * (0.92 + 0.16 * torch.rand(len(semg), 1, 1))
                semg = semg + 0.015 * torch.randn_like(semg)
                imu = imu + 0.015 * torch.randn_like(imu)
            loss = loss_function(model(semg, imu), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimiser.step()
            running_loss += float(loss.detach()) * len(target)
            sample_count += len(target)
        validation_logits, validation_targets = collect_logits(
            model, validation_loader
        )
        predictions = validation_logits.argmax(dim=1).numpy()
        result = classification_metrics(
            validation_targets.numpy(), predictions, class_names
        )
        scheduler.step(result["macro_f1"])
        record = {
            "epoch": epoch,
            "training_loss": running_loss / sample_count,
            "validation_accuracy": result["accuracy"],
            "validation_macro_f1": result["macro_f1"],
        }
        history.append(record)
        if result["macro_f1"] > best_f1 + 1e-5:
            best_f1 = result["macro_f1"]
            best_state = deepcopy(model.state_dict())
            remaining = 12
        else:
            remaining -= 1
            if remaining <= 0:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def select_quality_threshold(probabilities, targets):
    best = None
    for threshold in np.linspace(0.15, 0.85, 141):
        predictions = (probabilities >= threshold).astype(np.int64)
        result = classification_metrics(targets, predictions, QUALITY_NAMES)
        key = (result["macro_f1"], result["balanced_accuracy"])
        if best is None or key > best[0]:
            best = key, float(threshold), result
    return best[1], best[2]


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    usable_subjects = [subject for subject in range(1, 32) if subject not in {12, 25}]
    shuffled = np.random.default_rng(SEED).permutation(usable_subjects).tolist()
    training_subjects = sorted(int(value) for value in shuffled[:20])
    validation_subjects = sorted(int(value) for value in shuffled[20:24])
    test_subjects = sorted(int(value) for value in shuffled[24:])
    train = build_examples(args.dataset_zip, training_subjects)
    validation = build_examples(args.dataset_zip, validation_subjects)
    test = build_examples(args.dataset_zip, test_subjects)
    train, validation, test, imu_mean, imu_std = standardise_imu(
        train, validation, test
    )

    train_quality = remap_data(train, (train[2] != 0).astype(np.int64))
    validation_quality = remap_data(
        validation, (validation[2] != 0).astype(np.int64)
    )
    quality_model, quality_history = train_stage(
        train_quality, validation_quality, QUALITY_NAMES, args.epochs,
        args.batch_size,
    )

    train_review = remap_data(train, train[2] - 1, train[2] != 0)
    validation_review = remap_data(
        validation, validation[2] - 1, validation[2] != 0
    )
    review_model, review_history = train_stage(
        train_review, validation_review, REVIEW_NAMES, args.epochs,
        args.batch_size,
    )

    validation_logits, validation_targets = collect_logits(
        quality_model, make_loader(validation_quality, args.batch_size, False)
    )
    validation_wrong_probability = torch.softmax(
        validation_logits, dim=1
    )[:, 1].numpy()
    threshold, validation_quality_metrics = select_quality_threshold(
        validation_wrong_probability, validation_targets.numpy()
    )

    test_quality = remap_data(test, (test[2] != 0).astype(np.int64))
    test_quality_logits, test_quality_targets = collect_logits(
        quality_model, make_loader(test_quality, args.batch_size, False)
    )
    wrong_probability = torch.softmax(test_quality_logits, dim=1)[:, 1].numpy()
    quality_predictions = (wrong_probability >= threshold).astype(np.int64)
    quality_metrics = classification_metrics(
        test_quality_targets.numpy(), quality_predictions, QUALITY_NAMES
    )

    test_all_with_original_labels = remap_data(test, test[2])
    review_logits, _ = collect_logits(
        review_model,
        make_loader(test_all_with_original_labels, args.batch_size, False),
    )
    review_predictions = review_logits.argmax(dim=1).numpy()
    predictions = np.where(quality_predictions == 0, 0, review_predictions + 1)
    three_class_metrics = classification_metrics(
        test[2], predictions, CLASS_NAMES
    )
    for result in (quality_metrics, three_class_metrics):
        result.update({
            "held_out_subject_count": len(test_subjects),
            "test_subjects": test_subjects,
        })

    checkpoint = {
        "model_version": "quadriceps-extension-hierarchical-real-v2-candidate",
        "model_type": "two-stage-dual-branch-repetition-1d-cnn",
        "quality_model_state_dict": quality_model.state_dict(),
        "review_model_state_dict": review_model.state_dict(),
        "quality_threshold": threshold,
        "training_data_kind": "real_public_semg_imu",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "muscle_group": "Quadriceps",
        "exercise": "Seated knee extension",
        "class_names": list(CLASS_NAMES),
        "source_dataset": "KneE-PAD",
        "source_url": "https://zenodo.org/records/12112951",
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "test_subjects": test_subjects,
        "imu_training_mean": imu_mean.reshape(-1).tolist(),
        "imu_training_standard_deviation": imu_std.reshape(-1).tolist(),
        "validation_quality_metrics": validation_quality_metrics,
        "test_quality_metrics": quality_metrics,
        "test_three_class_metrics": three_class_metrics,
        "quality_training_history": quality_history,
        "review_training_history": review_history,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)
    print(json.dumps({
        "model_file": str(args.model_file),
        "quality_threshold_selected_on_validation": threshold,
        "validation_quality_metrics": validation_quality_metrics,
        "test_quality_metrics": quality_metrics,
        "test_three_class_metrics": three_class_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

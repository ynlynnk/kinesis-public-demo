"""Train a participant-separated repetition-level quadriceps movement CNN.

The candidate uses one affected-side rectus-femoris sEMG channel and all six
co-located IMU axes.  Every example is one complete seated-leg-extension
repetition, avoiding ambiguous partial windows while retaining the original
five held-out participants.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 20261002
LABEL_IDS = (3, 4, 5)
CLASS_NAMES = (
    "Correct extension",
    "Incomplete extension",
    "Compensatory leg lift",
)
QUALITY_CLASS_NAMES = ("Correct extension", "Movement requiring review")
SEMG_SAMPLES = 800
IMU_SAMPLES = 200
RIGHT_LEG_SUBJECTS = {
    4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 21, 22, 23, 25, 26, 28,
    29, 31,
}


class QuadricepsMovementRepetitionCNN(nn.Module):
    def __init__(self):
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
            nn.Dropout(0.40),
            nn.Linear((64 + 64) * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, semg, imu):
        envelope = F.avg_pool1d(semg.abs(), 41, stride=1, padding=20)
        semg_features = self.semg_features(torch.cat([semg, envelope], dim=1))
        imu_features = self.imu_features(imu)
        return self.classifier(
            torch.cat([semg_features.flatten(1), imu_features.flatten(1)], dim=1)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "quadriceps_movement"
            / "kneepad"
            / "dataset.zip"
        ),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "models"
            / "candidates"
            / "kinesis_quadriceps_movement_repetition_cnn_v2.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def resample_to_count(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    old_axis = np.linspace(0.0, 1.0, values.shape[-1], dtype=np.float64)
    new_axis = np.linspace(0.0, 1.0, count, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(new_axis, old_axis, values).astype(np.float32)
    return np.stack([
        np.interp(new_axis, old_axis, row).astype(np.float32) for row in values
    ])


def build_examples(dataset_zip: Path, subjects: list[int]):
    matcher = re.compile(r"dataset/Subject_(\d+)/(\d+)/Trial_(\d+)/emg\.npy$")
    semg_examples, imu_examples, labels, subject_ids = [], [], [], []
    wanted = set(subjects)
    with zipfile.ZipFile(dataset_zip) as archive:
        entries = []
        for name in archive.namelist():
            match = matcher.match(name)
            if not match:
                continue
            subject, label, trial = map(int, match.groups())
            if subject in wanted and label in LABEL_IDS:
                entries.append((subject, label, trial, name))

        for subject, label, _, name in sorted(entries):
            emg = np.load(io.BytesIO(archive.read(name))).astype(np.float32)
            imu = np.load(
                io.BytesIO(archive.read(name.replace("emg.npy", "imu.npy")))
            ).astype(np.float32)
            sensor = 0 if subject in RIGHT_LEG_SUBJECTS else 4
            semg_signal = resample_to_count(emg[sensor], SEMG_SAMPLES)
            semg_signal = (
                semg_signal - semg_signal.mean()
            ) / max(float(semg_signal.std()), 1e-3)
            semg_signal = np.clip(semg_signal, -12.0, 12.0)

            offset = sensor * 6
            accelerometer = imu[offset : offset + 3]
            gyroscope = np.deg2rad(imu[offset + 3 : offset + 6]).astype(np.float32)
            # Preserve axis direction.  The earlier candidate used absolute
            # values for every axis, which discarded the directional cues
            # needed to distinguish a normal extension from a compensatory
            # lift.  Remove only the starting-pose baseline while retaining a
            # rotation-invariant magnitude channel for each sensor type.
            baseline_count = max(1, accelerometer.shape[1] // 10)
            accelerometer = (
                accelerometer
                - accelerometer[:, :baseline_count].mean(axis=1, keepdims=True)
            )
            imu_channels = np.concatenate([
                accelerometer,
                np.linalg.norm(accelerometer, axis=0, keepdims=True),
                gyroscope,
                np.linalg.norm(gyroscope, axis=0, keepdims=True),
            ])
            imu_channels = resample_to_count(imu_channels, IMU_SAMPLES)
            semg_examples.append(semg_signal)
            imu_examples.append(imu_channels)
            labels.append(LABEL_IDS.index(label))
            subject_ids.append(subject)
    return (
        np.asarray(semg_examples, dtype=np.float32),
        np.asarray(imu_examples, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subject_ids, dtype=np.int64),
    )


def standardise_imu(train, *others):
    mean = train[1].mean(axis=(0, 2), keepdims=True)
    standard_deviation = np.maximum(train[1].std(axis=(0, 2), keepdims=True), 1e-3)

    def transform(data):
        values = list(data)
        values[1] = np.clip((values[1] - mean) / standard_deviation, -8, 8)
        return tuple(values)

    return transform(train), *(transform(data) for data in others), mean, standard_deviation


def make_loader(data, batch_size: int, shuffle: bool):
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
    logits, targets = [], []
    with torch.no_grad():
        for semg, imu, labels in loader:
            logits.append(model(semg, imu))
            targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def classification_metrics(targets, predictions, class_names):
    class_count = len(class_names)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    f1_values = []
    for index in range(class_count):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": {
            name: float(recalls[index]) for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(targets)),
    }


def train_model(train_loader, validation_loader, epochs):
    model = QuadricepsMovementRepetitionCNN()
    counts = torch.bincount(train_loader.dataset.tensors[2], minlength=3).float()
    weights = counts.sum() / torch.clamp(counts * 3, min=1.0)
    loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=4, min_lr=2e-5
    )
    best_state = None
    best_f1 = -1.0
    remaining = 12
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for semg, imu, labels in train_loader:
            optimiser.zero_grad()
            if epoch > 1:
                semg = semg * (0.95 + 0.10 * torch.rand(len(semg), 1, 1))
                semg = semg + 0.012 * torch.randn_like(semg)
                imu = imu + 0.010 * torch.randn_like(imu)
            loss = loss_function(model(semg, imu), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimiser.step()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
        validation_logits, validation_targets = collect_logits(model, validation_loader)
        predictions = validation_logits.argmax(dim=1).numpy()
        result = classification_metrics(
            validation_targets.numpy(), predictions, CLASS_NAMES
        )
        scheduler.step(result["macro_f1"])
        summary = {
            "epoch": epoch,
            "training_loss": total_loss / count,
            "validation_accuracy": result["accuracy"],
            "validation_macro_f1": result["macro_f1"],
            "learning_rate": optimiser.param_groups[0]["lr"],
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)
        if result["macro_f1"] > best_f1 + 1e-5:
            best_f1 = result["macro_f1"]
            best_state = deepcopy(model.state_dict())
            remaining = 12
        else:
            remaining -= 1
            if remaining <= 0:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def fit_temperature(logits, labels):
    log_temperature = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=60, line_search_fn="strong_wolfe"
    )
    loss_function = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        temperature = log_temperature.exp().clamp(0.25, 10.0)
        loss = loss_function(logits / temperature, labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.detach().exp().clamp(0.25, 10.0))


def main() -> None:
    args = parse_args()
    set_seed()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    usable_subjects = [subject for subject in range(1, 32) if subject not in {12, 25}]
    shuffled = np.random.default_rng(SEED).permutation(usable_subjects).tolist()
    training_subjects = sorted(int(value) for value in shuffled[:20])
    validation_subjects = sorted(int(value) for value in shuffled[20:24])
    test_subjects = sorted(int(value) for value in shuffled[24:])
    train_data = build_examples(args.dataset_zip, training_subjects)
    validation_data = build_examples(args.dataset_zip, validation_subjects)
    test_data = build_examples(args.dataset_zip, test_subjects)
    train_data, validation_data, test_data, imu_mean, imu_standard_deviation = (
        standardise_imu(train_data, validation_data, test_data)
    )
    model, history = train_model(
        make_loader(train_data, args.batch_size, True),
        make_loader(validation_data, args.batch_size, False),
        args.epochs,
    )
    validation_logits, validation_targets = collect_logits(
        model, make_loader(validation_data, args.batch_size, False)
    )
    temperature = fit_temperature(validation_logits, validation_targets)
    test_logits, test_targets = collect_logits(
        model, make_loader(test_data, args.batch_size, False)
    )
    predictions = (test_logits / temperature).argmax(dim=1).numpy()
    targets = test_targets.numpy()
    three_class = classification_metrics(targets, predictions, CLASS_NAMES)
    binary_targets = (targets != 0).astype(np.int64)
    binary_predictions = (predictions != 0).astype(np.int64)
    quality = classification_metrics(
        binary_targets, binary_predictions, QUALITY_CLASS_NAMES
    )
    three_class.update({
        "held_out_subject_count": len(test_subjects),
        "test_subjects": test_subjects,
    })
    quality.update({
        "held_out_subject_count": len(test_subjects),
        "test_subjects": test_subjects,
    })
    checkpoint = {
        "model_version": "quadriceps-extension-repetition-real-v2-candidate",
        "model_type": "dual-branch-semg-six-axis-imu-repetition-1d-cnn",
        "model_state_dict": model.state_dict(),
        "training_data_kind": "real_public_semg_imu",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "muscle_group": "Quadriceps",
        "exercise": "Seated knee extension",
        "class_names": list(CLASS_NAMES),
        "semg_samples": SEMG_SAMPLES,
        "imu_samples": IMU_SAMPLES,
        "imu_training_mean": imu_mean.reshape(-1).tolist(),
        "imu_training_standard_deviation": imu_standard_deviation.reshape(-1).tolist(),
        "source_dataset": "KneE-PAD",
        "source_url": "https://zenodo.org/records/12112951",
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "test_metrics": three_class,
        "test_quality_metrics": quality,
        "training_history": history,
        "temperature": temperature,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)
    print(json.dumps({
        "model_file": str(args.model_file),
        "temperature": temperature,
        "test_three_class_metrics": three_class,
        "test_quality_metrics": quality,
    }, indent=2))


if __name__ == "__main__":
    main()

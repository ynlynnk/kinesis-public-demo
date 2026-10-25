"""Train an improved participant-separated quadriceps movement CNN.

This candidate keeps the original held-out participants unchanged.  It uses
longer windows, preserves calibrated angular-speed magnitude, retains coarse
temporal structure, and reports both window- and repetition-level metrics.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quadriceps_runtime import MOVEMENT_CLASS_NAMES as SOURCE_CLASS_NAMES  # noqa: E402


SEED = 20261002
SOURCE_SAMPLE_RATE_HZ = 1259.2592592592594
MODEL_SAMPLE_RATE_HZ = 200
SOURCE_IMU_RATE_HZ = 148.14814814814815
MODEL_IMU_RATE_HZ = 50
WINDOW_SAMPLES = 512
IMU_WINDOW_SAMPLES = 128
WINDOW_STRIDE = 256
LABEL_IDS = (3, 4, 5)
MOVEMENT_CLASS_NAMES = ("Correct extension", "Movement requiring review")
RIGHT_LEG_SUBJECTS = {
    4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 21, 22, 23, 25, 26, 28,
    29, 31,
}


class QuadricepsMovementCNNV2(nn.Module):
    """Dual-branch CNN that preserves coarse within-repetition timing."""

    def __init__(self, class_count: int = len(MOVEMENT_CLASS_NAMES)):
        super().__init__()
        self.semg_features = nn.Sequential(
            nn.Conv1d(2, 24, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(6),
        )
        self.imu_features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(6),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear((64 + 32) * 6, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, class_count),
        )

    def forward(self, semg_window, imu_window):
        semg_envelope = F.avg_pool1d(
            semg_window.abs(), kernel_size=31, stride=1, padding=15
        )
        imu_envelope = F.avg_pool1d(
            imu_window.abs(), kernel_size=9, stride=1, padding=4
        )
        semg = self.semg_features(
            torch.cat([semg_window, semg_envelope], dim=1)
        )
        imu = self.imu_features(torch.cat([imu_window, imu_envelope], dim=1))
        return self.classifier(torch.cat([semg.flatten(1), imu.flatten(1)], dim=1))


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
            / "kinesis_quadriceps_movement_quality_cnn_v2.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=96)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def resample(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    target_count = int(round(len(signal) * target_rate / source_rate))
    source_axis = np.linspace(0.0, 1.0, len(signal), dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, target_count, dtype=np.float64)
    return np.interp(target_axis, source_axis, signal).astype(np.float32)


def window_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def padded_window(values: np.ndarray, start: int, size: int) -> np.ndarray:
    selected = values[start : start + size]
    if len(selected) == size:
        return selected
    return np.pad(selected, (0, size - len(selected)), mode="edge")


def build_examples(dataset_zip: Path, subjects: list[int]):
    matcher = re.compile(r"dataset/Subject_(\d+)/(\d+)/Trial_(\d+)/emg\.npy$")
    semg_windows, imu_windows, labels = [], [], []
    subject_ids, repetition_ids = [], []
    wanted = set(subjects)
    repetition_id = 0

    with zipfile.ZipFile(dataset_zip) as archive:
        entries = []
        for name in archive.namelist():
            match = matcher.match(name)
            if not match:
                continue
            subject, label_id, trial = map(int, match.groups())
            if subject in wanted and label_id in LABEL_IDS:
                entries.append((subject, label_id, trial, name))

        for subject, label_id, _, name in sorted(entries):
            recording = np.load(io.BytesIO(archive.read(name))).astype(np.float32)
            channel_index = 0 if subject in RIGHT_LEG_SUBJECTS else 4
            signal = resample(
                recording[channel_index], SOURCE_SAMPLE_RATE_HZ, MODEL_SAMPLE_RATE_HZ
            )
            signal = (signal - signal.mean()) / max(float(signal.std()), 1e-3)
            signal = np.clip(signal, -12.0, 12.0)

            imu_recording = np.load(
                io.BytesIO(archive.read(name.replace("emg.npy", "imu.npy")))
            ).astype(np.float32)
            sensor_offset = channel_index * 6
            gyro_degrees = imu_recording[sensor_offset + 3 : sensor_offset + 6]
            angular_speed = np.linalg.norm(gyro_degrees, axis=0)
            angular_speed = np.deg2rad(angular_speed).astype(np.float32)
            angular_speed = np.log1p(angular_speed)
            angular_speed = resample(
                angular_speed, SOURCE_IMU_RATE_HZ, MODEL_IMU_RATE_HZ
            )

            for start in window_starts(len(signal), WINDOW_SAMPLES, WINDOW_STRIDE):
                imu_start = int(round(start * MODEL_IMU_RATE_HZ / MODEL_SAMPLE_RATE_HZ))
                semg_windows.append(padded_window(signal, start, WINDOW_SAMPLES))
                imu_windows.append(
                    padded_window(angular_speed, imu_start, IMU_WINDOW_SAMPLES)
                )
                labels.append(0 if label_id == 3 else 1)
                subject_ids.append(subject)
                repetition_ids.append(repetition_id)
            repetition_id += 1

    if not semg_windows:
        raise ValueError("No matching knee-extension recordings were found.")
    return (
        np.asarray(semg_windows, dtype=np.float32),
        np.asarray(imu_windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subject_ids, dtype=np.int64),
        np.asarray(repetition_ids, dtype=np.int64),
    )


def standardise_imu(train_data, *other_data):
    mean = float(train_data[1].mean())
    standard_deviation = max(float(train_data[1].std()), 1e-3)

    def transform(data):
        values = list(data)
        values[1] = np.clip(
            (values[1] - mean) / standard_deviation, -8.0, 8.0
        ).astype(np.float32)
        return tuple(values)

    return transform(train_data), *(transform(data) for data in other_data), mean, standard_deviation


def make_loader(data, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(data[0][:, None, :]),
            torch.from_numpy(data[1][:, None, :]),
            torch.from_numpy(data[2]),
            torch.from_numpy(data[4]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def collect_logits(model: nn.Module, loader: DataLoader):
    model.eval()
    logits, targets, repetitions = [], [], []
    with torch.no_grad():
        for semg, imu, labels, repetition_ids in loader:
            logits.append(model(semg, imu))
            targets.append(labels)
            repetitions.append(repetition_ids)
    return torch.cat(logits), torch.cat(targets), torch.cat(repetitions)


def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    class_count = len(MOVEMENT_CLASS_NAMES)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    f1_values = []
    for index in range(class_count):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": {
            name: float(recalls[index])
            for index, name in enumerate(MOVEMENT_CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(targets)),
    }


def repetition_metrics(logits, targets, repetition_ids) -> dict:
    logits = logits.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    repetition_ids = repetition_ids.detach().cpu().numpy()
    grouped_logits, grouped_targets = [], []
    for repetition_id in np.unique(repetition_ids):
        mask = repetition_ids == repetition_id
        grouped_logits.append(logits[mask].mean(axis=0))
        grouped_targets.append(int(targets[mask][0]))
    predictions = np.asarray(grouped_logits).argmax(axis=1)
    return classification_metrics(np.asarray(grouped_targets), predictions)


def train_model(train_loader, validation_loader, epochs: int):
    model = QuadricepsMovementCNNV2()
    class_count = len(MOVEMENT_CLASS_NAMES)
    counts = torch.bincount(
        train_loader.dataset.tensors[2], minlength=class_count
    ).float()
    weights = counts.sum() / torch.clamp(counts * class_count, min=1.0)
    loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.04)
    optimiser = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=4, min_lr=2e-5
    )
    best_state = None
    best_score = -1.0
    patience_remaining = 12
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for semg, imu, labels, _ in train_loader:
            optimiser.zero_grad()
            if epoch > 1:
                semg = semg * (0.94 + 0.12 * torch.rand(len(semg), 1, 1))
                semg = semg + 0.015 * torch.randn_like(semg)
                imu = imu + 0.010 * torch.randn_like(imu)
            loss = loss_function(model(semg, imu), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimiser.step()
            total_loss += float(loss.detach()) * len(labels)
            total_count += len(labels)

        validation_logits, validation_targets, validation_repetitions = collect_logits(
            model, validation_loader
        )
        validation = repetition_metrics(
            validation_logits, validation_targets, validation_repetitions
        )
        score = validation["macro_f1"]
        scheduler.step(score)
        summary = {
            "epoch": epoch,
            "training_loss": total_loss / total_count,
            "validation_repetition_accuracy": validation["accuracy"],
            "validation_repetition_macro_f1": score,
            "learning_rate": optimiser.param_groups[0]["lr"],
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)

        if score > best_score + 1e-5:
            best_score = score
            best_state = deepcopy(model.state_dict())
            patience_remaining = 12
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
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
    if not args.dataset_zip.is_file():
        raise FileNotFoundError(args.dataset_zip)
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
    validation_logits, validation_targets, _ = collect_logits(
        model, make_loader(validation_data, args.batch_size, False)
    )
    temperature = fit_temperature(validation_logits, validation_targets)
    test_logits, test_targets, test_repetitions = collect_logits(
        model, make_loader(test_data, args.batch_size, False)
    )
    calibrated_test_logits = test_logits / temperature
    window_predictions = calibrated_test_logits.argmax(dim=1).numpy()
    window_metrics = classification_metrics(test_targets.numpy(), window_predictions)
    trial_metrics = repetition_metrics(
        calibrated_test_logits, test_targets, test_repetitions
    )

    checkpoint = {
        "model_version": "quadriceps-extension-quality-real-v2-candidate",
        "model_type": "dual-branch-semg-imu-temporal-1d-cnn",
        "model_state_dict": model.state_dict(),
        "training_data_kind": "real_public_semg",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "muscle_group": "Quadriceps",
        "exercise": "Seated knee extension",
        "class_names": list(MOVEMENT_CLASS_NAMES),
        "source_class_names": list(SOURCE_CLASS_NAMES),
        "label_mapping": {
            "Correct extension": "Correct extension",
            "Incomplete extension": "Movement requiring review",
            "Compensatory leg lift": "Movement requiring review",
        },
        "sample_rate_hz": MODEL_SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "imu_sample_rate_hz": MODEL_IMU_RATE_HZ,
        "imu_window_samples": IMU_WINDOW_SAMPLES,
        "window_seconds": WINDOW_SAMPLES / MODEL_SAMPLE_RATE_HZ,
        "normalization": "per-recording rectus-femoris z-score",
        "imu_transform": "log1p angular-speed magnitude in radians/second",
        "imu_training_mean": imu_mean,
        "imu_training_standard_deviation": imu_standard_deviation,
        "temperature": temperature,
        "source_dataset": "KneE-PAD",
        "source_url": "https://zenodo.org/records/12112951",
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "test_metrics": window_metrics,
        "test_repetition_metrics": trial_metrics,
        "training_history": history,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)
    print(json.dumps({
        "model_file": str(args.model_file),
        "temperature": temperature,
        "test_window_metrics": window_metrics,
        "test_repetition_metrics": trial_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

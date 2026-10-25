"""Train a participant-separated quadriceps movement CNN on KneE-PAD.

The model uses the affected-side rectus-femoris sEMG channel and predicts the
three seated knee-extension executions supplied by the dataset: correct,
incomplete range of motion, and compensatory lifting from the chair.
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
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quadriceps_runtime import (  # noqa: E402
    MOVEMENT_CLASS_NAMES,
    QuadricepsMovementCNN,
)


SEED = 20261002
SOURCE_SAMPLE_RATE_HZ = 1259.2592592592594
MODEL_SAMPLE_RATE_HZ = 200
SOURCE_IMU_RATE_HZ = 148.14814814814815
MODEL_IMU_RATE_HZ = 50
WINDOW_SAMPLES = 256
IMU_WINDOW_SAMPLES = 64
WINDOW_STRIDE = 128
LABEL_IDS = (3, 4, 5)
SOURCE_DOI = "10.5281/zenodo.12112951"
SOURCE_URL = "https://zenodo.org/records/12112951"
RIGHT_LEG_SUBJECTS = {
    4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 21, 22, 23, 25, 26, 28,
    29, 31,
}


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
            / "kinesis_quadriceps_movement_cnn.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
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


def build_examples(
    dataset_zip: Path,
    subjects: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matcher = re.compile(
        r"dataset/Subject_(\d+)/(\d+)/Trial_(\d+)/emg\.npy$"
    )
    windows: list[np.ndarray] = []
    imu_windows: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[int] = []
    wanted = set(subjects)

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
            recording = np.load(io.BytesIO(archive.read(name)))
            channel_index = 0 if subject in RIGHT_LEG_SUBJECTS else 4
            signal = resample(
                recording[channel_index].astype(np.float32),
                SOURCE_SAMPLE_RATE_HZ,
                MODEL_SAMPLE_RATE_HZ,
            )
            signal = (signal - signal.mean()) / max(float(signal.std()), 1e-3)
            signal = np.clip(signal, -12.0, 12.0)
            imu_name = name.replace("emg.npy", "imu.npy")
            imu_recording = np.load(io.BytesIO(archive.read(imu_name)))
            sensor_offset = channel_index * 6
            gyro = imu_recording[sensor_offset + 3 : sensor_offset + 6]
            angular_speed = np.linalg.norm(gyro, axis=0).astype(np.float32)
            angular_speed = resample(
                angular_speed,
                SOURCE_IMU_RATE_HZ,
                MODEL_IMU_RATE_HZ,
            )
            angular_speed = (
                angular_speed - angular_speed.mean()
            ) / max(float(angular_speed.std()), 1e-3)
            angular_speed = np.clip(angular_speed, -12.0, 12.0)
            for start in range(0, len(signal) - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
                start_seconds = start / MODEL_SAMPLE_RATE_HZ
                imu_start = int(round(start_seconds * MODEL_IMU_RATE_HZ))
                if imu_start + IMU_WINDOW_SAMPLES > len(angular_speed):
                    continue
                windows.append(signal[start : start + WINDOW_SAMPLES])
                imu_windows.append(
                    angular_speed[imu_start : imu_start + IMU_WINDOW_SAMPLES]
                )
                labels.append(LABEL_IDS.index(label_id))
                subject_ids.append(subject)

    if not windows:
        raise ValueError("No matching knee-extension recordings were found.")
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(imu_windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subject_ids, dtype=np.int64),
    )


def make_loader(windows, imu_windows, labels, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(windows[:, None, :]),
            torch.from_numpy(imu_windows[:, None, :]),
            torch.from_numpy(labels),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def collect_logits(model: nn.Module, loader: DataLoader):
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for windows, imu_windows, labels in loader:
            logits.append(model(windows, imu_windows))
            targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def confusion_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
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
        "per_class_recall": {
            name: float(recalls[index])
            for index, name in enumerate(MOVEMENT_CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(targets)),
    }


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


def train_model(train_loader, validation_loader, epochs: int):
    model = QuadricepsMovementCNN()
    counts = torch.bincount(train_loader.dataset.tensors[2], minlength=3).float()
    weights = counts.sum() / torch.clamp(counts * len(counts), min=1.0)
    loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    best_state = None
    best_accuracy = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for windows, imu_windows, labels in train_loader:
            optimiser.zero_grad()
            loss = loss_function(model(windows, imu_windows), labels)
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * len(labels)
            total_count += len(labels)
        validation_logits, validation_labels = collect_logits(model, validation_loader)
        accuracy = float(
            (validation_logits.argmax(dim=1) == validation_labels).float().mean()
        )
        summary = {
            "epoch": epoch,
            "training_loss": total_loss / total_count,
            "validation_accuracy": accuracy,
        }
        history.append(summary)
        print(json.dumps(summary))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, history


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
    model, history = train_model(
        make_loader(train_data[0], train_data[1], train_data[2], args.batch_size, True),
        make_loader(
            validation_data[0], validation_data[1], validation_data[2],
            args.batch_size, False,
        ),
        args.epochs,
    )
    validation_logits, validation_targets = collect_logits(
        model,
        make_loader(
            validation_data[0], validation_data[1], validation_data[2],
            args.batch_size, False,
        ),
    )
    temperature = fit_temperature(validation_logits, validation_targets)
    test_logits, test_targets = collect_logits(
        model,
        make_loader(test_data[0], test_data[1], test_data[2], args.batch_size, False),
    )
    probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
    metrics = confusion_metrics(test_targets.numpy(), probabilities.argmax(axis=1))
    metrics.update({
        "held_out_subject_count": len(test_subjects),
        "test_subjects": test_subjects,
    })

    checkpoint = {
        "model_version": "quadriceps-extension-real-v1",
        "model_type": "dual-branch-semg-imu-1d-cnn",
        "model_state_dict": model.state_dict(),
        "training_data_kind": "real_public_semg",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "muscle_group": "Quadriceps",
        "exercise": "Seated knee extension",
        "class_names": list(MOVEMENT_CLASS_NAMES),
        "sample_rate_hz": MODEL_SAMPLE_RATE_HZ,
        "source_sample_rate_hz": SOURCE_SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "imu_sample_rate_hz": MODEL_IMU_RATE_HZ,
        "source_imu_rate_hz": SOURCE_IMU_RATE_HZ,
        "imu_window_samples": IMU_WINDOW_SAMPLES,
        "window_seconds": WINDOW_SAMPLES / MODEL_SAMPLE_RATE_HZ,
        "normalization": "per-recording rectus-femoris z-score",
        "runtime_window_standardization": True,
        "temperature": temperature,
        "source_dataset": "KneE-PAD",
        "source_doi": SOURCE_DOI,
        "source_url": SOURCE_URL,
        "source_license": "CC BY 4.0",
        "source_channels": (
            "affected-side rectus femoris sEMG plus co-located gyroscope magnitude"
        ),
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "test_metrics": metrics,
        "training_history": history,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)
    print(json.dumps({
        "model_file": str(args.model_file),
        "temperature": temperature,
        "test_metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

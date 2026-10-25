"""Train a quadriceps fatigue-phase CNN on real public cycling sEMG.

The source labels no fatigue, transition to fatigue, and fatigue. Transition
and fatigue are combined into one positive fatigue-like phase so that the
dashboard score answers whether the current window resembles either labelled
fatigue stage, rather than discarding the final fatigue recordings.
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
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model1_signal_pipeline import EMA_ALPHA, ema_filter  # noqa: E402
from quadriceps_runtime import (  # noqa: E402
    FATIGUE_CLASS_NAMES,
    QuadricepsFatigueCNN,
)


SEED = 20261002
SOURCE_SAMPLE_RATE_HZ = 1000
DEVICE_SAMPLE_RATE_HZ = 500
MODEL_SAMPLE_RATE_HZ = 200
WINDOW_SAMPLES = 512
WINDOW_STRIDE = 256
BASELINE_SECONDS = 10
SOURCE_URL = "https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling"
CHANNEL_NAMES = ("Right rectus femoris", "Right vastus medialis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "quadriceps_fatigue"
            / "muscle_fatigue_cycling"
        ),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "models"
            / "kinesis_quadriceps_fatigue_cnn.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_participant(data_directory: Path, participant: int):
    data_file = data_directory / f"Patient{participant}.csv"
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    values = np.loadtxt(
        data_file,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1, 6, 9),
        dtype=np.float32,
    )
    sample_interval = float(np.median(np.diff(values[:5000, 0])))
    if not np.isclose(sample_interval, 1.0 / SOURCE_SAMPLE_RATE_HZ, atol=1e-6):
        raise ValueError(f"Unexpected timestamp interval in {data_file}.")
    return values[:, 1:3], values[:, 3].astype(np.int64)


def to_model_rate(signal: np.ndarray) -> np.ndarray:
    pair_count = len(signal) // 2
    device_signal = signal[: pair_count * 2].reshape(pair_count, 2).mean(axis=1)
    filtered, _ = ema_filter(device_signal.astype(np.float32), EMA_ALPHA)
    target_count = int(round(len(filtered) * MODEL_SAMPLE_RATE_HZ / DEVICE_SAMPLE_RATE_HZ))
    source_axis = np.linspace(0.0, 1.0, len(filtered), dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, target_count, dtype=np.float64)
    return np.interp(target_axis, source_axis, filtered).astype(np.float32)


def contiguous_runs(labels: np.ndarray, wanted_label: int):
    matches = labels == wanted_label
    changes = np.diff(np.concatenate(([False], matches, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return zip(starts, stops)


def build_participant_examples(data_directory: Path, participant: int):
    channels, labels = load_participant(data_directory, participant)
    baseline_count = BASELINE_SECONDS * SOURCE_SAMPLE_RATE_HZ
    baseline = channels[:baseline_count]
    baseline_mean = baseline.mean(axis=0)
    baseline_std = np.maximum(baseline.std(axis=0), 1e-3)
    normalized = np.clip((channels - baseline_mean) / baseline_std, -12.0, 12.0)

    windows, targets, participants, channel_ids = [], [], [], []
    for source_target in (0, 1, 2):
        target = 0 if source_target == 0 else 1
        for start, stop in contiguous_runs(labels, source_target):
            if stop - start < SOURCE_SAMPLE_RATE_HZ * 3:
                continue
            for channel_index in range(normalized.shape[1]):
                model_signal = to_model_rate(normalized[start:stop, channel_index])
                for window_start in range(
                    0,
                    len(model_signal) - WINDOW_SAMPLES + 1,
                    WINDOW_STRIDE,
                ):
                    windows.append(
                        model_signal[window_start : window_start + WINDOW_SAMPLES]
                    )
                    targets.append(target)
                    participants.append(participant)
                    channel_ids.append(channel_index)
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(participants, dtype=np.int64),
        np.asarray(channel_ids, dtype=np.int64),
    )


def balanced_subset(data, per_class_limit: int, seed: int):
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        candidates = np.flatnonzero(data[1] == label)
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class_limit])
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return tuple(values[selected] for values in data)


def make_loader(windows, labels, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(windows[:, None, :]),
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
        for windows, labels in loader:
            logits.append(model(windows))
            targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def binary_metrics(targets, predictions, positive_probabilities):
    matrix = np.zeros((2, 2), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    f1_values = []
    for index in range(2):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    positive = positive_probabilities[targets == 1]
    negative = positive_probabilities[targets == 0]
    comparisons = positive[:, None] - negative[None, :]
    auc = (
        np.count_nonzero(comparisons > 0)
        + 0.5 * np.count_nonzero(comparisons == 0)
    ) / comparisons.size
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "roc_auc": float(auc),
        "per_class_recall": {
            name: float(recalls[index])
            for index, name in enumerate(FATIGUE_CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(targets)),
    }


def train_model(train_loader, validation_loader, epochs: int):
    model = QuadricepsFatigueCNN()
    counts = torch.bincount(train_loader.dataset.tensors[1], minlength=2).float()
    weights = counts.sum() / torch.clamp(counts * 2, min=1.0)
    loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    best_state = None
    best_accuracy = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for windows, labels in train_loader:
            optimiser.zero_grad()
            loss = loss_function(model(windows), labels)
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
    set_seed()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    participant_data = {
        participant: build_participant_examples(args.data_directory, participant)
        for participant in (1, 2, 3)
    }
    training_parts = [
        balanced_subset(participant_data[participant], 300, SEED + participant)
        for participant in (1, 2)
    ]
    combined = tuple(
        np.concatenate([part[index] for part in training_parts])
        for index in range(4)
    )
    rng = np.random.default_rng(SEED)
    validation_indices = []
    training_indices = []
    for participant in (1, 2):
        for label in (0, 1):
            candidates = np.flatnonzero(
                (combined[2] == participant) & (combined[1] == label)
            )
            rng.shuffle(candidates)
            split = max(1, int(round(len(candidates) * 0.15)))
            validation_indices.extend(candidates[:split])
            training_indices.extend(candidates[split:])
    training_indices = np.asarray(sorted(training_indices), dtype=np.int64)
    validation_indices = np.asarray(sorted(validation_indices), dtype=np.int64)
    train_windows, train_labels = combined[0][training_indices], combined[1][training_indices]
    validation_windows = combined[0][validation_indices]
    validation_labels = combined[1][validation_indices]
    test_data = participant_data[3]

    model, history = train_model(
        make_loader(train_windows, train_labels, args.batch_size, True),
        make_loader(validation_windows, validation_labels, args.batch_size, False),
        args.epochs,
    )
    test_logits, test_targets = collect_logits(
        model,
        make_loader(test_data[0], test_data[1], args.batch_size, False),
    )
    probabilities = torch.softmax(test_logits, dim=1).numpy()
    metrics = binary_metrics(
        test_targets.numpy(),
        probabilities.argmax(axis=1),
        probabilities[:, 1],
    )
    metrics.update({"held_out_subject_count": 1, "test_subjects": [3]})

    checkpoint = {
        "model_version": "quadriceps-fatigue-phase-real-v1",
        "model_type": "single-channel-1d-cnn",
        "model_state_dict": model.state_dict(),
        "training_data_kind": "real_public_semg",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "label_kind": "self_perceived_transition_or_fatigue",
        "muscle_group": "Quadriceps",
        "exercise": "Fatiguing cycling protocol",
        "class_names": list(FATIGUE_CLASS_NAMES),
        "sample_rate_hz": MODEL_SAMPLE_RATE_HZ,
        "source_sample_rate_hz": SOURCE_SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SAMPLES / MODEL_SAMPLE_RATE_HZ,
        "normalization": "per-participant/channel first-10-second baseline",
        "temperature": 1.0,
        "source_dataset": "Muscle Fatigue in Cyclist",
        "source_url": SOURCE_URL,
        "source_license": "CC BY 4.0",
        "source_channels": list(CHANNEL_NAMES),
        "source_participant_count": 3,
        "training_subjects": [1, 2],
        "validation_kind": "stratified windows from training participants",
        "test_metrics": metrics,
        "training_history": history,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)
    print(json.dumps({
        "model_file": str(args.model_file),
        "test_metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

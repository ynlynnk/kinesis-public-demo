"""Train a subject-separated fatigue-phase CNN on real public upper-limb sEMG.

Dataset:
    Multi-channel Surface EMG Dataset for Fatigue analysis
    DOI: 10.5281/zenodo.5189275
    Licence: CC BY 4.0

Each participant held a 6 kg load with the elbow at 90 degrees for 120 seconds.
The first 30 seconds are used as the low-fatigue phase and the last 30 seconds
as the fatigue-like phase. These are protocol-phase labels, not clinical
fatigue diagnoses or self-reported severity labels.
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

from forearm_fatigue_runtime import (  # noqa: E402
    CLASS_NAMES,
    ForearmFatigueCNN,
)


SEED = 20260929
SAMPLE_RATE_HZ = 200
WINDOW_SAMPLES = 512
WINDOW_STRIDE = 256
BASELINE_SECONDS = 10
LOW_PHASE_SECONDS = (5, 35)
FATIGUE_PHASE_SECONDS = (90, 120)
SOURCE_DOI = "10.5281/zenodo.5189275"
SOURCE_URL = "https://zenodo.org/records/5189275"


def parse_args() -> argparse.Namespace:
    source_directory = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "forearm_fatigue"
        / "zenodo_5189275"
        / "extracted"
        / "Dataset EMG Fatigue"
        / "Data as txt Files"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=Path, default=source_directory)
    parser.add_argument(
        "--model-file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "models"
            / "kinesis_forearm_fatigue_cnn.pt"
        ),
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "models"
            / "kinesis_fatigue_verification_samples.npz"
        ),
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def set_reproducible_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_subject(data_directory: Path, subject: int) -> np.ndarray:
    data_file = data_directory / f"sub{subject}.txt"
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    recording = np.loadtxt(data_file, dtype=np.float32)
    if recording.ndim != 2 or recording.shape[1] != 8:
        raise ValueError(f"Unexpected recording shape: {recording.shape}")
    if len(recording) < SAMPLE_RATE_HZ * 115:
        raise ValueError(f"Subject {subject} recording is too short.")
    return recording


def build_examples(
    data_directory: Path,
    subjects: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows = []
    labels = []
    example_subjects = []
    example_channels = []
    start_seconds = []
    phases = ((0, LOW_PHASE_SECONDS), (1, FATIGUE_PHASE_SECONDS))

    for subject in subjects:
        recording = load_subject(data_directory, subject)
        baseline = recording[: BASELINE_SECONDS * SAMPLE_RATE_HZ]
        baseline_mean = baseline.mean(axis=0)
        baseline_standard_deviation = np.maximum(baseline.std(axis=0), 1e-3)
        normalized = (recording - baseline_mean) / baseline_standard_deviation
        normalized = np.clip(normalized, -12.0, 12.0)

        for label, (phase_start, phase_stop) in phases:
            first = phase_start * SAMPLE_RATE_HZ
            last = min(
                phase_stop * SAMPLE_RATE_HZ - WINDOW_SAMPLES,
                len(recording) - WINDOW_SAMPLES,
            )
            for start in range(first, last + 1, WINDOW_STRIDE):
                for channel_index in range(8):
                    windows.append(
                        normalized[
                            start : start + WINDOW_SAMPLES,
                            channel_index,
                        ]
                    )
                    labels.append(label)
                    example_subjects.append(subject)
                    example_channels.append(channel_index + 1)
                    start_seconds.append(start / SAMPLE_RATE_HZ)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(example_subjects, dtype=np.int64),
        np.asarray(example_channels, dtype=np.int64),
        np.asarray(start_seconds, dtype=np.float32),
    )


def make_loader(
    windows: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(windows[:, None, :]),
        torch.from_numpy(labels),
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for batch_windows, batch_labels in loader:
            logits.append(model(batch_windows))
            labels.append(batch_labels)
    return torch.cat(logits), torch.cat(labels)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean())


def binary_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    fatigue_probabilities: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return confusion matrix, macro F1, and ROC AUC without sklearn."""

    matrix = np.zeros((2, 2), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    f1_values = []
    for class_index in range(2):
        true_positive = matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - true_positive
        false_negative = matrix[class_index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    positive_scores = fatigue_probabilities[targets == 1]
    negative_scores = fatigue_probabilities[targets == 0]
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    roc_auc = float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )
    return matrix, float(np.mean(f1_values)), roc_auc


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_temperature = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [log_temperature],
        lr=0.05,
        max_iter=80,
        line_search_fn="strong_wolfe",
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


def train_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
) -> tuple[ForearmFatigueCNN, list[dict]]:
    model = ForearmFatigueCNN()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=0.5,
        patience=3,
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_state = None
    best_accuracy = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        example_count = 0
        for windows, labels in train_loader:
            optimiser.zero_grad()
            logits = model(windows)
            loss = loss_function(logits, labels)
            loss.backward()
            optimiser.step()
            running_loss += float(loss.detach()) * len(labels)
            example_count += len(labels)

        validation_logits, validation_labels = collect_logits(
            model,
            validation_loader,
        )
        validation_accuracy = accuracy_from_logits(
            validation_logits,
            validation_labels,
        )
        scheduler.step(validation_accuracy)
        summary = {
            "epoch": epoch,
            "training_loss": running_loss / example_count,
            "validation_accuracy": validation_accuracy,
        }
        history.append(summary)
        print(json.dumps(summary))
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def select_verification_examples(
    windows: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    channels: np.ndarray,
    start_seconds: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = probabilities.argmax(axis=1)
    selected = []
    for class_index in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(
            (labels == class_index) & (predictions == class_index)
        )
        if not len(candidates):
            candidates = np.flatnonzero(labels == class_index)
        confidence = probabilities[candidates, predictions[candidates]]
        selected.append(candidates[np.argmin(np.abs(confidence - 0.80))])
    return (
        windows[selected],
        labels[selected],
        subjects[selected],
        channels[selected],
        start_seconds[selected],
    )


def main() -> None:
    args = parse_args()
    set_reproducible_seed()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    all_subjects = list(range(1, 16))
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(all_subjects).tolist()
    training_subjects = sorted(int(value) for value in shuffled[:10])
    validation_subjects = sorted(int(value) for value in shuffled[10:12])
    test_subjects = sorted(int(value) for value in shuffled[12:])

    train_data = build_examples(args.data_directory, training_subjects)
    validation_data = build_examples(args.data_directory, validation_subjects)
    test_data = build_examples(args.data_directory, test_subjects)
    train_loader = make_loader(
        train_data[0], train_data[1], args.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        validation_data[0], validation_data[1], args.batch_size, shuffle=False
    )
    test_loader = make_loader(
        test_data[0], test_data[1], args.batch_size, shuffle=False
    )

    model, history = train_model(
        train_loader,
        validation_loader,
        args.epochs,
    )
    validation_logits, validation_targets = collect_logits(
        model,
        validation_loader,
    )
    temperature = fit_temperature(validation_logits, validation_targets)
    test_logits, test_targets = collect_logits(model, test_loader)
    probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    targets = test_targets.numpy()
    matrix, macro_f1, roc_auc = binary_metrics(
        targets,
        predictions,
        probabilities[:, 1],
    )
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    metrics = {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "per_class_recall": {
            name: float(recalls[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(targets)),
        "held_out_subject_count": len(test_subjects),
        "test_subjects": test_subjects,
    }

    checkpoint = {
        "model_version": "upper-limb-fatigue-phase-real-v1",
        "model_type": "single-channel-1d-cnn",
        "model_state_dict": model.state_dict(),
        "training_data_kind": "real_public_semg",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "label_kind": "protocol_phase_proxy",
        "muscle_group": "Upper limb",
        "exercise": "Sustained 6 kg elbow-flexion hold",
        "class_names": list(CLASS_NAMES),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE_HZ,
        "normalization": "per-participant/channel first-10-second baseline",
        "temperature": temperature,
        "source_dataset": "Multi-channel Surface EMG Dataset for Fatigue analysis",
        "source_doi": SOURCE_DOI,
        "source_url": SOURCE_URL,
        "source_license": "CC BY 4.0",
        "phase_definition": {
            "low_fatigue_seconds": list(LOW_PHASE_SECONDS),
            "fatigue_like_seconds": list(FATIGUE_PHASE_SECONDS),
        },
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "test_metrics": metrics,
        "training_history": history,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)

    verification = select_verification_examples(
        test_data[0],
        test_data[1],
        test_data[2],
        test_data[3],
        test_data[4],
        probabilities,
    )
    np.savez_compressed(
        args.samples_file,
        windows=verification[0],
        labels=verification[1],
        subjects=verification[2],
        channels=verification[3],
        start_seconds=verification[4],
    )
    print(json.dumps({
        "model_file": str(args.model_file),
        "samples_file": str(args.samples_file),
        "temperature": temperature,
        "test_metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

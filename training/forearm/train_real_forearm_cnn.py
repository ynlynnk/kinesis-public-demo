"""Train a participant-separated forearm movement CNN on real public sEMG.

Dataset:
    Multi-Channel Electromyography Signal Acquisition of Forearm
    DOI: 10.17632/p77jn92bzg.1
    Licence: CC BY 4.0

The public dataset has eight Myo-armband sEMG channels sampled at 200 Hz. To
match the planned one-electrode KINESIS input, every electrode channel is used
as an independent one-channel training example. The classifier predicts only
Neutral, Pronation, or Supination; it does not predict fatigue or clinical
movement correctness.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forearm_movement_runtime import (  # noqa: E402
    CLASS_NAMES,
    ForearmMovementCNN,
)


SEED = 20260929
SAMPLE_RATE_HZ = 200
WINDOW_SAMPLES = 256
REPETITION_START_SECONDS = (1.0, 4.0, 7.0, 10.0, 13.0)
WINDOW_OFFSET_SECONDS = 0.25
TRAIN_JITTER_SECONDS = (-0.15, 0.0, 0.15)
EVALUATION_JITTER_SECONDS = (0.0,)
SOURCE_DOI = "10.17632/p77jn92bzg.1"
SOURCE_URL = "https://data.mendeley.com/datasets/p77jn92bzg/1"


def parse_args() -> argparse.Namespace:
    source_directory = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "forearm"
        / "mendeley_p77jn92bzg_v1"
        / "Multi-Channel Electromyography Signal Acquisition of Forearm"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=Path,
        default=source_directory / "signal_dataset.dat",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "models"
            / "kinesis_forearm_movement_cnn.pt"
        ),
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "models"
            / "kinesis_forearm_verification_samples.npz"
        ),
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def set_reproducible_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_source_data(data_file: Path) -> pd.DataFrame:
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    names = [f"channel_{index}" for index in range(1, 9)] + [
        "movement",
        "subject",
    ]
    dtypes = {name: np.float32 for name in names[:8]}
    dtypes.update({"movement": np.int16, "subject": np.int16})
    frame = pd.read_csv(
        data_file,
        header=None,
        names=names,
        dtype=dtypes,
    )
    frame = frame[frame["movement"].between(1, 6)].copy()

    expected_subjects = 50
    expected_rows_per_recording = 3200
    if frame["subject"].nunique() != expected_subjects:
        raise ValueError("The source does not contain the expected 50 subjects.")
    group_sizes = frame.groupby(["subject", "movement"]).size()
    if not group_sizes.eq(expected_rows_per_recording).all():
        raise ValueError("A subject/movement recording has an unexpected length.")
    return frame


def movement_to_class(raw_movement: int) -> int:
    return (raw_movement - 1) // 2


def build_examples(
    frame: pd.DataFrame,
    subjects: list[int],
    jitters: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_names = [f"channel_{index}" for index in range(1, 9)]
    windows = []
    labels = []
    example_subjects = []
    example_channels = []

    for subject in subjects:
        subject_frame = frame[frame["subject"] == subject]
        recordings = {
            movement: subject_frame[
                subject_frame["movement"] == movement
            ][channel_names].to_numpy(dtype=np.float32)
            for movement in range(1, 7)
        }
        for movement, recording in recordings.items():
            neutral_movement = 1 if movement % 2 else 2
            baseline = recordings[neutral_movement]
            baseline_mean = baseline.mean(axis=0)
            baseline_standard_deviation = baseline.std(axis=0)
            baseline_standard_deviation = np.maximum(
                baseline_standard_deviation,
                1e-3,
            )

            for channel_index in range(8):
                for repetition_start in REPETITION_START_SECONDS:
                    for jitter in jitters:
                        start_seconds = (
                            repetition_start
                            + WINDOW_OFFSET_SECONDS
                            + jitter
                        )
                        start = int(round(start_seconds * SAMPLE_RATE_HZ))
                        stop = start + WINDOW_SAMPLES
                        window = recording[start:stop, channel_index]
                        if len(window) != WINDOW_SAMPLES:
                            raise ValueError("A configured window exceeds its recording.")
                        normalized = (
                            window - baseline_mean[channel_index]
                        ) / baseline_standard_deviation[channel_index]
                        windows.append(np.clip(normalized, -12.0, 12.0))
                        labels.append(movement_to_class(movement))
                        example_subjects.append(subject)
                        example_channels.append(channel_index + 1)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(example_subjects, dtype=np.int64),
        np.asarray(example_channels, dtype=np.int64),
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
) -> tuple[ForearmMovementCNN, list[dict]]:
    model = ForearmMovementCNN()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=0.5,
        patience=3,
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.04)
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
        epoch_summary = {
            "epoch": epoch,
            "training_loss": running_loss / example_count,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

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
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = probabilities.argmax(axis=1)
    selected = []
    for class_index in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(
            (labels == class_index) & (predictions == class_index)
        )
        if not len(candidates):
            candidates = np.flatnonzero(labels == class_index)
        confidence = probabilities[candidates, predictions[candidates]]
        target = 0.80
        selected.append(candidates[np.argmin(np.abs(confidence - target))])
    return (
        windows[selected],
        labels[selected],
        subjects[selected],
        channels[selected],
    )


def main() -> None:
    args = parse_args()
    set_reproducible_seed()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    frame = load_source_data(args.data_file)
    rng = np.random.default_rng(SEED)
    subjects = rng.permutation(frame["subject"].unique()).tolist()
    training_subjects = sorted(int(value) for value in subjects[:35])
    validation_subjects = sorted(int(value) for value in subjects[35:42])
    test_subjects = sorted(int(value) for value in subjects[42:])

    train_windows, train_labels, _, _ = build_examples(
        frame,
        training_subjects,
        TRAIN_JITTER_SECONDS,
    )
    validation_windows, validation_labels, _, _ = build_examples(
        frame,
        validation_subjects,
        EVALUATION_JITTER_SECONDS,
    )
    test_windows, test_labels, test_subject_ids, test_channels = build_examples(
        frame,
        test_subjects,
        EVALUATION_JITTER_SECONDS,
    )

    train_loader = make_loader(
        train_windows,
        train_labels,
        args.batch_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_windows,
        validation_labels,
        args.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        test_windows,
        test_labels,
        args.batch_size,
        shuffle=False,
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
    test_target_array = test_targets.numpy()
    matrix = confusion_matrix(
        test_target_array,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )
    per_class_recall = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    test_metrics = {
        "accuracy": float(np.mean(predictions == test_target_array)),
        "macro_f1": float(
            f1_score(test_target_array, predictions, average="macro")
        ),
        "per_class_recall": {
            name: float(per_class_recall[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(test_target_array)),
        "held_out_subject_count": len(test_subjects),
        "test_subjects": test_subjects,
    }

    checkpoint = {
        "model_version": "forearm-movement-real-v1",
        "model_type": "single-channel-1d-cnn",
        "model_state_dict": model.state_dict(),
        "training_data_kind": "real_public_semg",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "muscle_group": "Forearm",
        "exercise": "Forearm movement recognition",
        "class_names": list(CLASS_NAMES),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE_HZ,
        "normalization": "per-subject/channel neutral baseline",
        "temperature": temperature,
        "source_dataset": (
            "Multi-Channel Electromyography Signal Acquisition of Forearm"
        ),
        "source_doi": SOURCE_DOI,
        "source_url": SOURCE_URL,
        "source_license": "CC BY 4.0",
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "test_metrics": test_metrics,
        "training_history": history,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)

    verification = select_verification_examples(
        test_windows,
        test_labels,
        test_subject_ids,
        test_channels,
        probabilities,
    )
    np.savez_compressed(
        args.samples_file,
        windows=verification[0],
        labels=verification[1],
        subjects=verification[2],
        channels=verification[3],
    )

    print(json.dumps({
        "model_file": str(args.model_file),
        "samples_file": str(args.samples_file),
        "temperature": temperature,
        "test_metrics": test_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

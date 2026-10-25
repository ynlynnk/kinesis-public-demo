"""Train a participant-separated forearm movement CNN on 40 real people.

Dataset:
    Dataset for multi-channel surface electromyography (sEMG) signals of
    hand gestures, DOI 10.17632/ckwc76xr2z.2, CC BY 4.0.

Only the documented neutral, supination and pronation intervals are used.
The four electrodes are treated as independent one-electrode examples so the
checkpoint remains compatible with the KINESIS single-sEMG-channel contract.
Participants, rather than windows, are separated across train/validation/test.
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
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forearm_movement_runtime import CLASS_NAMES, ForearmMovementCNN  # noqa: E402


SEED = 20261023
SOURCE_DOI = "10.17632/ckwc76xr2z.2"
SOURCE_URL = "https://data.mendeley.com/datasets/ckwc76xr2z/2"
SOURCE_SAMPLE_RATE_HZ = 2000
TARGET_SAMPLE_RATE_HZ = 200
TARGET_WINDOW_SAMPLES = 256
TARGET_WINDOW_SECONDS = TARGET_WINDOW_SAMPLES / TARGET_SAMPLE_RATE_HZ
CYCLE_SECONDS = 104
BETWEEN_CYCLE_REST_SECONDS = 30
CYCLE_START_SECONDS = tuple(
    index * (CYCLE_SECONDS + BETWEEN_CYCLE_REST_SECONDS)
    for index in range(5)
)
GESTURES = (
    ("Neutral", 0, 4.0),
    ("Supination", 2, 84.0),
    ("Pronation", 1, 94.0),
)
REST_INTERVALS = (
    (0.0, 4.0),
    (10.0, 14.0),
    (20.0, 24.0),
    (30.0, 34.0),
    (40.0, 44.0),
    (50.0, 54.0),
    (60.0, 64.0),
    (80.0, 84.0),
    (100.0, 104.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "forearm_movement_mendeley_ckwc76xr2z_v2"
            / "filtered_mat"
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
            / "kinesis_forearm_movement_ckwc76xr2z_cnn_v1.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def resample_window(values: np.ndarray) -> np.ndarray:
    old_axis = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    new_axis = np.linspace(
        0.0,
        1.0,
        TARGET_WINDOW_SAMPLES,
        dtype=np.float64,
    )
    return np.interp(new_axis, old_axis, values).astype(np.float32)


def load_participant(path: Path, expected_subject: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    bundle = loadmat(path)
    sample_rate = int(np.asarray(bundle["fs"]).reshape(-1)[0])
    subject = int(np.asarray(bundle["iD"]).reshape(-1)[0])
    recording = np.asarray(bundle["data"], dtype=np.float32)
    if sample_rate != SOURCE_SAMPLE_RATE_HZ:
        raise ValueError(f"Unexpected sample rate in {path}: {sample_rate}")
    if subject != expected_subject:
        raise ValueError(f"Unexpected subject ID in {path}: {subject}")
    if recording.shape != (640 * SOURCE_SAMPLE_RATE_HZ, 4):
        raise ValueError(f"Unexpected recording shape in {path}: {recording.shape}")
    if not np.isfinite(recording).all():
        raise ValueError(f"Non-finite source value in {path}")
    return recording


def participant_baseline(recording: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rest_segments = []
    for cycle_start in CYCLE_START_SECONDS:
        for start_seconds, stop_seconds in REST_INTERVALS:
            first = int(round((cycle_start + start_seconds) * SOURCE_SAMPLE_RATE_HZ))
            last = int(round((cycle_start + stop_seconds) * SOURCE_SAMPLE_RATE_HZ))
            rest_segments.append(recording[first:last])
    rest = np.concatenate(rest_segments, axis=0)
    mean = rest.mean(axis=0)
    standard_deviation = np.maximum(rest.std(axis=0), 1e-5)
    return mean, standard_deviation


def build_examples(
    data_directory: Path,
    subjects: list[int],
    offsets_seconds: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows, labels, participant_ids, channel_ids, cycle_ids = [], [], [], [], []
    source_window_samples = int(round(TARGET_WINDOW_SECONDS * SOURCE_SAMPLE_RATE_HZ))

    for subject in subjects:
        recording = load_participant(
            data_directory / f"{subject}_filtered.mat",
            subject,
        )
        baseline_mean, baseline_std = participant_baseline(recording)
        for cycle_index, cycle_start in enumerate(CYCLE_START_SECONDS, start=1):
            for _, class_index, gesture_start in GESTURES:
                gesture_centre = cycle_start + gesture_start + 3.0
                for offset_seconds in offsets_seconds:
                    centre = gesture_centre + offset_seconds
                    first = int(
                        round(
                            centre * SOURCE_SAMPLE_RATE_HZ
                            - source_window_samples / 2
                        )
                    )
                    last = first + source_window_samples
                    if first < 0 or last > len(recording):
                        raise ValueError("A documented gesture window exceeds its file.")
                    segment = recording[first:last]
                    normalized = (segment - baseline_mean) / baseline_std
                    normalized = np.clip(normalized, -12.0, 12.0)
                    for channel_index in range(4):
                        windows.append(resample_window(normalized[:, channel_index]))
                        labels.append(class_index)
                        participant_ids.append(subject)
                        channel_ids.append(channel_index + 1)
                        cycle_ids.append(cycle_index)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(participant_ids, dtype=np.int64),
        np.asarray(channel_ids, dtype=np.int64),
        np.asarray(cycle_ids, dtype=np.int64),
    )


def make_loader(
    windows: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
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


def collect_logits(model: nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for windows, labels in loader:
            logits.append(model(windows))
            targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
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


def train_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
) -> tuple[ForearmMovementCNN, list[dict]]:
    model = ForearmMovementCNN()
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=2e-5,
    )
    best_state = None
    best_f1 = -1.0
    patience_remaining = 12
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for windows, labels in train_loader:
            optimiser.zero_grad()
            if epoch > 1:
                windows = windows * (0.92 + 0.16 * torch.rand(len(windows), 1, 1))
                windows = windows + 0.012 * torch.randn_like(windows)
            loss = loss_function(model(windows), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimiser.step()
            running_loss += float(loss.detach()) * len(labels)
            count += len(labels)

        validation_logits, validation_targets = collect_logits(
            model,
            validation_loader,
        )
        result = classification_metrics(
            validation_targets.numpy(),
            validation_logits.argmax(dim=1).numpy(),
        )
        scheduler.step(result["macro_f1"])
        summary = {
            "epoch": epoch,
            "training_loss": running_loss / count,
            "validation_accuracy": result["accuracy"],
            "validation_macro_f1": result["macro_f1"],
            "learning_rate": optimiser.param_groups[0]["lr"],
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)
        if result["macro_f1"] > best_f1 + 1e-5:
            best_f1 = result["macro_f1"]
            best_state = deepcopy(model.state_dict())
            patience_remaining = 12
        else:
            patience_remaining -= 1
            if patience_remaining == 0:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def main() -> None:
    args = parse_args()
    set_seed()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    subject_rng = np.random.default_rng(SEED)
    subjects = subject_rng.permutation(np.arange(1, 41)).tolist()
    training_subjects = sorted(int(value) for value in subjects[:28])
    validation_subjects = sorted(int(value) for value in subjects[28:34])
    test_subjects = sorted(int(value) for value in subjects[34:])

    training = build_examples(
        args.data_directory,
        training_subjects,
        (-1.5, -0.75, 0.0, 0.75, 1.5),
    )
    validation = build_examples(
        args.data_directory,
        validation_subjects,
        (-1.5, 0.0, 1.5),
    )
    test = build_examples(
        args.data_directory,
        test_subjects,
        (-1.5, 0.0, 1.5),
    )
    train_loader = make_loader(training[0], training[1], args.batch_size, True)
    validation_loader = make_loader(
        validation[0], validation[1], args.batch_size, False
    )
    test_loader = make_loader(test[0], test[1], args.batch_size, False)
    model, history = train_model(train_loader, validation_loader, args.epochs)

    logits, targets = collect_logits(model, test_loader)
    predictions = logits.argmax(dim=1).numpy()
    test_metrics = classification_metrics(targets.numpy(), predictions)
    test_metrics.update(
        {
            "held_out_subjects": test_subjects,
            "held_out_subject_count": len(test_subjects),
            "evaluation_kind": "participant-held-out-single-electrode-windows",
        }
    )

    checkpoint = {
        "model_version": "forearm-movement-ckwc76xr2z-real-v1-candidate",
        "model_type": "single-channel-1d-cnn",
        "model_state_dict": model.state_dict(),
        "training_data_kind": "real_public_semg",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "candidate_only": True,
        "muscle_group": "Forearm",
        "class_names": list(CLASS_NAMES),
        "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "window_samples": TARGET_WINDOW_SAMPLES,
        "window_seconds": TARGET_WINDOW_SECONDS,
        "normalization": "per-participant/channel documented rest baseline",
        "source_dataset": (
            "Dataset for multi-channel surface electromyography (sEMG) "
            "signals of hand gestures"
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
    print(
        json.dumps(
            {
                "model_file": str(args.model_file),
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

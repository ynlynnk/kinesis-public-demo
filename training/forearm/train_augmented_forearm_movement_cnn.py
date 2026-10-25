"""Train an augmented participant-separated forearm movement CNN.

The candidate combines two real-human public sEMG datasets while keeping
participants separated between training, validation, and testing:

* Mendeley p77jn92bzg: 50 people, Myo armband, 200 Hz.
* Zenodo 4039550: 10 people, eight channels, 1000 Hz, three electrode
  placements, ten 5-second repetitions followed by 3-second rest periods.

The original Mendeley held-out subjects are unchanged so the candidate can be
compared directly with the deployed checkpoint.  The Zenodo test participants
provide a second, independent held-out result under electrode displacement.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forearm_movement_runtime import CLASS_NAMES, ForearmMovementCNN  # noqa: E402
from training.forearm import train_real_forearm_cnn as original  # noqa: E402


SEED = 20261023
ZENODO_SAMPLE_RATE_HZ = 1000
TARGET_SAMPLE_RATE_HZ = original.SAMPLE_RATE_HZ
TARGET_WINDOW_SAMPLES = original.WINDOW_SAMPLES
TARGET_WINDOW_SECONDS = TARGET_WINDOW_SAMPLES / TARGET_SAMPLE_RATE_HZ
REPETITION_SECONDS = 8.0
MOVEMENT_SECONDS = 5.0
REST_SECONDS = 3.0
ZENODO_DOI = "10.5281/zenodo.4039550"
ZENODO_URL = "https://zenodo.org/records/4039550"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "forearm"
            / "mendeley_p77jn92bzg_v1"
            / "Multi-Channel Electromyography Signal Acquisition of Forearm"
            / "signal_dataset.dat"
        ),
    )
    parser.add_argument(
        "--added-directory",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "forearm_movement_zenodo_4039550"
            / "extracted"
            / "EMG dataset"
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
            / "kinesis_forearm_movement_augmented_cnn_v2.pt"
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


def load_zenodo_recording(path: Path) -> np.ndarray:
    # A very small number of source rows are incomplete.  The public archive
    # otherwise contains eight whitespace-separated channels, so skip only
    # malformed rows rather than discarding an entire participant recording.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recording = np.genfromtxt(
            path,
            dtype=np.float32,
            usecols=range(8),
            invalid_raise=False,
        )
    recording = recording[np.isfinite(recording).all(axis=1)]
    if recording.ndim != 2 or recording.shape[1] != 8:
        raise ValueError(f"Unexpected recording shape in {path}: {recording.shape}")
    if len(recording) < 60_000:
        raise ValueError(
            f"Recording is too short for ten protocol cycles in {path}: "
            f"{len(recording)} samples"
        )
    return recording


def estimate_protocol_alignment(recording: np.ndarray) -> tuple[float, float, dict]:
    """Locate the 5 s active part within each 8 s protocol cycle.

    The archive documents ten 5-second movements separated by 3-second rests,
    but the stored row count varies substantially between recordings.  We
    therefore infer the effective samples per cycle from each file and search
    only the documented 0-3 second phase range.  The selected phase maximises
    the median movement/rest RMS contrast on the most informative electrode.
    This avoids inventing labels from arbitrary fixed row offsets.
    """

    centred = recording - np.median(recording, axis=0, keepdims=True)
    cumulative_energy = np.vstack(
        [np.zeros((1, recording.shape[1]), dtype=np.float64),
         np.cumsum(np.square(centred, dtype=np.float64), axis=0)]
    )
    cycle_samples = len(recording) / 10.0
    movement_samples = cycle_samples * MOVEMENT_SECONDS / REPETITION_SECONDS
    best = None

    for phase_fraction in np.linspace(
        0.0,
        REST_SECONDS / REPETITION_SECONDS,
        49,
    ):
        repetition_contrasts = []
        for repetition in range(10):
            block_first = int(round(repetition * cycle_samples))
            block_last = min(
                len(recording),
                int(round((repetition + 1) * cycle_samples)),
            )
            movement_first = min(
                block_last,
                int(round(block_first + phase_fraction * cycle_samples)),
            )
            movement_last = min(
                block_last,
                int(round(movement_first + movement_samples)),
            )
            active_count = max(1, movement_last - movement_first)
            active_energy = (
                cumulative_energy[movement_last]
                - cumulative_energy[movement_first]
            ) / active_count

            rest_energy_sum = np.zeros(recording.shape[1], dtype=np.float64)
            rest_count = 0
            if movement_first > block_first:
                rest_energy_sum += (
                    cumulative_energy[movement_first]
                    - cumulative_energy[block_first]
                )
                rest_count += movement_first - block_first
            if block_last > movement_last:
                rest_energy_sum += (
                    cumulative_energy[block_last]
                    - cumulative_energy[movement_last]
                )
                rest_count += block_last - movement_last
            rest_energy = rest_energy_sum / max(1, rest_count)
            repetition_contrasts.append(
                0.5 * np.log((active_energy + 1e-12) / (rest_energy + 1e-12))
            )

        contrasts = np.asarray(repetition_contrasts)
        channel_contrasts = np.median(contrasts, axis=0)
        channel_index = int(np.argmax(channel_contrasts))
        score = float(channel_contrasts[channel_index])
        if best is None or score > best[0]:
            best = (score, phase_fraction, channel_index, contrasts)

    if best is None:
        raise RuntimeError("Protocol alignment did not produce a candidate.")
    score, phase_fraction, channel_index, contrasts = best
    audit = {
        "phase_seconds": float(phase_fraction * REPETITION_SECONDS),
        "effective_sample_rate_hz": float(cycle_samples / REPETITION_SECONDS),
        "best_channel": channel_index + 1,
        "median_log_rms_contrast": score,
        "positive_contrast_repetitions": int(
            np.sum(contrasts[:, channel_index] > 0.0)
        ),
    }
    return cycle_samples, phase_fraction * cycle_samples, audit


def central_window_start(first: int, last: int, window_samples: int) -> int:
    if last - first < window_samples:
        raise ValueError("A protocol section is shorter than the model window.")
    return first + (last - first - window_samples) // 2


def build_zenodo_examples(
    data_directory: Path,
    subjects: list[int],
    jitters: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    participant_ids: list[int] = []
    channel_ids: list[int] = []
    placement_ids: list[int] = []
    alignment_audits = []

    for subject in subjects:
        subject_directory = data_directory / f"ID{subject}"
        for movement_code, class_index in (("PR", 1), ("SU", 2)):
            for placement in (1, 2, 3):
                path = subject_directory / f"ID{subject}_{movement_code}_P{placement}.txt"
                if not path.is_file():
                    raise FileNotFoundError(f"Dataset file not found: {path}")
                recording = load_zenodo_recording(path)
                cycle_samples, phase_samples, audit = estimate_protocol_alignment(
                    recording
                )
                audit["file"] = str(path)
                alignment_audits.append(audit)
                source_window_samples = int(
                    round(TARGET_WINDOW_SECONDS / REPETITION_SECONDS * cycle_samples)
                )

                protocol_sections = []
                rest_segments = []
                for repetition in range(10):
                    block_first = int(round(repetition * cycle_samples))
                    block_last = min(
                        len(recording),
                        int(round((repetition + 1) * cycle_samples)),
                    )
                    movement_first = min(
                        block_last,
                        int(round(block_first + phase_samples)),
                    )
                    movement_last = min(
                        block_last,
                        int(
                            round(
                                movement_first
                                + cycle_samples
                                * MOVEMENT_SECONDS
                                / REPETITION_SECONDS
                            )
                        ),
                    )
                    before_rest = (block_first, movement_first)
                    after_rest = (movement_last, block_last)
                    rest_first, rest_last = max(
                        (before_rest, after_rest),
                        key=lambda section: section[1] - section[0],
                    )
                    protocol_sections.append(
                        (
                            movement_first,
                            movement_last,
                            rest_first,
                            rest_last,
                        )
                    )
                    if rest_last > rest_first:
                        rest_segments.append(recording[rest_first:rest_last])
                rest = np.concatenate(rest_segments, axis=0)
                baseline_mean = rest.mean(axis=0)
                baseline_std = np.maximum(rest.std(axis=0), 1e-4)

                for movement_first, movement_last, rest_first, rest_last in protocol_sections:
                    movement_start = central_window_start(
                        movement_first,
                        movement_last,
                        source_window_samples,
                    )
                    rest_start = central_window_start(
                        rest_first,
                        rest_last,
                        source_window_samples,
                    )
                    for jitter_seconds in jitters:
                        effective_rate_hz = cycle_samples / REPETITION_SECONDS
                        jitter = int(round(jitter_seconds * effective_rate_hz))
                        for label, start in ((class_index, movement_start), (0, rest_start)):
                            first = start + jitter
                            last = first + source_window_samples
                            if first < 0 or last > len(recording):
                                continue
                            segment = recording[first:last]
                            normalized = (segment - baseline_mean) / baseline_std
                            normalized = np.clip(normalized, -12.0, 12.0)
                            for channel in range(8):
                                windows.append(resample_window(normalized[:, channel]))
                                labels.append(label)
                                participant_ids.append(subject)
                                channel_ids.append(channel + 1)
                                placement_ids.append(placement)

    if alignment_audits:
        scores = np.asarray(
            [item["median_log_rms_contrast"] for item in alignment_audits]
        )
        positive_counts = np.asarray(
            [item["positive_contrast_repetitions"] for item in alignment_audits]
        )
        print(
            json.dumps(
                {
                    "zenodo_protocol_alignment": {
                        "recording_count": len(alignment_audits),
                        "median_log_rms_contrast": float(np.median(scores)),
                        "minimum_log_rms_contrast": float(np.min(scores)),
                        "minimum_positive_repetitions": int(np.min(positive_counts)),
                    }
                }
            ),
            flush=True,
        )

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(participant_ids, dtype=np.int64),
        np.asarray(channel_ids, dtype=np.int64),
        np.asarray(placement_ids, dtype=np.int64),
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


def train_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
) -> tuple[ForearmMovementCNN, list[dict]]:
    model = ForearmMovementCNN()
    counts = torch.bincount(train_loader.dataset.tensors[1], minlength=3).float()
    weights = counts.sum() / torch.clamp(counts * len(CLASS_NAMES), min=1.0)
    loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.04)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=2e-5,
    )
    best_state = None
    best_f1 = -1.0
    patience_remaining = 10
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for windows, labels in train_loader:
            optimiser.zero_grad()
            if epoch > 1:
                windows = windows * (0.95 + 0.10 * torch.rand(len(windows), 1, 1))
                windows = windows + 0.01 * torch.randn_like(windows)
            loss = loss_function(model(windows), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimiser.step()
            running_loss += float(loss.detach()) * len(labels)
            count += len(labels)

        logits, targets = collect_logits(model, validation_loader)
        result = metrics(targets.numpy(), logits.argmax(dim=1).numpy())
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
            patience_remaining = 10
        else:
            patience_remaining -= 1
            if patience_remaining == 0:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def evaluate(model: nn.Module, data: tuple[np.ndarray, ...], batch_size: int) -> dict:
    loader = make_loader(data[0], data[1], batch_size, shuffle=False)
    logits, targets = collect_logits(model, loader)
    return metrics(targets.numpy(), logits.argmax(dim=1).numpy())


def main() -> None:
    args = parse_args()
    set_seed()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    frame = original.load_source_data(args.original_file)
    original_rng = np.random.default_rng(original.SEED)
    original_subjects = original_rng.permutation(frame["subject"].unique()).tolist()
    original_train_subjects = sorted(int(value) for value in original_subjects[:35])
    original_validation_subjects = sorted(int(value) for value in original_subjects[35:42])
    original_test_subjects = sorted(int(value) for value in original_subjects[42:])

    added_rng = np.random.default_rng(SEED)
    added_subjects = added_rng.permutation(np.arange(1, 11)).tolist()
    added_train_subjects = sorted(int(value) for value in added_subjects[:7])
    added_validation_subjects = [int(added_subjects[7])]
    added_test_subjects = sorted(int(value) for value in added_subjects[8:])

    original_train = original.build_examples(
        frame,
        original_train_subjects,
        original.TRAIN_JITTER_SECONDS,
    )
    original_validation = original.build_examples(
        frame,
        original_validation_subjects,
        original.EVALUATION_JITTER_SECONDS,
    )
    original_test = original.build_examples(
        frame,
        original_test_subjects,
        original.EVALUATION_JITTER_SECONDS,
    )
    added_train = build_zenodo_examples(
        args.added_directory,
        added_train_subjects,
        (-0.20, 0.0, 0.20),
    )
    added_validation = build_zenodo_examples(
        args.added_directory,
        added_validation_subjects,
        (0.0,),
    )
    added_test = build_zenodo_examples(
        args.added_directory,
        added_test_subjects,
        (0.0,),
    )

    train_windows = np.concatenate([original_train[0], added_train[0]])
    train_labels = np.concatenate([original_train[1], added_train[1]])
    validation_windows = np.concatenate([original_validation[0], added_validation[0]])
    validation_labels = np.concatenate([original_validation[1], added_validation[1]])

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
    model, history = train_model(train_loader, validation_loader, args.epochs)

    original_metrics = evaluate(model, original_test, args.batch_size)
    original_metrics["held_out_subjects"] = original_test_subjects
    added_metrics = evaluate(model, added_test, args.batch_size)
    added_metrics["held_out_subjects"] = added_test_subjects
    combined_metrics = evaluate(
        model,
        (
            np.concatenate([original_test[0], added_test[0]]),
            np.concatenate([original_test[1], added_test[1]]),
        ),
        args.batch_size,
    )

    checkpoint = {
        "model_version": "forearm-movement-real-augmented-v2-candidate",
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
        "window_seconds": TARGET_WINDOW_SAMPLES / TARGET_SAMPLE_RATE_HZ,
        "source_datasets": [
            {
                "name": "Multi-Channel Electromyography Signal Acquisition of Forearm",
                "doi": original.SOURCE_DOI,
                "url": original.SOURCE_URL,
                "license": "CC BY 4.0",
            },
            {
                "name": "Surface electromyogram dataset recorded from forearm for 9 hand movements and three electrode array positions",
                "doi": ZENODO_DOI,
                "url": ZENODO_URL,
                "license": "CC BY 4.0",
            },
        ],
        "training_subjects": {
            "original": original_train_subjects,
            "added": added_train_subjects,
        },
        "validation_subjects": {
            "original": original_validation_subjects,
            "added": added_validation_subjects,
        },
        "test_metrics": {
            "original_unchanged_holdout": original_metrics,
            "added_dataset_holdout": added_metrics,
            "combined": combined_metrics,
        },
        "training_history": history,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)
    print(
        json.dumps(
            {
                "model_file": str(args.model_file),
                "original_unchanged_holdout": original_metrics,
                "added_dataset_holdout": added_metrics,
                "combined": combined_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Train a participant-separated quadriceps fatigue-pattern feature CNN.

The source contains real fatigue-protocol sEMG features from 45 participants.
Only non-zero measured quadriceps sequences are retained.  Early and late
protocol windows are treated as low-fatigue and fatigue-like phase proxies;
they are not clinical fatigue-severity labels.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 20261002
FEATURE_NAMES = ("MDF", "mnIMDF", "mxIMDF", "SE", "PE")
MUSCLE_NAMES = ("RECTUSFEM", "VASTUSLAT", "VASTUSMED")
CLASS_NAMES = ("Low-fatigue phase", "Fatigue-like phase")
WINDOW_CYCLES = 64
WINDOW_STRIDE = 16
SOURCE_URL = "https://zenodo.org/records/20847204"
SOURCE_DOI = "10.5281/zenodo.20847204"


class QuadricepsFatigueFeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(len(FEATURE_NAMES), 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(96 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, len(CLASS_NAMES)),
        )

    def forward(self, values):
        return self.classifier(self.features(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "quadriceps_fatigue"
            / "zenodo_20847204"
            / "emg_table.csv"
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
            / "kinesis_quadriceps_fatigue_feature_cnn_v2.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_sequences(data_file: Path) -> dict[tuple[str, str], np.ndarray]:
    frame = pd.read_csv(data_file)
    cycle_columns = [column for column in frame if column.startswith("r_cycle_")]
    frame = frame[frame["muscle"].isin(MUSCLE_NAMES)].copy()
    sequences = {}
    for (subject, muscle), group in frame.groupby(["anonym_code", "muscle"]):
        by_feature = group.set_index("feature_name")
        if not all(name in by_feature.index for name in FEATURE_NAMES):
            continue
        raw = np.stack([
            by_feature.loc[name, cycle_columns].to_numpy(dtype=np.float32)
            for name in FEATURE_NAMES
        ])
        measured = np.any(raw != 0.0, axis=0)
        if not measured.any():
            continue
        final = int(np.flatnonzero(measured)[-1]) + 1
        sequence = raw[:, :final]
        if len(sequence.T) < WINDOW_CYCLES * 3:
            continue
        sequences[(str(subject), str(muscle))] = sequence
    return sequences


def valid_subjects(sequences) -> list[str]:
    counts = {}
    for subject, _ in sequences:
        counts[subject] = counts.get(subject, 0) + 1
    return sorted(subject for subject, count in counts.items() if count == len(MUSCLE_NAMES))


def build_examples(sequences, subjects: list[str]):
    windows, labels, subject_ids, muscle_ids = [], [], [], []
    wanted = set(subjects)
    for (subject, muscle), sequence in sorted(sequences.items()):
        if subject not in wanted:
            continue
        baseline_count = max(WINDOW_CYCLES, int(round(sequence.shape[1] * 0.10)))
        baseline = sequence[:, :baseline_count]
        mean = baseline.mean(axis=1, keepdims=True)
        standard_deviation = np.maximum(baseline.std(axis=1, keepdims=True), 1e-3)
        normalized = np.clip((sequence - mean) / standard_deviation, -10.0, 10.0)
        length = normalized.shape[1]
        regions = (
            (0, 0, int(round(length * 0.30))),
            (1, int(round(length * 0.70)), length),
        )
        for label, first, last in regions:
            for start in range(first, last - WINDOW_CYCLES + 1, WINDOW_STRIDE):
                windows.append(normalized[:, start : start + WINDOW_CYCLES])
                labels.append(label)
                subject_ids.append(subject)
                muscle_ids.append(muscle)
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subject_ids),
        np.asarray(muscle_ids),
    )


def make_loader(data, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1])),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def collect_logits(model, loader):
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for values, labels in loader:
            logits.append(model(values))
            targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def metrics(targets: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    f1_values = []
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    for index in range(2):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    positive = scores[targets == 1]
    negative = scores[targets == 0]
    comparisons = positive[:, None] - negative[None, :]
    auc = (
        np.count_nonzero(comparisons > 0)
        + 0.5 * np.count_nonzero(comparisons == 0)
    ) / comparisons.size
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "roc_auc": float(auc),
        "per_class_recall": {
            name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(targets)),
    }


def train_model(train_loader, validation_loader, epochs: int):
    model = QuadricepsFatigueFeatureCNN()
    counts = torch.bincount(train_loader.dataset.tensors[1], minlength=2).float()
    weights = counts.sum() / torch.clamp(counts * 2, min=1.0)
    loss_function = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.04)
    optimiser = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=4, min_lr=2e-5
    )
    best_state = None
    best_f1 = -1.0
    remaining = 12
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for values, labels in train_loader:
            optimiser.zero_grad()
            if epoch > 1:
                values = values + 0.015 * torch.randn_like(values)
            loss = loss_function(model(values), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimiser.step()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
        validation_logits, validation_targets = collect_logits(model, validation_loader)
        probabilities = torch.softmax(validation_logits, dim=1).numpy()
        result = metrics(
            validation_targets.numpy(), probabilities.argmax(axis=1), probabilities[:, 1]
        )
        scheduler.step(result["macro_f1"])
        summary = {
            "epoch": epoch,
            "training_loss": total_loss / count,
            "validation_accuracy": result["accuracy"],
            "validation_macro_f1": result["macro_f1"],
            "validation_roc_auc": result["roc_auc"],
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
    set_seed()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    sequences = load_sequences(args.data_file)
    subjects = valid_subjects(sequences)
    shuffled = np.random.default_rng(SEED).permutation(subjects).tolist()
    training_subjects = sorted(str(value) for value in shuffled[:24])
    validation_subjects = sorted(str(value) for value in shuffled[24:29])
    test_subjects = sorted(str(value) for value in shuffled[29:])
    if len(test_subjects) < 5:
        raise ValueError("At least 34 fully measured participants are required.")

    train_data = build_examples(sequences, training_subjects)
    validation_data = build_examples(sequences, validation_subjects)
    test_data = build_examples(sequences, test_subjects)
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
    probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
    test_metrics = metrics(
        test_targets.numpy(), probabilities.argmax(axis=1), probabilities[:, 1]
    )
    test_metrics.update({
        "held_out_subject_count": len(test_subjects),
        "test_subjects": test_subjects,
    })

    checkpoint = {
        "model_version": "quadriceps-fatigue-feature-real-v2-candidate",
        "model_type": "five-feature-temporal-1d-cnn",
        "model_state_dict": model.state_dict(),
        "training_data_kind": "real_public_semg_features",
        "trained_on_labelled_data": True,
        "validated_on_public_real_data": True,
        "validated_on_kinesis_data": False,
        "label_kind": "early-versus-late-protocol-phase-proxy",
        "muscle_group": "Quadriceps",
        "class_names": list(CLASS_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "source_muscles": list(MUSCLE_NAMES),
        "window_cycles": WINDOW_CYCLES,
        "phase_definition": {"early_fraction": 0.30, "late_fraction": 0.30},
        "source_dataset": (
            "EMG and IMU data for Complexity analysis of sEMG and IMU-based "
            "kinematics for muscular fatigue monitoring in dynamic sports settings"
        ),
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "source_license": "CC BY 4.0",
        "training_subjects": training_subjects,
        "validation_subjects": validation_subjects,
        "test_metrics": test_metrics,
        "training_history": history,
        "temperature": temperature,
    }
    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_file)
    print(json.dumps({
        "model_file": str(args.model_file),
        "valid_subject_count": len(subjects),
        "temperature": temperature,
        "test_metrics": test_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

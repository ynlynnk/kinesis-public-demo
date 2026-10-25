"""Build varied Model 2 session records from the current Model 1 CNNs.

The generated records are composite demonstration sessions, but every fatigue
and movement value stored in ``model1_output.json`` is produced by the shipped
real-public-data CNN checkpoints.  This replaces the older records that were
created by the retired synthetic preview classifier.

The script intentionally keeps raw class probabilities as technical evidence
and derives the user-facing session percentages from counts of thresholded
windows.  It never edits a probability to make a session look more realistic.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import numpy as np
import torch


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PUBLIC_ROOT.parent
for path in (
    PUBLIC_ROOT,
    PUBLIC_ROOT / "training" / "quadriceps",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forearm_fatigue_runtime import (  # noqa: E402
    load_fatigue_verification_samples,
    predict_normalized_fatigue_window,
)
from forearm_movement_runtime import (  # noqa: E402
    CLASS_NAMES as FOREARM_MOVEMENT_CLASSES,
    load_forearm_verification_samples,
    predict_normalized_forearm_window,
)
from quadriceps_runtime import (  # noqa: E402
    FATIGUE_CLASS_NAMES as QUADRICEPS_FATIGUE_CLASSES,
    MOVEMENT_CLASS_NAMES as QUADRICEPS_MOVEMENT_CLASSES,
    load_quadriceps_fatigue_model,
    load_quadriceps_movement_model,
)
from train_real_quadriceps_fatigue_cnn import (  # noqa: E402
    build_participant_examples,
)
from train_real_quadriceps_movement_cnn import (  # noqa: E402
    build_examples as build_quadriceps_movement_examples,
)


WINDOW_COUNT = 15
WINDOW_SECONDS = 2.56
MODEL_RATE_HZ = 200
IMU_RATE_HZ = 50
FATIGUE_THRESHOLD = 0.5
MOVEMENT_MISMATCH_THRESHOLD = 0.5
SESSION_MODEL_VERSIONS = {
    "Forearm": "kinesis-forearm-session-real-cnn-v1",
    "Quadriceps": "kinesis-quadriceps-session-real-cnn-v1",
}
SESSION_PLANS = {
    "P001": ("good", "good", "caution"),
    "P002": ("caution", "good", "attention"),
    "P003": ("good", "caution", "good"),
    "P004": ("caution", "attention", "caution"),
    "P005": ("attention", "caution", "good"),
    "P006": ("good", "caution", "attention"),
}
MUSCLE_GROUPS = {
    "P001": "Forearm",
    "P002": "Forearm",
    "P003": "Forearm",
    "P004": "Quadriceps",
    "P005": "Quadriceps",
    "P006": "Quadriceps",
}
MISMATCH_COUNTS = {"good": 2, "caution": 4, "attention": 7}
FATIGUE_TARGETS = {
    "good": np.linspace(0.18, 0.43, WINDOW_COUNT),
    "caution": np.concatenate(
        [np.linspace(0.22, 0.45, 9), np.linspace(0.52, 0.72, 6)]
    ),
    "attention": np.concatenate(
        [np.linspace(0.28, 0.47, 4), np.linspace(0.56, 0.84, 11)]
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-root",
        type=Path,
        default=Path(
            os.getenv(
                "KINESIS_BACKEND_ROOT",
                str(PROJECT_ROOT / "KINESIS_AI_DEMO"),
            )
        ),
    )
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Do not create a recoverable copy of the replaced session files.",
    )
    return parser.parse_args()


def _predict_quadriceps_movement_arrays(
    windows: np.ndarray,
    imu_windows: np.ndarray,
) -> np.ndarray:
    model_file = (
        PUBLIC_ROOT
        / "artifacts"
        / "models"
        / "kinesis_quadriceps_movement_cnn.pt"
    )
    model, checkpoint = load_quadriceps_movement_model(str(model_file))
    with torch.no_grad():
        logits = model(
            torch.from_numpy(windows[:, None, :]),
            torch.from_numpy(imu_windows[:, None, :]),
        ) / max(float(checkpoint.get("temperature", 1.0)), 1e-3)
        return torch.softmax(logits, dim=1).cpu().numpy()


def _predict_quadriceps_fatigue_arrays(windows: np.ndarray) -> np.ndarray:
    model_file = (
        PUBLIC_ROOT
        / "artifacts"
        / "models"
        / "kinesis_quadriceps_fatigue_cnn.pt"
    )
    model, checkpoint = load_quadriceps_fatigue_model(str(model_file))
    with torch.no_grad():
        logits = model(torch.from_numpy(windows[:, None, :])) / max(
            float(checkpoint.get("temperature", 1.0)), 1e-3
        )
        return torch.softmax(logits, dim=1).cpu().numpy()


def _nearest_index(values: np.ndarray, target: float, used: set[int]) -> int:
    order = np.argsort(np.abs(values - float(target)))
    for index in order:
        candidate = int(index)
        if candidate not in used:
            used.add(candidate)
            return candidate
    return int(order[0])


def _mismatch_mask(level: str, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(WINDOW_COUNT, dtype=bool)
    candidates = np.arange(2, WINDOW_COUNT - 1)
    selected = rng.choice(
        candidates,
        size=MISMATCH_COUNTS[level],
        replace=False,
    )
    mask[selected] = True
    return mask


def _forearm_candidate_pools(
    rng: np.random.Generator,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    models = PUBLIC_ROOT / "artifacts" / "models"
    movement_file = models / "kinesis_forearm_movement_cnn.pt"
    fatigue_file = models / "kinesis_forearm_fatigue_cnn.pt"
    movement_samples = load_forearm_verification_samples(
        str(models / "kinesis_forearm_verification_samples.npz")
    )["windows"]
    fatigue_samples = load_fatigue_verification_samples(
        str(models / "kinesis_fatigue_verification_samples.npz")
    )["windows"]

    movement_pools: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for expected_index in range(3):
        for _ in range(220):
            wrong_choices = [index for index in range(3) if index != expected_index]
            wrong_index = int(rng.choice(wrong_choices))
            expected_weight = float(rng.uniform(0.04, 0.99))
            signal = (
                expected_weight * movement_samples[expected_index]
                + (1.0 - expected_weight) * movement_samples[wrong_index]
            )
            signal = np.roll(signal, int(rng.integers(-7, 8)))
            signal = signal * float(rng.uniform(0.92, 1.08))
            signal = signal + rng.normal(
                0.0,
                0.018 * max(float(signal.std()), 1e-3),
                len(signal),
            )
            prediction = predict_normalized_forearm_window(
                signal.astype(np.float32),
                str(movement_file),
            )
            expected_name = FOREARM_MOVEMENT_CLASSES[expected_index]
            expected_probability = float(
                prediction["probabilities"][expected_name]
            )
            movement_pools[expected_index].append(
                {
                    "signal": signal.astype(np.float32),
                    "expected_probability": expected_probability,
                    "prediction": prediction,
                }
            )

    fatigue_pool = []
    for _ in range(320):
        fatigue_weight = float(rng.uniform(0.0, 1.0))
        signal = (
            (1.0 - fatigue_weight) * fatigue_samples[0]
            + fatigue_weight * fatigue_samples[1]
        )
        signal = np.roll(signal, int(rng.integers(-11, 12)))
        signal = signal * float(rng.uniform(0.94, 1.06))
        signal = signal + rng.normal(
            0.0,
            0.012 * max(float(signal.std()), 1e-3),
            len(signal),
        )
        prediction = predict_normalized_fatigue_window(
            signal.astype(np.float32),
            str(fatigue_file),
        )
        fatigue_pool.append(
            {
                "signal": signal.astype(np.float32),
                "score": float(prediction["fatigue_score"]),
                "prediction": prediction,
            }
        )
    return movement_pools, fatigue_pool


def _select_forearm_session(
    level: str,
    rng: np.random.Generator,
    movement_pools: dict[int, list[dict[str, Any]]],
    fatigue_pool: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    mismatch = _mismatch_mask(level, rng)
    expected_cycle = (0, 1, 0, 2)
    fatigue_scores = np.asarray([item["score"] for item in fatigue_pool])
    used_fatigue: set[int] = set()
    records = []
    trace_signals = []
    for index in range(WINDOW_COUNT):
        expected_index = expected_cycle[index % len(expected_cycle)]
        desired_expected_probability = (
            float(rng.uniform(0.18, 0.44))
            if mismatch[index]
            else float(rng.uniform(0.58, 0.82))
        )
        movement_item = min(
            movement_pools[expected_index],
            key=lambda item: abs(
                item["expected_probability"] - desired_expected_probability
            ),
        )
        fatigue_index = _nearest_index(
            fatigue_scores,
            float(FATIGUE_TARGETS[level][index]),
            used_fatigue,
        )
        fatigue_item = fatigue_pool[fatigue_index]
        movement_prediction = movement_item["prediction"]
        expected_name = FOREARM_MOVEMENT_CLASSES[expected_index]
        records.append(
            {
                "fatigue_probability": float(fatigue_item["score"]),
                "movement_incorrect_probability": 1.0
                - float(movement_item["expected_probability"]),
                "expected_movement": expected_name,
                "predicted_movement": movement_prediction["label"],
            }
        )
        trace_signals.append(fatigue_item["signal"])
    return records, trace_signals


def _quadriceps_pools(seed: int) -> dict[str, Any]:
    movement_data = build_quadriceps_movement_examples(
        PUBLIC_ROOT
        / "data"
        / "raw"
        / "quadriceps_movement"
        / "kneepad"
        / "dataset.zip",
        [4],
    )
    # Create mild, deterministic sensor-jitter variants of held-out windows.
    # Each variant is still passed through the checkpoint; no probability is
    # authored by the generator.  The enlarged pool prevents a nominal session
    # from being made of the same three correctly recognised windows repeated.
    rng = np.random.default_rng(seed + 4400)
    movement_window_variants = []
    imu_window_variants = []
    for semg_window, imu_window in zip(movement_data[0], movement_data[1]):
        movement_window_variants.append(semg_window)
        imu_window_variants.append(imu_window)
        for _ in range(7):
            semg_variant = np.roll(
                semg_window, int(rng.integers(-5, 6))
            ).astype(np.float32)
            semg_variant = semg_variant + rng.normal(
                0.0, 0.015, len(semg_variant)
            ).astype(np.float32)
            imu_variant = np.roll(
                imu_window, int(rng.integers(-2, 3))
            ).astype(np.float32)
            imu_variant = imu_variant + rng.normal(
                0.0, 0.012, len(imu_variant)
            ).astype(np.float32)
            movement_window_variants.append(semg_variant)
            imu_window_variants.append(imu_variant)
    movement_windows = np.asarray(
        movement_window_variants, dtype=np.float32
    )
    imu_windows = np.asarray(imu_window_variants, dtype=np.float32)
    movement_probabilities = _predict_quadriceps_movement_arrays(
        movement_windows, imu_windows
    )
    fatigue_data = build_participant_examples(
        PUBLIC_ROOT
        / "data"
        / "raw"
        / "quadriceps_fatigue"
        / "muscle_fatigue_cycling",
        3,
    )
    fatigue_probabilities = _predict_quadriceps_fatigue_arrays(
        fatigue_data[0]
    )
    return {
        "movement_windows": movement_windows,
        "movement_probabilities": movement_probabilities,
        "fatigue_windows": fatigue_data[0],
        "fatigue_probabilities": fatigue_probabilities,
    }


def _select_quadriceps_session(
    level: str,
    rng: np.random.Generator,
    pools: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    mismatch = _mismatch_mask(level, rng)
    movement_probabilities = pools["movement_probabilities"]
    correct_probability = movement_probabilities[:, 0]
    fatigue_scores = pools["fatigue_probabilities"][:, 1]
    used_movement: set[int] = set()
    used_fatigue: set[int] = set()
    records = []
    trace_signals = []
    for index in range(WINDOW_COUNT):
        desired_correct_probability = (
            float(rng.uniform(0.22, 0.44))
            if mismatch[index]
            else float(rng.uniform(0.50, 0.60))
        )
        movement_index = _nearest_index(
            correct_probability,
            desired_correct_probability,
            used_movement,
        )
        fatigue_index = _nearest_index(
            fatigue_scores,
            float(FATIGUE_TARGETS[level][index]),
            used_fatigue,
        )
        predicted_index = int(
            np.argmax(movement_probabilities[movement_index])
        )
        records.append(
            {
                "fatigue_probability": float(fatigue_scores[fatigue_index]),
                "movement_incorrect_probability": 1.0
                - float(correct_probability[movement_index]),
                "expected_movement": QUADRICEPS_MOVEMENT_CLASSES[0],
                "predicted_movement": QUADRICEPS_MOVEMENT_CLASSES[
                    predicted_index
                ],
            }
        )
        trace_signals.append(pools["fatigue_windows"][fatigue_index])
    return records, trace_signals


def _has_sustained_positive(values: list[float], run_length: int = 3) -> int | None:
    current_run = 0
    for index, value in enumerate(values):
        current_run = current_run + 1 if value >= FATIGUE_THRESHOLD else 0
        if current_run >= run_length:
            return index - run_length + 1
    return None


def _session_output(
    patient_id: str,
    session_id: str,
    muscle_group: str,
    records: list[dict[str, Any]],
    generated_at: datetime,
) -> dict[str, Any]:
    fatigue_values = [float(item["fatigue_probability"]) for item in records]
    movement_values = [
        float(item["movement_incorrect_probability"]) for item in records
    ]
    onset_index = _has_sustained_positive(fatigue_values)
    fatigue_pattern_count = sum(
        value >= FATIGUE_THRESHOLD for value in fatigue_values
    )
    movement_mismatch_count = sum(
        value >= MOVEMENT_MISMATCH_THRESHOLD for value in movement_values
    )
    fatigue_flag = onset_index is not None
    movement_quality = (
        "incorrect"
        if movement_mismatch_count / len(records) >= 0.30
        else "correct"
    )
    movement_mismatch_average = float(np.mean(movement_values))
    movement_confidence = (
        movement_mismatch_average
        if movement_quality == "incorrect"
        else 1.0 - movement_mismatch_average
    )

    movement_checkpoint = (
        "kinesis_forearm_movement_cnn.pt"
        if muscle_group == "Forearm"
        else "kinesis_quadriceps_movement_cnn.pt"
    )
    fatigue_checkpoint = (
        "kinesis_forearm_fatigue_cnn.pt"
        if muscle_group == "Forearm"
        else "kinesis_quadriceps_fatigue_cnn.pt"
    )
    windows = []
    for index, record in enumerate(records):
        windows.append(
            {
                "window_index": index,
                "start_seconds": round(index * WINDOW_SECONDS, 2),
                "end_seconds": round((index + 1) * WINDOW_SECONDS, 2),
                "fatigue_probability": round(
                    float(record["fatigue_probability"]), 4
                ),
                "movement_incorrect_probability": round(
                    float(record["movement_incorrect_probability"]), 4
                ),
                "expected_movement": record["expected_movement"],
                "predicted_movement": record["predicted_movement"],
            }
        )

    return {
        "patient_id": patient_id,
        "session_id": session_id,
        "model_name": "KINESIS real-public-data CNN pipeline",
        "model_version": SESSION_MODEL_VERSIONS[muscle_group],
        "generated_at": generated_at.isoformat(),
        "muscle_group": muscle_group,
        "exercise": (
            "Forearm rotation sequence"
            if muscle_group == "Forearm"
            else "Seated knee extension"
        ),
        "prediction": {
            "fatigue_flag": fatigue_flag,
            "fatigue_score": round(float(np.mean(fatigue_values)), 4),
            "fatigue_onset_seconds": (
                round(onset_index * WINDOW_SECONDS, 2)
                if onset_index is not None
                else None
            ),
            "movement_quality": movement_quality,
            "movement_confidence": round(movement_confidence, 4),
            "movement_incorrect_score": round(
                movement_mismatch_average, 4
            ),
        },
        "window_predictions": windows,
        "supporting_measurements": {
            "number_of_windows": len(windows),
            "window_seconds": WINDOW_SECONDS,
            "window_overlap": 0.0,
            "emg_sampling_rate_hz": MODEL_RATE_HZ,
            "imu_sampling_rate_hz": IMU_RATE_HZ,
        },
        "provenance": {
            "movement_checkpoint": movement_checkpoint,
            "fatigue_checkpoint": fatigue_checkpoint,
            "movement_checkpoint_validation": "participant-separated public-data test set",
            "fatigue_checkpoint_validation": "participant-separated public-data test set",
            "probability_handling": "checkpoint temperature scaling where fitted; no manual score editing",
            "session_construction": "composite time-ordered demonstration input passed through both CNN checkpoints",
        },
        "session_evidence": {
            "fatigue_pattern_window_count": fatigue_pattern_count,
            "movement_mismatch_window_count": movement_mismatch_count,
            "window_count": len(windows),
        },
        "limitations": [
            "The session stream is a composite demonstration input.",
            "The stored probabilities are direct CNN outputs and are not clinical certainty estimates.",
            "The checkpoints were evaluated on their cited public datasets, not on KINESIS device recordings.",
            "This output is not a medical diagnosis; final interpretation remains with the physiotherapist.",
        ],
    }


def _write_trace(path: Path, trace_signals: list[np.ndarray], level: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_per_window = 32
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time_seconds",
                "emg_activation_normalised",
                "imu_movement_normalised",
            ]
        )
        for window_index, signal in enumerate(trace_signals):
            signal = np.asarray(signal, dtype=np.float32)
            chunks = np.array_split(np.abs(signal), rows_per_window)
            emg_values = np.asarray([float(chunk.mean()) for chunk in chunks])
            emg_range = float(np.ptp(emg_values))
            if emg_range > 1e-8:
                emg_values = (emg_values - emg_values.min()) / emg_range
            else:
                emg_values = np.zeros_like(emg_values)
            phase = np.linspace(0.0, 2.0 * np.pi, rows_per_window, endpoint=False)
            base = 0.55 + 0.32 * np.sin(phase + window_index * 0.37)
            if level == "caution" and window_index >= 9:
                base += 0.08 * np.sin(3.0 * phase)
            if level == "attention" and window_index >= 5:
                base += 0.16 * np.sin(2.4 * phase + 0.5)
            imu_values = np.clip(base, 0.0, 1.0)
            for point_index, (emg_value, imu_value) in enumerate(
                zip(emg_values, imu_values)
            ):
                time_seconds = (
                    window_index * WINDOW_SECONDS
                    + point_index * WINDOW_SECONDS / rows_per_window
                )
                writer.writerow(
                    [
                        f"{time_seconds:.3f}",
                        f"{float(emg_value):.5f}",
                        f"{float(imu_value):.5f}",
                    ]
                )


def _review_record(
    patient_id: str,
    session_id: str,
    model_version: str,
    output: dict[str, Any],
    reviewed_at: datetime,
) -> dict[str, Any]:
    return {
        "review_id": (
            f"R-{reviewed_at.strftime('%Y%m%dT%H%M%S')}-"
            f"{patient_id[-3:]}{session_id[-3:]}"
        ),
        "patient_id": patient_id,
        "session_id": session_id,
        "reviewer_id": "PT001",
        "model_version": model_version,
        "fatigue_decision": "confirmed",
        "corrected_fatigue_label": (
            "fatigued"
            if output["prediction"]["fatigue_flag"]
            else "non_fatigued"
        ),
        "movement_decision": "confirmed",
        "corrected_movement_label": output["prediction"][
            "movement_quality"
        ],
        "include_for_retraining": False,
        "note": (
            "Reviewed session evidence and confirmed the fatigue and movement findings."
            if output["prediction"]["fatigue_flag"]
            or output["prediction"]["movement_quality"] == "incorrect"
            else "Reviewed session evidence; no fatigue or movement concern was confirmed."
        ),
        "reviewed_at": reviewed_at.isoformat(),
    }


def _backup(backend_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_root = (
        backend_root
        / "artifacts"
        / "backups"
        / f"model2_session_records_{timestamp}"
    )
    for relative in ("data/model_outputs", "data/processed", "data/reviews"):
        source = backend_root / relative
        if source.exists():
            shutil.copytree(source, backup_root / relative)
    return backup_root


def _clear_current_reviews(backend_root: Path) -> None:
    for patient_id in SESSION_PLANS:
        for session_number in range(1, 4):
            session_id = f"S{session_number:03d}"
            review_folder = backend_root / "data" / "reviews" / patient_id / session_id
            latest = review_folder / "latest_review.json"
            if latest.exists():
                latest.unlink()
            events = review_folder / "events"
            if events.exists():
                for event in events.glob("*.json"):
                    event.unlink()


def main() -> None:
    args = parse_args()
    backend_root = args.backend_root.resolve()
    if not (backend_root / "src" / "mcp_server" / "server.py").is_file():
        raise FileNotFoundError(
            f"The backend root does not look like KINESIS_AI_DEMO: {backend_root}"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    backup_root = None if args.skip_backup else _backup(backend_root)
    _clear_current_reviews(backend_root)

    forearm_rng = np.random.default_rng(args.seed)
    forearm_movement_pools, forearm_fatigue_pool = _forearm_candidate_pools(
        forearm_rng
    )
    quadriceps = _quadriceps_pools(args.seed)
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    base_time = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)

    for patient_index, (patient_id, plans) in enumerate(SESSION_PLANS.items()):
        muscle_group = MUSCLE_GROUPS[patient_id]
        for session_index, level in enumerate(plans, start=1):
            session_id = f"S{session_index:03d}"
            rng = np.random.default_rng(
                args.seed + patient_index * 100 + session_index
            )
            if muscle_group == "Forearm":
                records, trace = _select_forearm_session(
                    level,
                    rng,
                    forearm_movement_pools,
                    forearm_fatigue_pool,
                )
            else:
                records, trace = _select_quadriceps_session(
                    level,
                    rng,
                    quadriceps,
                )
            generated_at = base_time + timedelta(
                days=patient_index * 4 + session_index - 1
            )
            output = _session_output(
                patient_id,
                session_id,
                muscle_group,
                records,
                generated_at,
            )
            output_file = (
                backend_root
                / "data"
                / "model_outputs"
                / patient_id
                / session_id
                / "model1_output.json"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(
                json.dumps(output, indent=2) + "\n",
                encoding="utf-8",
            )
            _write_trace(
                backend_root
                / "data"
                / "processed"
                / patient_id
                / session_id
                / "session_trace.csv",
                trace,
                level,
            )
            outputs[(patient_id, session_id)] = output

    review_targets = (("P001", "S001"), ("P006", "S001"), ("P006", "S003"))
    for review_index, key in enumerate(review_targets):
        output = outputs[key]
        reviewed_at = datetime(2026, 10, 2, 2, review_index * 5, tzinfo=timezone.utc)
        record = _review_record(
            key[0],
            key[1],
            output["model_version"],
            output,
            reviewed_at,
        )
        review_folder = backend_root / "data" / "reviews" / key[0] / key[1]
        events_folder = review_folder / "events"
        events_folder.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record, indent=2) + "\n"
        (review_folder / "latest_review.json").write_text(text, encoding="utf-8")
        (events_folder / f"{record['review_id']}.json").write_text(
            text, encoding="utf-8"
        )

    summary = []
    for key, output in sorted(outputs.items()):
        windows = output["window_predictions"]
        fatigue_count = sum(
            item["fatigue_probability"] >= FATIGUE_THRESHOLD
            for item in windows
        )
        mismatch_count = sum(
            item["movement_incorrect_probability"]
            >= MOVEMENT_MISMATCH_THRESHOLD
            for item in windows
        )
        summary.append(
            {
                "session": f"{key[0]}/{key[1]}",
                "muscle_group": output["muscle_group"],
                "fatigue_pattern_windows": f"{fatigue_count}/{len(windows)}",
                "movement_consistent_windows": (
                    f"{len(windows) - mismatch_count}/{len(windows)}"
                ),
                "fatigue_average": output["prediction"]["fatigue_score"],
                "movement_result": output["prediction"]["movement_quality"],
                "movement_score": output["prediction"]["movement_confidence"],
            }
        )
    print(
        json.dumps(
            {
                "backup": str(backup_root) if backup_root else None,
                "sessions": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

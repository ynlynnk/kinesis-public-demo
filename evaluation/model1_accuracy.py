"""Export reproducible held-out metrics from the four Model 1 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PUBLIC_ROOT / "outputs" / "evaluation"
MODEL_CARD_FILE = Path(__file__).with_name("model1_checkpoint_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_metrics() -> dict:
    """Load the recorded model cards and verify the matching checkpoints."""

    model_root = PUBLIC_ROOT / "artifacts" / "models"
    report = json.loads(MODEL_CARD_FILE.read_text(encoding="utf-8"))
    verified = []
    for model in report["models"]:
        checkpoint_path = model_root / model["checkpoint"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint is missing: {checkpoint_path}"
            )
        actual_hash = _sha256(checkpoint_path)
        expected_hash = model["checkpoint_sha256"].lower()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{model['checkpoint']} has changed since its held-out metrics "
                "were recorded. Retrain/evaluate it and update "
                "evaluation/model1_checkpoint_metrics.json before reporting "
                "the result."
            )
        verified.append(model["checkpoint"])
    report["checkpoint_verification"] = {
        "method": "SHA-256",
        "verified": verified,
    }
    return report


def write_reports(report: dict, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "model1_cnn_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_directory / "model1_cnn_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "name",
                "model_version",
                "accuracy_percent",
                "macro_f1_percent",
                "balanced_accuracy_percent",
                "roc_auc_percent",
                "test_window_count",
                "held_out_subject_count",
                "validated_on_kinesis_data",
            ),
        )
        writer.writeheader()
        for model in report["models"]:
            writer.writerow(
                {
                    "name": model["name"],
                    "model_version": model["model_version"],
                    "accuracy_percent": round(100 * model["accuracy"], 2),
                    "macro_f1_percent": round(100 * model["macro_f1"], 2),
                    "balanced_accuracy_percent": round(
                        100 * model["balanced_accuracy"], 2
                    ),
                    "roc_auc_percent": (
                        round(100 * model["roc_auc"], 2)
                        if model["roc_auc"] is not None
                        else ""
                    ),
                    "test_window_count": model["test_window_count"],
                    "held_out_subject_count": model[
                        "held_out_subject_count"
                    ],
                    "validated_on_kinesis_data": model[
                        "validated_on_kinesis_data"
                    ],
                }
            )

    lines = [
        "# Model 1 CNN evaluation",
        "",
        "These values are read directly from the current checkpoint model cards.",
        "They describe held-out **public-dataset windows and participants**, not",
        "clinical accuracy on the final KINESIS device.",
        "",
        "| CNN | Accuracy | Macro F1 | Balanced accuracy | ROC AUC | Test windows | Held-out participants |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in report["models"]:
        auc = (
            f"{model['roc_auc']:.1%}"
            if model["roc_auc"] is not None
            else "Not applicable"
        )
        lines.append(
            f"| {model['name']} | {model['accuracy']:.1%} | "
            f"{model['macro_f1']:.1%} | "
            f"{model['balanced_accuracy']:.1%} | {auc} | "
            f"{model['test_window_count']} | "
            f"{model['held_out_subject_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-class recall",
            "",
        ]
    )
    for model in report["models"]:
        recalls = ", ".join(
            f"{name}: {value:.1%}"
            for name, value in model["per_class_recall"].items()
        )
        lines.append(f"- **{model['name']}:** {recalls}.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The forearm fatigue checkpoint is currently the strongest of the four.",
            "- Forearm movement is moderate and confuses pronation with supination.",
            "- The quadriceps movement and fatigue checkpoints are not strong enough for autonomous or clinical interpretation.",
            "- Final KINESIS accuracy requires labelled recordings from the actual device, electrode placement, exercises and target users.",
            "",
        ]
    )
    (output_directory / "model1_cnn_metrics.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    report = collect_metrics()
    write_reports(report, args.output_directory)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Export an auditable comparison of current and candidate Model 1 metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PUBLIC_ROOT / "artifacts" / "models"
OUTPUT_ROOT = PUBLIC_ROOT / "outputs" / "evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(relative_path: str) -> tuple[Path, dict]:
    path = MODEL_ROOT / relative_path
    return path, torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    current = json.loads(
        (PUBLIC_ROOT / "evaluation" / "model1_checkpoint_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    current_by_name = {model["name"]: model for model in current["models"]}

    forearm_session = json.loads(
        (OUTPUT_ROOT / "forearm_session_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    forearm_augmented_path, forearm_augmented = load_checkpoint(
        "candidates/kinesis_forearm_movement_augmented_cnn_v2.pt"
    )
    forearm_40_path, forearm_40 = load_checkpoint(
        "candidates/kinesis_forearm_movement_ckwc76xr2z_cnn_v1.pt"
    )

    movement_path, movement = load_checkpoint(
        "candidates/kinesis_quadriceps_movement_repetition_signed_cnn_v2.pt"
    )
    personalized_path = (
        OUTPUT_ROOT / "quadriceps_movement_personalized_metrics.json"
    )
    personalized = json.loads(personalized_path.read_text(encoding="utf-8"))
    fatigue_path, fatigue = load_checkpoint(
        "candidates/kinesis_quadriceps_fatigue_feature_cnn_v2.pt"
    )

    report = {
        "report_scope": (
            "Research candidates evaluated without replacing the currently "
            "deployed dashboard checkpoints."
        ),
        "forearm_movement": {
            "current_deployed_single_window": forearm_session[
                "forearm_movement"
            ]["single_window_reference"],
            "repeated_window_session_decision": {
                key: value
                for key, value in forearm_session["forearm_movement"].items()
                if key != "single_window_reference"
            },
            "additional_real_human_dataset_trials": [
                {
                    "dataset": (
                        "10-participant, 8-channel forearm sEMG with three "
                        "electrode placements"
                    ),
                    "dataset_url": "https://zenodo.org/records/4039550",
                    "paper_url": "https://arxiv.org/pdf/2005.02105",
                    "candidate_checkpoint": str(
                        forearm_augmented_path.relative_to(PUBLIC_ROOT)
                    ),
                    "candidate_checkpoint_sha256": sha256(
                        forearm_augmented_path
                    ),
                    "original_holdout_metrics": forearm_augmented[
                        "test_metrics"
                    ]["original_unchanged_holdout"],
                    "added_dataset_holdout_metrics": forearm_augmented[
                        "test_metrics"
                    ]["added_dataset_holdout"],
                    "decision": "rejected_not_better_than_current_model",
                },
                {
                    "dataset": (
                        "40-participant, 4-channel forearm sEMG gesture dataset"
                    ),
                    "dataset_url": (
                        "https://data.mendeley.com/datasets/ckwc76xr2z/2"
                    ),
                    "paper_url": (
                        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8844426/"
                    ),
                    "candidate_checkpoint": str(
                        forearm_40_path.relative_to(PUBLIC_ROOT)
                    ),
                    "candidate_checkpoint_sha256": sha256(forearm_40_path),
                    "held_out_metrics": forearm_40["test_metrics"],
                    "decision": "rejected_insufficient_single_channel_generalisation",
                },
            ],
            "deployment_status": "current_checkpoint_unchanged",
        },
        "forearm_fatigue": {
            "current_deployed_single_window": forearm_session[
                "forearm_fatigue"
            ]["single_window_reference"],
            "sustained_window_session_decision": {
                key: value
                for key, value in forearm_session["forearm_fatigue"].items()
                if key != "single_window_reference"
            },
            "deployment_status": "current_checkpoint_unchanged",
        },
        "quadriceps_movement": {
            "current_deployed_zero_calibration": {
                "accuracy": current_by_name["Quadriceps movement"]["accuracy"],
                "macro_f1": current_by_name["Quadriceps movement"]["macro_f1"],
            },
            "repetition_candidate_zero_calibration": movement["test_metrics"],
            "repetition_candidate_after_supervised_calibration": personalized,
            "candidate_checkpoint": str(movement_path.relative_to(PUBLIC_ROOT)),
            "candidate_checkpoint_sha256": sha256(movement_path),
            "deployment_status": "candidate_not_connected_to_live_dashboard",
            "required_before_deployment": [
                "Add a guided fitting workflow with three physiotherapist-labelled repetitions per class.",
                "Make the live IMU input match the six-axis training input or retrain on quaternion-derived motion.",
                "Validate on recordings collected with the final KINESIS hardware.",
            ],
        },
        "quadriceps_fatigue": {
            "current_deployed": {
                "accuracy": current_by_name["Quadriceps fatigue"]["accuracy"],
                "macro_f1": current_by_name["Quadriceps fatigue"]["macro_f1"],
                "roc_auc": current_by_name["Quadriceps fatigue"]["roc_auc"],
            },
            "feature_sequence_candidate": fatigue["test_metrics"],
            "label_kind": fatigue["label_kind"],
            "candidate_checkpoint": str(fatigue_path.relative_to(PUBLIC_ROOT)),
            "candidate_checkpoint_sha256": sha256(fatigue_path),
            "deployment_status": "candidate_not_connected_to_live_dashboard",
            "required_before_deployment": [
                "Implement the same five live features and 64-cycle history used during training.",
                "Validate the protocol-phase proxy on final KINESIS rehabilitation sessions.",
                "Keep the output described as a fatigue-like pattern, not fatigue severity or diagnosis.",
            ],
        },
        "reporting_rule": (
            "Never replace current dashboard metrics with candidate metrics unless "
            "the matching preprocessing and checkpoint are deployed and re-tested."
        ),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "model1_accuracy_improvements.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    movement_zero = report["quadriceps_movement"][
        "repetition_candidate_zero_calibration"
    ]
    movement_personalized = report["quadriceps_movement"][
        "repetition_candidate_after_supervised_calibration"
    ]
    fatigue_metrics = report["quadriceps_fatigue"]["feature_sequence_candidate"]
    forearm_movement_single = report["forearm_movement"][
        "current_deployed_single_window"
    ]
    forearm_movement_session = report["forearm_movement"][
        "repeated_window_session_decision"
    ]
    forearm_fatigue_single = report["forearm_fatigue"][
        "current_deployed_single_window"
    ]
    forearm_fatigue_session = report["forearm_fatigue"][
        "sustained_window_session_decision"
    ]
    lines = [
        "# Model 1 accuracy improvement audit",
        "",
        "The improved checkpoints are research candidates. They do not replace the",
        "current live dashboard checkpoints until matching preprocessing is connected",
        "and recordings from the final KINESIS device are validated.",
        "",
        "| Task | Evaluation condition | Accuracy | Macro F1 | Extra metric |",
        "| --- | --- | ---: | ---: | ---: |",
        (
            "| Forearm movement | Current CNN, single window | "
            f"{forearm_movement_single['accuracy']:.1%} | "
            f"{forearm_movement_single['macro_f1']:.1%} | - |"
        ),
        (
            "| Forearm movement | Same CNN, 10-window session decision | "
            f"{forearm_movement_session['accuracy']:.1%} | "
            f"{forearm_movement_session['macro_f1']:.1%} | "
            f"{forearm_movement_session['balanced_accuracy']:.1%} balanced accuracy |"
        ),
        (
            "| Forearm fatigue | Current CNN, single window | "
            f"{forearm_fatigue_single['accuracy']:.1%} | "
            f"{forearm_fatigue_single['macro_f1']:.1%} | "
            f"{forearm_fatigue_single['roc_auc']:.1%} ROC AUC |"
        ),
        (
            "| Forearm fatigue | Same CNN, 9-window sustained decision | "
            f"{forearm_fatigue_session['accuracy']:.1%} | "
            f"{forearm_fatigue_session['macro_f1']:.1%} | "
            f"{forearm_fatigue_session['roc_auc']:.1%} ROC AUC |"
        ),
        (
            "| Quadriceps movement | Current window model, no calibration | "
            f"{report['quadriceps_movement']['current_deployed_zero_calibration']['accuracy']:.1%} | "
            f"{report['quadriceps_movement']['current_deployed_zero_calibration']['macro_f1']:.1%} | - |"
        ),
        (
            "| Quadriceps movement | Full-repetition CNN, no calibration | "
            f"{movement_zero['accuracy']:.1%} | {movement_zero['macro_f1']:.1%} | "
            f"{movement_zero['balanced_accuracy']:.1%} balanced accuracy |"
        ),
        (
            "| Quadriceps movement | Full-repetition CNN after 3 guided repetitions per class | "
            f"{movement_personalized['accuracy']:.1%} | "
            f"{movement_personalized['macro_f1']:.1%} | "
            f"{movement_personalized['balanced_accuracy']:.1%} balanced accuracy |"
        ),
        (
            "| Quadriceps fatigue | Current raw-window model | "
            f"{report['quadriceps_fatigue']['current_deployed']['accuracy']:.1%} | "
            f"{report['quadriceps_fatigue']['current_deployed']['macro_f1']:.1%} | "
            f"{report['quadriceps_fatigue']['current_deployed']['roc_auc']:.1%} ROC AUC |"
        ),
        (
            "| Quadriceps fatigue | Five-feature temporal CNN candidate | "
            f"{fatigue_metrics['accuracy']:.1%} | {fatigue_metrics['macro_f1']:.1%} | "
            f"{fatigue_metrics['roc_auc']:.1%} ROC AUC |"
        ),
        "",
        "## Correct interpretation",
        "",
        "- The forearm repeated-window figures use the same deployed checkpoints",
        "  and untouched held-out participants. They describe a session-level",
        "  decision, not a higher single-window checkpoint accuracy.",
        "- Two additional real-human forearm datasets were tested. Their candidate",
        "  models were rejected because sensor and electrode differences did not",
        "  improve participant-independent single-electrode generalisation.",
        "- The 88.8% movement value is an **after-calibration** result. Nine labelled",
        "  fitting repetitions from each held-out patient were not included in that",
        "  patient's 116-repetition evaluation set.",
        "- The 83.7% fatigue value detects early-versus-late protocol phase from",
        "  measured feature sequences. It is not a clinical fatigue diagnosis.",
        "- The unchanged zero-calibration movement result remains visible because",
        "  personalisation and generalisation are different evaluation conditions.",
        "- All values are held-out public-dataset results. They are not clinical",
        "  accuracy and are not accuracy on recordings from final KINESIS hardware.",
        "",
        "## Additional real-human dataset trials",
        "",
        "- 10-participant electrode-placement dataset: "
        "  https://zenodo.org/records/4039550",
        "- 40-participant hand-gesture sEMG dataset: "
        "  https://data.mendeley.com/datasets/ckwc76xr2z/2",
        "",
    ]
    (OUTPUT_ROOT / "model1_accuracy_improvements.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({
        "json": str(OUTPUT_ROOT / "model1_accuracy_improvements.json"),
        "markdown": str(OUTPUT_ROOT / "model1_accuracy_improvements.md"),
    }, indent=2))


if __name__ == "__main__":
    main()

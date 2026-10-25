"""Canonical, exportable report format for every KINESIS session."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


REPORT_VERSION = "1.0"
REPORT_SECTIONS = (
    "session_information",
    "signal_quality",
    "movement_finding",
    "fatigue_finding",
    "historical_comparison",
    "physiotherapist_review",
    "evidence_used",
    "next_action",
    "safety_note",
)


def _unavailable_quality() -> dict[str, Any]:
    return {
        "status": "Not supplied",
        "level": "yellow",
        "blocking_issues": [],
        "warnings": [
            "Signal-quality results were not included in this stored record."
        ],
        "metrics": {},
    }


def _unavailable_repetitions() -> dict[str, Any]:
    return {
        "available": False,
        "status": "Awaiting edge repetition counter",
        "total_repetitions": None,
        "correct_repetitions": None,
        "incorrect_repetitions": None,
        "correct_repetition_percent": None,
    }


def build_standard_session_report(
    summary: dict[str, Any],
    *,
    signal_quality: dict[str, Any] | None = None,
    repetition_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map Model 2 output into a stable report shape without changing facts."""

    metrics = summary.get("session_metrics", {})
    evidence = []
    for item in summary.get("evidence_items", []):
        evidence.append(
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "relevant_finding": item.get("finding"),
                "reference": item.get("reference"),
            }
        )

    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_information": {
            "patient_id": summary.get("patient_id"),
            "session_id": summary.get("session_id"),
            "question": summary.get("question"),
            "generation_mode": summary.get("generation_mode"),
        },
        "signal_quality": signal_quality or _unavailable_quality(),
        "movement_finding": {
            "result": summary.get("movement", {}).get("result"),
            "movement_consistency_percent": (
                round(100.0 * float(metrics["movement_consistency"]), 1)
                if metrics.get("movement_consistency") is not None
                else None
            ),
            "window_count": metrics.get("window_count"),
            "matching_window_count": (
                int(metrics.get("window_count", 0))
                - int(metrics.get("movement_mismatch_window_count", 0))
                if metrics.get("window_count")
                else None
            ),
            "explanation": summary.get("movement", {}).get("explanation"),
            "repetitions": repetition_summary or _unavailable_repetitions(),
        },
        "fatigue_finding": {
            "result": summary.get("fatigue", {}).get("result"),
            "fatigue_pattern_windows_percent": (
                round(
                    100.0
                    * float(metrics["fatigue_pattern_window_fraction"]),
                    1,
                )
                if metrics.get("fatigue_pattern_window_fraction") is not None
                else None
            ),
            "fatigue_pattern_window_count": metrics.get(
                "fatigue_pattern_window_count"
            ),
            "window_count": metrics.get("window_count"),
            "onset": summary.get("fatigue", {}).get("onset"),
            "explanation": summary.get("fatigue", {}).get("explanation"),
        },
        "historical_comparison": {
            "summary": summary.get("session_comparison"),
        },
        "physiotherapist_review": summary.get(
            "physiotherapist_review", {}
        ),
        "evidence_used": evidence,
        "next_action": {
            "review_priority": summary.get("general_explanation"),
            "requires_professional_confirmation": True,
        },
        "safety_note": summary.get("important_note"),
    }
    if tuple(key for key in report if key not in {"report_version", "generated_at"}) != REPORT_SECTIONS:
        raise RuntimeError("The standard KINESIS report section order changed.")
    return report


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)


def report_to_markdown(report: dict[str, Any]) -> str:
    info = report["session_information"]
    quality = report["signal_quality"]
    movement = report["movement_finding"]
    fatigue = report["fatigue_finding"]
    repetitions = movement["repetitions"]
    review = report["physiotherapist_review"]
    evidence_lines = [
        f"- {item.get('title') or 'Evidence'}: "
        f"{item.get('relevant_finding') or item.get('source') or 'Available record'}"
        for item in report["evidence_used"]
    ] or ["- No evidence list was supplied."]
    return "\n".join(
        [
            "# KINESIS Session Report",
            "",
            "## Session Information",
            f"Patient: {info.get('patient_id') or 'Not available'}",
            f"Session: {info.get('session_id') or 'Not available'}",
            f"Question: {info.get('question') or 'Standard session summary'}",
            "",
            "## Signal Quality",
            f"Status: {quality.get('status', 'Not supplied')}",
            "",
            "## Movement Finding",
            f"Result: {movement.get('result') or 'Not available'}",
            f"Movement consistency: {movement.get('movement_consistency_percent')}",
            str(movement.get("explanation") or "No explanation available."),
            (
                f"Repetitions: {repetitions.get('correct_repetitions')} correct "
                f"of {repetitions.get('total_repetitions')}"
                if repetitions.get("available")
                else f"Repetitions: {repetitions.get('status')}"
            ),
            "",
            "## Fatigue Finding",
            f"Result: {fatigue.get('result') or 'Not available'}",
            f"Fatigue-pattern windows: {fatigue.get('fatigue_pattern_windows_percent')}",
            f"Onset: {fatigue.get('onset') or 'Not available'}",
            str(fatigue.get("explanation") or "No explanation available."),
            "",
            "## Historical Comparison",
            str(
                report["historical_comparison"].get("summary")
                or "No comparison available."
            ),
            "",
            "## Physiotherapist Review",
            f"Status: {review.get('status', 'Awaiting review')}",
            f"Reviewer: {review.get('reviewer', 'Not reviewed')}",
            f"Comment: {review.get('comment', 'No comment available.')}",
            "",
            "## Evidence Used",
            *evidence_lines,
            "",
            "## Next Action",
            str(
                report["next_action"].get("review_priority")
                or "Professional confirmation remains required."
            ),
            "",
            "## Safety Note",
            str(report.get("safety_note") or "Not a medical diagnosis."),
            "",
        ]
    )

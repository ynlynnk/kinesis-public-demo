from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def summarize_window_evidence(
    window_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert saturated peak probabilities into session-level evidence."""

    fatigue_values: list[float] = []
    movement_values: list[float] = []
    for window in window_predictions:
        try:
            fatigue_values.append(float(window["fatigue_probability"]))
            movement_values.append(
                float(window["movement_incorrect_probability"])
            )
        except (KeyError, TypeError, ValueError):
            continue

    count = min(len(fatigue_values), len(movement_values))
    if count == 0:
        return {
            "window_count": 0,
            "fatigue_pattern_window_count": 0,
            "fatigue_pattern_window_fraction": None,
            "fatigue_average_probability": None,
            "fatigue_peak_probability": None,
            "movement_mismatch_window_count": 0,
            "movement_mismatch_window_fraction": None,
            "movement_consistency": None,
            "movement_average_mismatch_probability": None,
            "movement_peak_mismatch_probability": None,
        }

    fatigue_values = fatigue_values[:count]
    movement_values = movement_values[:count]
    fatigue_count = sum(value >= 0.5 for value in fatigue_values)
    mismatch_count = sum(value >= 0.5 for value in movement_values)
    return {
        "window_count": count,
        "fatigue_pattern_window_count": fatigue_count,
        "fatigue_pattern_window_fraction": fatigue_count / count,
        "fatigue_average_probability": sum(fatigue_values) / count,
        "fatigue_peak_probability": max(fatigue_values),
        "movement_mismatch_window_count": mismatch_count,
        "movement_mismatch_window_fraction": mismatch_count / count,
        "movement_consistency": 1.0 - (mismatch_count / count),
        "movement_average_mismatch_probability": sum(movement_values) / count,
        "movement_peak_mismatch_probability": max(movement_values),
    }


def _percentage(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "Not available"


def _friendly_fatigue_label(label: str | None) -> str:
    labels = {
        "fatigued": "Fatigue pattern detected",
        "non_fatigued": "No fatigue pattern detected",
    }
    return labels.get(str(label), "Fatigue result unavailable")


def _friendly_movement_label(label: str | None) -> str:
    labels = {
        "correct": "Movement matched the model's learned pattern",
        "incorrect": "Movement may need physiotherapist review",
    }
    return labels.get(str(label), "Movement result unavailable")


def _friendly_decision(decision: str | None) -> str:
    labels = {
        "confirmed": "Agreed with the AI suggestion",
        "corrected": "Changed the AI suggestion",
        "uncertain": "Marked as unsure",
    }
    return labels.get(str(decision), "Decision unavailable")


def _format_review_time(value: str | None) -> str:
    if not value:
        return "Not available"
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    malaysia_time = timezone(timedelta(hours=8))
    local_time = timestamp.astimezone(malaysia_time)
    return local_time.strftime("%d %b %Y, %H:%M MYT")


def _build_trend_note(
    patient_id: str,
    session_id: str,
    comparison: dict[str, Any],
) -> str:
    sessions = comparison.get("sessions", [])
    selected_index = next(
        (
            index
            for index, session in enumerate(sessions)
            if session.get("session_id") == session_id
        ),
        None,
    )

    if selected_index is None:
        return "No comparison information is available for this session."
    if selected_index == 0:
        return (
            f"This is the earliest available session for patient {patient_id}, "
            "so there is no earlier session to compare it with."
        )

    current = sessions[selected_index]
    previous = sessions[selected_index - 1]
    current_windows = current.get("window_summary", {})
    previous_windows = previous.get("window_summary", {})

    if (
        int(current_windows.get("window_count", 0)) > 0
        and int(previous_windows.get("window_count", 0)) > 0
    ):
        current_fatigue_fraction = float(
            current_windows.get("fatigue_pattern_window_fraction", 0.0)
        )
        previous_fatigue_fraction = float(
            previous_windows.get("fatigue_pattern_window_fraction", 0.0)
        )
        current_movement_consistency = float(
            current_windows.get("movement_consistency", 0.0)
        )
        previous_movement_consistency = float(
            previous_windows.get("movement_consistency", 0.0)
        )
        return (
            f"Compared with {previous.get('session_id', 'the previous session')}, "
            f"fatigue-pattern windows changed from "
            f"{previous_fatigue_fraction:.1%} to "
            f"{current_fatigue_fraction:.1%}. Movement consistency changed "
            f"from {previous_movement_consistency:.1%} to "
            f"{current_movement_consistency:.1%}. These percentages describe "
            "the share of analysed windows showing each pattern; they are not "
            "clinical severity scores. The physiotherapist should interpret "
            "the changes together with the exercise and session context."
        )

    current_fatigue = float(current.get("fatigue_score", 0.0))
    previous_fatigue = float(previous.get("fatigue_score", 0.0))
    difference = current_fatigue - previous_fatigue

    if abs(difference) < 0.001:
        fatigue_change = "was similar to"
    elif difference > 0:
        fatigue_change = "was higher than"
    else:
        fatigue_change = "was lower than"

    return (
        f"Compared with {previous.get('session_id', 'the previous session')}, "
        f"the model's fatigue probability {fatigue_change} the earlier result. "
        "This difference is an experimental indicator and requires professional "
        "interpretation."
    )


def _latest_review(reviews: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not reviews:
        return None
    return next(
        (review for review in reviews if review.get("is_latest")),
        reviews[0],
    )


def generate_demonstration_summary(
    summary_context: dict[str, Any],
) -> dict[str, Any]:
    """Create a deterministic preview without calling an external LLM."""
    request = summary_context.get("request", {})
    model_output = summary_context.get("original_ai_findings", {})
    prediction = model_output.get("prediction", {})
    comparison = summary_context.get("historical_session_comparison", {})
    feedback = summary_context.get("physiotherapist_review_status", {})
    reviews = summary_context.get("physiotherapist_review_history", [])
    knowledge = summary_context.get("general_knowledge", [])
    visualisation_windows = summary_context.get(
        "visualisation_summary",
        {},
    ).get("window_predictions", [])
    session_metrics = summarize_window_evidence(
        visualisation_windows or model_output.get("window_predictions", [])
    )

    patient_id = str(request.get("patient_id", "Unknown"))
    session_id = str(request.get("session_id", "Unknown"))
    fatigue_label = (
        "fatigued" if prediction.get("fatigue_flag") else "non_fatigued"
    )
    movement_label = str(prediction.get("movement_quality", "unknown"))
    onset = prediction.get("fatigue_onset_seconds")
    review = _latest_review(reviews)

    if review:
        review_summary = {
            "status": str(feedback.get("review_status", "Review saved")),
            "reviewer": str(review.get("reviewer_id", "Not available")),
            "reviewed_at": _format_review_time(review.get("reviewed_at")),
            "fatigue_decision": _friendly_decision(
                review.get("fatigue_decision")
            ),
            "final_fatigue_result": _friendly_fatigue_label(
                review.get("final_fatigue_label")
            ),
            "movement_decision": _friendly_decision(
                review.get("movement_decision")
            ),
            "final_movement_result": _friendly_movement_label(
                review.get("final_movement_label")
            ),
            "comment": str(review.get("note") or "No comment was added."),
        }
    else:
        review_summary = {
            "status": str(
                feedback.get(
                    "review_status",
                    "Awaiting physiotherapist review",
                )
            ),
            "reviewer": "Not reviewed",
            "reviewed_at": "Not available",
            "fatigue_decision": "Not reviewed",
            "final_fatigue_result": "Not reviewed",
            "movement_decision": "Not reviewed",
            "final_movement_result": "Not reviewed",
            "comment": "No review comment is available.",
        }

    sources: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    for item in knowledge:
        source = str(item.get("source", ""))
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        sources.append(
            {
                "title": str(item.get("document_title", "Knowledge note")),
                "section": str(item.get("section_title", "Relevant section")),
                "source": source,
                "excerpt": str(item.get("text", "")),
                "links": list(item.get("links", []))[:6],
            }
        )
        if len(sources) == 3:
            break

    needs_review = (
        fatigue_label == "fatigued" or movement_label == "incorrect"
    )
    headline = (
        "The AI flagged a pattern for physiotherapist review"
        if needs_review
        else "The AI did not flag a fatigue pattern or movement mismatch"
    )
    window_count = int(session_metrics["window_count"])
    fatigue_window_count = int(
        session_metrics["fatigue_pattern_window_count"]
    )
    mismatch_window_count = int(
        session_metrics["movement_mismatch_window_count"]
    )
    fatigue_explanation = (
        f"The CNN analysed {window_count} windows, and "
        f"{fatigue_window_count} ({session_metrics['fatigue_pattern_window_fraction']:.1%}) "
        "crossed the fatigue-pattern threshold. This count describes how "
        "often the learned pattern appeared across the analysed recording; "
        "it is not a clinical severity or certainty score."
        if window_count
        else (
            "There were not enough window-level results to describe how the "
            "fatigue pattern changed across this session."
        )
    )
    movement_explanation = (
        f"The learned movement pattern was matched in "
        f"{window_count - mismatch_window_count} of {window_count} windows "
        f"({session_metrics['movement_consistency']:.1%} consistency). "
        "This percentage describes the distribution of window-level model "
        "results, not certainty that the movement was clinically correct. "
        "Any mismatch windows should be reviewed "
        "alongside the IMU trace and exercise context."
        if window_count
        else (
            "There were not enough window-level results to describe movement "
            "consistency across this session."
        )
    )

    return {
        "generation_mode": "Local session summary",
        "patient_id": patient_id,
        "session_id": session_id,
        "headline": headline,
        "fatigue": {
            "result": _friendly_fatigue_label(fatigue_label),
            "probability": _percentage(prediction.get("fatigue_score")),
            "probability_value": float(
                prediction.get("fatigue_score", 0.0)
            ),
            "onset": (
                f"Approximately {float(onset):.1f} seconds"
                if onset is not None
                else "No sustained fatigue onset was detected"
            ),
            "explanation": fatigue_explanation,
        },
        "movement": {
            "result": _friendly_movement_label(movement_label),
            "confidence": _percentage(
                prediction.get("movement_confidence")
            ),
            "confidence_value": float(
                prediction.get("movement_confidence", 0.0)
            ),
            "explanation": movement_explanation,
        },
        "session_metrics": session_metrics,
        "session_comparison": _build_trend_note(
            patient_id,
            session_id,
            comparison,
        ),
        "physiotherapist_review": review_summary,
        "knowledge_sources": sources,
        "general_explanation": (
            "The session-wide percentages are intended to make the pattern "
            "across the recording visible instead of relying on a single "
            "highest-confidence window. They should be considered with the "
            "signal plots, prior sessions, and physiotherapist review before "
            "any conclusion is made."
        ),
        "important_note": (
            "This automated summary supports professional review and is not "
            "a medical diagnosis. Final interpretation remains with the "
            "physiotherapist."
        ),
    }

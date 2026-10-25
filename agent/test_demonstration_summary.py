from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from summary_generator import generate_demonstration_summary


HERE = Path(__file__).resolve().parent


def main() -> None:
    context_path = HERE / "session_context_test.json"
    payload = json.loads(context_path.read_text(encoding="utf-8-sig"))
    summary = generate_demonstration_summary(payload["summary_context"])

    required_sections = {
        "generation_mode",
        "headline",
        "fatigue",
        "movement",
        "session_comparison",
        "physiotherapist_review",
        "important_note",
    }
    missing = required_sections.difference(summary)
    if missing:
        raise AssertionError(f"Missing summary sections: {sorted(missing)}")
    if summary["generation_mode"] != "Local session summary":
        raise AssertionError("Local summary mode is not clearly labelled.")
    if not 0.0 <= summary["fatigue"]["probability_value"] <= 1.0:
        raise AssertionError("Fatigue probability is outside the chart range.")
    if not 0.0 <= summary["movement"]["confidence_value"] <= 1.0:
        raise AssertionError("Movement confidence is outside the chart range.")

    flagged_context = deepcopy(payload["summary_context"])
    flagged_context["request"]["session_id"] = "S002"
    prediction = flagged_context["original_ai_findings"]["prediction"]
    prediction["fatigue_flag"] = True
    prediction["fatigue_score"] = 0.82
    prediction["fatigue_onset_seconds"] = 5.5
    prediction["movement_quality"] = "incorrect"
    flagged_context["physiotherapist_review_history"] = []
    flagged_context["physiotherapist_review_status"] = {
        "review_status": "Awaiting physiotherapist review"
    }
    flagged_summary = generate_demonstration_summary(flagged_context)
    if "flagged" not in flagged_summary["headline"]:
        raise AssertionError("Flagged findings were not highlighted.")
    if flagged_summary["physiotherapist_review"]["reviewer"] != "Not reviewed":
        raise AssertionError("Unreviewed state was not preserved.")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

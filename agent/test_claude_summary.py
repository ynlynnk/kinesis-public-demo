from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from claude_summary import (
    generate_general_model2_answer,
    generate_model2_summary,
    generate_role_model2_answer,
    model2_status,
)


HERE = Path(__file__).resolve().parent


def load_context() -> dict:
    payload = json.loads(
        (HERE / "session_context_test.json").read_text(encoding="utf-8-sig")
    )
    return payload["summary_context"]


def test_without_key_uses_local_mode() -> None:
    context = load_context()
    context["request"]["user_query"] = "What is the fatigue result?"
    with patch.dict(os.environ, {}, clear=True):
        summary = generate_model2_summary(context)
    assert summary["generation_mode"] == "Local session summary"
    assert summary["question"] == "What is the fatigue result?"
    assert summary["fatigue"]["probability_value"] == 0.0001
    assert summary["session_metrics"]["window_count"] == 19
    assert 0.0 <= summary["session_metrics"][
        "fatigue_pattern_window_fraction"
    ] <= 1.0
    assert 0.0 <= summary["session_metrics"]["movement_consistency"] <= 1.0
    evidence_titles = {item["title"] for item in summary["evidence_items"]}
    assert "Model 1 session findings" in evidence_titles
    assert "Window-level session evidence" in evidence_titles


def test_model_output_windows_are_used_when_visualisation_is_not_requested() -> None:
    context = load_context()
    context["visualisation_summary"]["window_predictions"] = []
    with patch.dict(os.environ, {}, clear=True):
        summary = generate_model2_summary(context)

    assert summary["session_metrics"]["window_count"] == 19
    assert summary["session_metrics"]["movement_consistency"] > 0.0


def test_gemini_is_selected_and_cannot_change_model1_facts() -> None:
    requests = []
    response_json = json.dumps(
        {
            "headline": "Session evidence is ready for review",
            "fatigue_explanation": "No fatigue pattern was flagged.",
            "movement_explanation": "Movement matched the learned pattern.",
            "session_comparison": "No earlier session was available.",
            "general_explanation": "The explanation uses verified results.",
        }
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(text=response_json)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

    fake_genai = SimpleNamespace(Client=FakeClient)
    context = load_context()
    context["request"]["user_query"] = "Explain this session."
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "KINESIS_MODEL2_PROVIDER": "gemini",
            "KINESIS_GEMINI_MODEL": "test-gemini-model",
        },
        clear=True,
    ), patch.dict(
        sys.modules,
        {
            "google": SimpleNamespace(genai=fake_genai),
            "google.genai": fake_genai,
        },
    ):
        status = model2_status()
        summary = generate_model2_summary(context)

    assert status == {
        "configured": True,
        "provider": "gemini",
        "model": "test-gemini-model",
    }
    assert summary["generation_mode"] == "AI-generated session summary"
    assert summary["model2_provider"] == "Gemini"
    assert summary["model2_verified"] is True
    assert summary["headline"] == "Session evidence is ready for review"
    assert summary["fatigue"]["probability_value"] == 0.0001
    assert summary["movement"]["confidence_value"] == 0.9998
    request = requests[0]
    assert request["model"] == "test-gemini-model"
    assert request["config"]["response_mime_type"] == "application/json"
    assert "3 to 5 complete sentences" in request["config"][
        "system_instruction"
    ]
    payload = json.loads(request["contents"])
    assert payload["question"] == "Explain this session."
    assert "patient_id" not in payload
    assert payload["selected_session_window_evidence"]["window_count"] == 19


def test_claude_changes_words_but_not_model1_facts() -> None:
    requests = []
    response_json = json.dumps(
        {
            "headline": "No pattern needs immediate review",
            "fatigue_explanation": "The model did not flag fatigue.",
            "movement_explanation": "The movement matched its learned pattern.",
            "session_comparison": "This is the first session in the sequence.",
            "general_explanation": "sEMG records electrical muscle activity.",
        }
    )

    class FakeMessages:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=response_json)]
            )

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = FakeMessages()

    context = load_context()
    context["request"]["user_query"] = "Compare fatigue and movement."
    context["retrieval_plan"] = {
        "intents": ["fatigue", "movement"],
        "mcp_tools": ["get_model1_outputs", "get_session_visualisation"],
    }
    fake_module = SimpleNamespace(Anthropic=FakeAnthropic)
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "test-key",
            "KINESIS_CLAUDE_MODEL": "test-model",
        },
        clear=True,
    ), patch.dict(sys.modules, {"anthropic": fake_module}):
        summary = generate_model2_summary(context)

    assert summary["generation_mode"] == "AI-generated session summary"
    assert summary["question"] == "Compare fatigue and movement."
    assert summary["headline"] == "No pattern needs immediate review"
    assert summary["fatigue"]["probability_value"] == 0.0001
    assert summary["movement"]["confidence_value"] == 0.9998
    payload = json.loads(requests[0]["messages"][0]["content"])
    assert payload["question"] == "Compare fatigue and movement."
    assert payload["requested_topics"] == ["fatigue", "movement"]


def test_patient_answer_uses_minimum_necessary_evidence() -> None:
    requests = []

    class FakeMessages:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=json.dumps(
                            {"answer": "Your session is awaiting review."}
                        ),
                    )
                ]
            )

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = FakeMessages()

    context = {
        "request": {
            "requesting_role": "patient",
            "user_query": "What is the status of this session?",
        },
        "role_evidence": {
            "patient_id": "P006",
            "session_id": "S001",
            "session_status": "Awaiting physiotherapist review",
            "summary": "Your exercise session was received and analysed.",
            "next_step": "Your physiotherapist will review it.",
            "safety_note": "The AI does not diagnose conditions.",
            "history": [
                {
                    "session_id": "S001",
                    "status": "Awaiting physiotherapist review",
                    "reviewed": False,
                    "analysis_date": "2026-09-10",
                }
            ],
        },
        "general_knowledge": [],
    }
    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "test-key"},
        clear=True,
    ), patch.dict(
        sys.modules,
        {"anthropic": SimpleNamespace(Anthropic=FakeAnthropic)},
    ):
        answer = generate_role_model2_answer(context)

    assert answer["generation_mode"] == "AI-generated session answer"
    assert answer["answer"] == "Your session is awaiting review."
    payload = json.loads(requests[0]["messages"][0]["content"])
    assert payload["role"] == "patient"
    assert "patient_id" not in payload["authorised_session_evidence"]
    assert "session_id" not in payload["authorised_session_evidence"]
    assert payload["authorised_session_evidence"]["session_history"] == [
        {
            "order": 1,
            "status": "Awaiting physiotherapist review",
            "reviewed": False,
            "analysis_date": "2026-09-10",
        }
    ]
    visible_evidence = json.dumps(answer["evidence_items"])
    assert "Reviewed session summary" in visible_evidence
    assert "P006" not in visible_evidence
    assert "S001" not in visible_evidence


def test_gemini_patient_answer_uses_minimum_necessary_evidence() -> None:
    requests = []

    class FakeModels:
        def generate_content(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                text=json.dumps(
                    {"answer": "Your session is waiting for review."}
                )
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

    context = {
        "request": {
            "requesting_role": "patient",
            "user_query": "What happens next?",
        },
        "role_evidence": {
            "patient_id": "P006",
            "session_id": "S001",
            "session_status": "Awaiting physiotherapist review",
            "summary": "Your exercise session was received and analysed.",
            "next_step": "Your physiotherapist will review it.",
            "safety_note": "The AI does not diagnose conditions.",
            "history": [
                {
                    "session_id": "S001",
                    "status": "Awaiting physiotherapist review",
                    "reviewed": False,
                    "analysis_date": "2026-09-10",
                }
            ],
        },
        "general_knowledge": [],
    }
    fake_genai = SimpleNamespace(Client=FakeClient)
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "KINESIS_MODEL2_PROVIDER": "gemini",
        },
        clear=True,
    ), patch.dict(
        sys.modules,
        {
            "google": SimpleNamespace(genai=fake_genai),
            "google.genai": fake_genai,
        },
    ):
        answer = generate_role_model2_answer(context)

    assert answer["generation_mode"] == "AI-generated session answer"
    assert answer["model2_provider"] == "Gemini"
    assert answer["model2_verified"] is True
    assert answer["answer"] == "Your session is waiting for review."
    payload = json.loads(requests[0]["contents"])
    assert payload["role"] == "patient"
    assert "patient_id" not in payload["authorised_session_evidence"]
    assert "session_id" not in payload["authorised_session_evidence"]
    assert payload["authorised_session_evidence"]["session_history"][0][
        "reviewed"
    ] is False


def test_gemini_failure_is_clearly_marked_as_local_fallback() -> None:
    class FailingModels:
        def generate_content(self, **_kwargs):
            raise RuntimeError("Provider unavailable")

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FailingModels()

    fake_genai = SimpleNamespace(Client=FakeClient)
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "KINESIS_MODEL2_PROVIDER": "gemini",
            "KINESIS_GEMINI_MODEL": "test-gemini-model",
        },
        clear=True,
    ), patch.dict(
        sys.modules,
        {
            "google": SimpleNamespace(genai=fake_genai),
            "google.genai": fake_genai,
        },
    ):
        summary = generate_model2_summary(load_context())

    assert summary["generation_mode"] == "Local session summary"
    assert summary["model2_provider"] == "Gemini"
    assert summary["model2_model"] == "test-gemini-model"
    assert summary["model2_verified"] is False
    assert summary["model2_error"] == "RuntimeError"


def test_unrelated_patient_question_skips_external_ai() -> None:
    context = {
        "request": {
            "requesting_role": "patient",
            "user_query": "What will the weather be tomorrow?",
        },
        "retrieval_plan": {"in_scope": False, "intents": []},
        "role_evidence": {
            "session_status": "Physiotherapist review completed",
            "summary": "Session completed.",
            "next_step": "Follow the current plan.",
            "safety_note": "This is not a diagnosis.",
            "history": [],
        },
        "general_knowledge": [],
    }
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "KINESIS_MODEL2_PROVIDER": "gemini",
        },
        clear=True,
    ):
        answer = generate_role_model2_answer(context)

    assert answer["generation_mode"] == "Scope guidance"
    assert answer["model2_verified"] is False
    assert answer["model2_skipped_reason"] == "out_of_scope"
    assert "selected rehabilitation session" in answer["answer"]
    assert answer["evidence_items"][0]["source"] == "Model 2 rule check"
    assert "no patient evidence" in answer["evidence_items"][0][
        "finding"
    ].lower()


def test_general_question_uses_only_approved_knowledge() -> None:
    requests = []

    class FakeModels:
        def generate_content(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "answer": (
                            "Surface EMG records electrical activity associated "
                            "with muscle activation. It does not diagnose a "
                            "condition or directly measure strength."
                        )
                    }
                )
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

    context = {
        "request": {
            "requesting_role": "patient",
            "user_query": "What does surface EMG measure?",
        },
        "retrieval_plan": {
            "in_scope": True,
            "intents": ["general_rehabilitation_knowledge"],
        },
        "general_knowledge": [
            {
                "source": "knowledge/emg/semg_basics.md",
                "document_title": "Surface EMG basics",
                "section_title": "What surface EMG measures",
                "text": "Surface EMG records muscle-associated electrical activity.",
                "links": ["https://pubmed.ncbi.nlm.nih.gov/22169134/"],
            }
        ],
    }
    fake_genai = SimpleNamespace(Client=FakeClient)
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "KINESIS_MODEL2_PROVIDER": "gemini",
        },
        clear=True,
    ), patch.dict(
        sys.modules,
        {
            "google": SimpleNamespace(genai=fake_genai),
            "google.genai": fake_genai,
        },
    ):
        answer = generate_general_model2_answer(context)

    assert answer["answer_scope"] == "general_knowledge"
    assert answer["model2_verified"] is True
    assert answer["audience_role"] == "patient"
    assert answer["evidence_items"][0]["reference"] == (
        "Approved KINESIS information"
    )
    assert "muscle-associated electrical activity" in answer[
        "evidence_items"
    ][0]["excerpt"]
    assert answer["evidence_items"][0]["links"] == [
        "https://pubmed.ncbi.nlm.nih.gov/22169134/"
    ]
    payload = json.loads(requests[0]["contents"])
    assert payload["session_selected"] is False
    assert payload["patient_data_used"] is False
    assert "patient_id" not in json.dumps(payload)
    assert "session_id" not in json.dumps(payload)


def test_unrelated_general_question_skips_external_ai() -> None:
    context = {
        "request": {
            "requesting_role": "patient",
            "user_query": "What will the weather be tomorrow?",
        },
        "retrieval_plan": {
            "in_scope": False,
            "intents": ["general_rehabilitation_knowledge"],
        },
        "general_knowledge": [],
    }
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "KINESIS_MODEL2_PROVIDER": "gemini",
        },
        clear=True,
    ):
        answer = generate_general_model2_answer(context)

    assert answer["generation_mode"] == "Scope guidance"
    assert answer["model2_verified"] is False
    assert answer["model2_skipped_reason"] == "out_of_scope"
    assert "no patient record" in answer["evidence_items"][0][
        "finding"
    ].lower()


if __name__ == "__main__":
    test_without_key_uses_local_mode()
    test_model_output_windows_are_used_when_visualisation_is_not_requested()
    test_gemini_is_selected_and_cannot_change_model1_facts()
    test_claude_changes_words_but_not_model1_facts()
    test_patient_answer_uses_minimum_necessary_evidence()
    test_gemini_patient_answer_uses_minimum_necessary_evidence()
    test_gemini_failure_is_clearly_marked_as_local_fallback()
    test_unrelated_patient_question_skips_external_ai()
    test_general_question_uses_only_approved_knowledge()
    test_unrelated_general_question_skips_external_ai()
    print("Model 2 provider and safety tests passed.")

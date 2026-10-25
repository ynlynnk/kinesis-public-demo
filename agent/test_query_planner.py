from __future__ import annotations

import sys
from types import SimpleNamespace


# The planner is pure Python. These lightweight substitutes keep this unit
# check independent from the running MCP service and the retrieval packages.
sys.modules.setdefault("mcp", SimpleNamespace(Client=object))
sys.modules.setdefault(
    "agent.knowledge_retriever",
    SimpleNamespace(retrieve_knowledge=lambda **_kwargs: []),
)

from agent.agent import plan_session_query


def test_fatigue_progress_question() -> None:
    plan = plan_session_query(
        "How has fatigue changed from the previous session?"
    )
    assert plan["in_scope"] is True
    assert plan["intents"] == ["fatigue", "progress"]
    assert "get_model1_outputs" in plan["mcp_tools"]
    assert "compare_session_trends" in plan["mcp_tools"]
    assert "get_session_visualisation" in plan["mcp_tools"]
    assert "get_review_history" not in plan["mcp_tools"]


def test_movement_review_question() -> None:
    plan = plan_session_query("Review the movement result")
    assert plan["intents"] == ["movement", "review"]
    assert "get_session_visualisation" in plan["mcp_tools"]
    assert "get_feedback_status" in plan["mcp_tools"]
    assert "get_review_history" in plan["mcp_tools"]
    assert "compare_session_trends" not in plan["mcp_tools"]


def test_overall_summary_requests_all_evidence() -> None:
    plan = plan_session_query("Summarise the overall session")
    assert plan["intents"] == [
        "fatigue",
        "movement",
        "progress",
        "review",
    ]
    assert plan["mcp_tools"] == [
        "list_available_sessions",
        "get_model1_outputs",
        "compare_session_trends",
        "get_session_visualisation",
        "get_feedback_status",
        "get_review_history",
    ]


def test_unrelated_question_is_outside_scope() -> None:
    plan = plan_session_query("What will the weather be tomorrow?")
    assert plan["in_scope"] is False
    assert plan["intents"] == []
    assert plan["mcp_tools"] == [
        "list_available_sessions",
        "get_model1_outputs",
    ]


if __name__ == "__main__":
    test_fatigue_progress_question()
    test_movement_review_question()
    test_overall_summary_requests_all_evidence()
    test_unrelated_question_is_outside_scope()
    print("Model 2 query planner tests passed.")

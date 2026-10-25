from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp import Client

try:
    from .claude_summary import (
        generate_general_model2_answer,
        generate_model2_summary,
        generate_role_model2_answer,
    )
    from .summary_generator import summarize_window_evidence
except ImportError:
    from claude_summary import (
        generate_general_model2_answer,
        generate_model2_summary,
        generate_role_model2_answer,
    )
    from summary_generator import summarize_window_evidence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = (
    PROJECT_ROOT
    / "agent"
    / "skills"
    / "kinesis-session-analysis"
    / "SKILL.md"
)

MCP_SERVER_URL = os.getenv(
    "KINESIS_MCP_URL",
    "http://127.0.0.1:8000/mcp",
)


def retrieve_approved_knowledge(query: str, top_k: int) -> list[dict[str, Any]]:
    """Load the optional document retriever only when analysis needs it."""

    try:
        from .knowledge_retriever import retrieve_knowledge
    except ImportError:
        from knowledge_retriever import retrieve_knowledge
    return retrieve_knowledge(query=query, top_k=top_k)


def load_skill() -> str:
    if not SKILL_PATH.exists():
        raise FileNotFoundError(
            f"KINESIS skill was not found: {SKILL_PATH}"
        )

    return SKILL_PATH.read_text(
        encoding="utf-8-sig"
    ).strip()


async def call_mcp_tool(
    client: Client,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    result = await client.call_tool(
        tool_name,
        arguments,
    )

    if result.is_error:
        raise RuntimeError(
            f"{tool_name} returned an error: {result.content}"
        )

    if result.structured_content is None:
        raise RuntimeError(
            f"{tool_name} did not return structured data."
        )

    return result.structured_content


def create_knowledge_query(
    model_output: dict[str, Any],
    user_query: str = "",
) -> str:
    prediction = model_output.get("prediction", {})

    fatigue_flag = bool(
        prediction.get("fatigue_flag", False)
    )
    movement_quality = prediction.get(
        "movement_quality",
        "unknown",
    )

    topics = [
        user_query.strip(),
        "surface EMG interpretation",
        "IMU movement analysis",
        "rehabilitation session monitoring",
    ]

    if fatigue_flag:
        topics.extend(
            [
                "fatigue pattern",
                "fatigue onset",
                "fatigue indicators",
            ]
        )
    else:
        topics.append(
            "no sustained fatigue pattern detected"
        )

    if movement_quality == "incorrect":
        topics.extend(
            [
                "movement may need correction",
                "movement quality review",
            ]
        )
    else:
        topics.append(
            "movement matched learned pattern"
        )

    return " ".join(topic for topic in topics if topic)


def plan_session_query(user_query: str) -> dict[str, Any]:
    """Map a plain-language question to the evidence needed from MCP."""

    clean_query = " ".join(user_query.split()).strip()
    lowered = clean_query.casefold()
    intents: list[str] = []
    summary_requested = any(
        phrase in lowered
        for phrase in ("summary", "summarise", "summarize", "overall")
    )

    if any(word in lowered for word in ("fatigue", "tired", "onset", "semg")):
        intents.append("fatigue")
    if any(
        word in lowered
        for word in ("movement", "motion", "angle", "imu", "pronation", "supination")
    ):
        intents.append("movement")
    if any(
        word in lowered
        for word in ("progress", "trend", "previous", "history", "compare", "change")
    ):
        intents.append("progress")
    if any(
        word in lowered
        for word in ("review", "physiotherapist", "decision", "comment")
    ):
        intents.append("review")

    general_session_terms = (
        "session",
        "exercise",
        "result",
        "value",
        "mean",
        "finding",
        "indicator",
        "score",
        "status",
        "next step",
        "continue",
        "follow-up",
        "follow up",
        "feedback",
        "care plan",
        "rehabilitation",
        "performance",
        "consistent",
        "steady",
        "concern",
    )
    in_scope = bool(
        not clean_query
        or intents
        or summary_requested
        or any(term in lowered for term in general_session_terms)
    )

    if summary_requested or (in_scope and not intents):
        intents = ["fatigue", "movement", "progress", "review"]

    tools = ["list_available_sessions", "get_model1_outputs"]
    if "progress" in intents:
        tools.append("compare_session_trends")
    if "fatigue" in intents or "movement" in intents:
        tools.append("get_session_visualisation")
    if "review" in intents:
        tools.extend(["get_feedback_status", "get_review_history"])

    return {
        "user_query": clean_query or "Summarise this rehabilitation session.",
        "in_scope": in_scope,
        "intents": intents,
        "mcp_tools": tools,
    }


def plan_general_query(user_query: str) -> dict[str, Any]:
    """Check whether a question fits the approved rehabilitation library."""

    clean_query = " ".join(user_query.split()).strip()
    lowered = clean_query.casefold()
    rehabilitation_terms = (
        "rehabilitation",
        "physiotherapy",
        "physiotherapist",
        "exercise",
        "muscle",
        "fatigue",
        "semg",
        "surface emg",
        "electromyography",
        "imu",
        "movement",
        "motion",
        "sensor",
        "activation",
        "kinematic",
        "wearable",
        "recovery",
        "session monitoring",
        "model 1",
        "cnn",
        "kinesis",
        "diagnosis",
    )
    return {
        "user_query": clean_query,
        "in_scope": bool(
            clean_query and any(term in lowered for term in rehabilitation_terms)
        ),
        "intents": ["general_rehabilitation_knowledge"],
        "mcp_tools": [],
    }


def build_general_question_context(
    requesting_role: str,
    user_query: str,
) -> dict[str, Any]:
    """Answer without loading a patient or rehabilitation session."""

    if requesting_role not in {"patient", "clinician", "physiotherapist"}:
        raise PermissionError("This role is not available for general questions.")

    retrieval_plan = plan_general_query(user_query)
    knowledge_context = []
    if retrieval_plan["in_scope"]:
        knowledge_context = retrieve_approved_knowledge(
            query=retrieval_plan["user_query"],
            top_k=4,
        )

    summary_context = {
        "workflow_instructions": load_skill(),
        "request": {
            "requesting_role": requesting_role,
            "user_query": retrieval_plan["user_query"],
        },
        "retrieval_plan": retrieval_plan,
        "general_knowledge": knowledge_context,
    }
    return {
        "automatic_summary": generate_general_model2_answer(summary_context),
        "knowledge_result_count": len(knowledge_context),
    }


def filter_selected_reviews(
    review_history: dict[str, Any],
    patient_id: str,
    session_id: str,
) -> list[dict[str, Any]]:
    selected_reviews = []

    for review in review_history.get("reviews", []):
        if (
            review.get("patient_id") == patient_id
            and review.get("session_id") == session_id
        ):
            selected_reviews.append(review)

    return selected_reviews


async def build_session_context(
    patient_id: str,
    session_id: str,
    user_query: str = "Summarise this rehabilitation session.",
) -> dict[str, Any]:
    requesting_role = "physiotherapist"
    retrieval_plan = plan_session_query(user_query)

    async with Client(
        MCP_SERVER_URL,
        raise_exceptions=True,
    ) as client:
        catalog = await call_mcp_tool(
            client,
            "list_available_sessions",
            {
                "requesting_role": requesting_role,
            },
        )

        authorised_sessions = {
            (
                item.get("patient_id"),
                item.get("session_id"),
            )
            for item in catalog.get("sessions", [])
        }

        if (
            patient_id,
            session_id,
        ) not in authorised_sessions:
            raise PermissionError(
                "The selected patient and session are not "
                "available to this physiotherapist workflow."
            )

        model_output = await call_mcp_tool(
            client,
            "get_model1_outputs",
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "requesting_role": requesting_role,
            },
        )

        if "compare_session_trends" in retrieval_plan["mcp_tools"]:
            trend_data = await call_mcp_tool(
                client,
                "compare_session_trends",
                {
                    "patient_id": patient_id,
                    "requesting_role": requesting_role,
                },
            )
            for session in trend_data.get("sessions", []):
                compared_session_id = str(session.get("session_id", ""))
                if not compared_session_id:
                    continue
                compared_output = (
                    model_output
                    if compared_session_id == session_id
                    else await call_mcp_tool(
                        client,
                        "get_model1_outputs",
                        {
                            "patient_id": patient_id,
                            "session_id": compared_session_id,
                            "requesting_role": requesting_role,
                        },
                    )
                )
                session["window_summary"] = summarize_window_evidence(
                    compared_output.get("window_predictions", [])
                )
        else:
            trend_data = {"sessions": []}

        if "get_session_visualisation" in retrieval_plan["mcp_tools"]:
            visualisation_data = await call_mcp_tool(
                client,
                "get_session_visualisation",
                {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "requesting_role": requesting_role,
                },
            )
        else:
            visualisation_data = {
                "window_predictions": [],
                "signal_overlay": [],
                "data_note": "Not required for this question.",
            }

        if "get_feedback_status" in retrieval_plan["mcp_tools"]:
            feedback_status = await call_mcp_tool(
                client,
                "get_feedback_status",
                {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "requesting_role": requesting_role,
                },
            )
        else:
            feedback_status = {
                "review_status": "Not requested for this question"
            }

        review_history_note = None

        if "get_review_history" in retrieval_plan["mcp_tools"]:
            try:
                review_history = await call_mcp_tool(
                    client,
                    "get_review_history",
                    {
                        "requesting_role": requesting_role,
                    },
                )
            except Exception as error:
                review_history = {"reviews": []}
                review_history_note = (
                    "Detailed review history was unavailable: "
                    f"{error}"
                )
        else:
            review_history = {"reviews": []}

    knowledge_query = create_knowledge_query(
        model_output,
        retrieval_plan["user_query"],
    )
    knowledge_context = retrieve_approved_knowledge(
        query=knowledge_query,
        top_k=5,
    )

    selected_reviews = filter_selected_reviews(
        review_history,
        patient_id,
        session_id,
    )

    summary_context = {
        "workflow_instructions": load_skill(),
        "request": {
            "patient_id": patient_id,
            "session_id": session_id,
            "requesting_role": requesting_role,
            "user_query": retrieval_plan["user_query"],
        },
        "retrieval_plan": retrieval_plan,
        "original_ai_findings": model_output,
        "historical_session_comparison": trend_data,
        "physiotherapist_review_status": feedback_status,
        "physiotherapist_review_history": selected_reviews,
        "review_history_note": review_history_note,
        "visualisation_summary": {
            "window_predictions": visualisation_data.get(
                "window_predictions",
                [],
            ),
            "signal_point_count": len(
                visualisation_data.get(
                    "signal_overlay",
                    [],
                )
            ),
            "data_note": visualisation_data.get(
                "data_note"
            ),
        },
        "general_knowledge": knowledge_context,
        "generation_requirements": [
            "Separate AI findings from physiotherapist review.",
            "Use plain and non-diagnostic language.",
            "Do not invent missing measurements.",
            "Do not recommend treatment changes.",
            "Do not describe a training record as a live patient recording.",
            "Cite each knowledge document that is used.",
        ],
    }

    return {
        "summary_context": summary_context,
        "visualisation_data": visualisation_data,
        "automatic_summary": generate_model2_summary(
            summary_context
        ),
    }


async def build_role_session_context(
    patient_id: str,
    session_id: str,
    requesting_role: str,
    requesting_patient_id: str | None,
    user_query: str,
) -> dict[str, Any]:
    """Build a minimum-necessary patient or clinician answer context."""

    if requesting_role not in {"patient", "clinician"}:
        raise PermissionError("This role-specific workflow is not available.")

    retrieval_plan = plan_session_query(user_query)
    catalog_arguments: dict[str, Any] = {
        "requesting_role": requesting_role,
    }
    if requesting_role == "patient":
        catalog_arguments["requesting_patient_id"] = requesting_patient_id

    async with Client(MCP_SERVER_URL, raise_exceptions=True) as client:
        catalog = await call_mcp_tool(
            client,
            "list_available_sessions",
            catalog_arguments,
        )
        authorised_sessions = {
            (item.get("patient_id"), item.get("session_id"))
            for item in catalog.get("sessions", [])
        }
        if (patient_id, session_id) not in authorised_sessions:
            raise PermissionError(
                "The selected patient and session are not available to this "
                "user role."
            )

        if requesting_role == "patient":
            primary_tool = "get_patient_summary"
            primary_arguments = {
                "patient_id": patient_id,
                "session_id": session_id,
                "requesting_role": requesting_role,
                "requesting_patient_id": requesting_patient_id,
            }
        else:
            primary_tool = "get_clinician_summary"
            primary_arguments = {
                "patient_id": patient_id,
                "session_id": session_id,
                "requesting_role": requesting_role,
            }

        role_evidence = await call_mcp_tool(
            client,
            primary_tool,
            primary_arguments,
        )

        trend_data = {"sessions": []}
        if (
            requesting_role == "clinician"
            and "compare_session_trends" in retrieval_plan["mcp_tools"]
        ):
            trend_data = await call_mcp_tool(
                client,
                "compare_session_trends",
                {
                    "patient_id": patient_id,
                    "requesting_role": requesting_role,
                },
            )

    knowledge_context = retrieve_approved_knowledge(
        query=retrieval_plan["user_query"],
        top_k=3,
    )
    role_evidence["authorised_session_comparison"] = trend_data
    summary_context = {
        "workflow_instructions": load_skill(),
        "request": {
            "requesting_role": requesting_role,
            "user_query": retrieval_plan["user_query"],
        },
        "retrieval_plan": retrieval_plan,
        "role_evidence": role_evidence,
        "historical_session_comparison": trend_data,
        "general_knowledge": knowledge_context,
    }
    return {
        "primary_tool": primary_tool,
        "role_evidence": role_evidence,
        "automatic_summary": generate_role_model2_answer(summary_context),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the automatic KINESIS session-analysis "
            "context for a physiotherapist."
        )
    )
    parser.add_argument("patient_id")
    parser.add_argument("session_id")
    arguments = parser.parse_args()

    result = asyncio.run(
        build_session_context(
            patient_id=arguments.patient_id,
            session_id=arguments.session_id,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

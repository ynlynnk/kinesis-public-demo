"""Floating chat agent: cross-session tools, conversation memory and de-identification."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

# agent/test_query_planner.py installs a stub retriever in sys.modules; use the real one.
if not hasattr(sys.modules.get("agent.knowledge_retriever"), "KNOWLEDGE_ROOT"):
    for name in ("agent.knowledge_retriever", "agent.kinesis_session_rag", "agent.kinesis_progress_agent",
                 "agent.kinesis_chat_agent"):
        sys.modules.pop(name, None)

from agent import kinesis_chat_agent as ca  # noqa: E402
from agent import kinesis_progress_agent as pa  # noqa: E402
from test_kinesis_progress import _report  # noqa: E402

PATIENT = "Jane_Doe"


class ChatAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdir = Path(self.tmp.name) / PATIENT
        plan = (("20261001_100000", _report(5, 12, 85, 20, {"EARLY_DOWN": 7})),
                ("20261004_100000", _report(11, 12, 95, 15)),
                ("20261008_100000", _report(8, 12, 90, 30, {"EARLY_DOWN": 2, "FOREARM_LIFT": 2})),
                ("20261011_100000", _report(4, 12, 80, 60, {"FOREARM_LIFT": 8})))
        for stamp, report in plan:
            path = self.pdir / f"se{stamp}"
            path.mkdir(parents=True)
            (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
        self.viewed = "se20261008_100000"     # S3: the agent must still reach S4 and S1

    def tearDown(self):
        self.tmp.cleanup()

    def test_tools_cover_the_whole_record(self):
        history = pa.load_history(self.pdir, "se20261011_100000")
        tools = ca.make_chat_tools(self.pdir, history, "S3")
        self.assertEqual(len(tools["list_session_history"]()["sessions"]), 4)
        self.assertEqual(tools["get_session_detail"]("S4")["session"], "S4")
        self.assertEqual(tools["get_progress_overview"]()["this_session"], "S3")
        self.assertEqual(tools["get_progress_overview"]("S4")["this_session"], "S4")
        fault = tools["find_fault"]("forearm_lift")
        self.assertEqual([f["session"] for f in fault["sessions_with_fault"]], ["S3", "S4"])
        self.assertEqual(fault["sessions_without"], ["S1", "S2"])
        self.assertEqual([r["label"] for r in tools["compare_sessions"](["S2", "S4"])["sessions"]], ["S2", "S4"])
        self.assertIn("error", tools["get_risk_flags"]("S9"))

    def test_local_answer_reaches_other_sessions(self):
        with mock.patch.object(ca, "_model2_provider", return_value=("gemini", None, "m")):
            result = ca.chat_answer(self.pdir, self.viewed, "Has FOREARM_LIFT happened before? Compare with the best.")
        self.assertTrue(result["answer"].startswith("S3 (day 7, Bicep curls): 8/12 good reps (66.7%), "
                                                    "against 66.7% over the previous 2 session(s). Trend: stable · "
                                                    "risk tag: MONITOR."))
        findings = {f["label"]: f["detail"] for f in result["findings"]}
        self.assertLessEqual(len(findings), 4)
        self.assertEqual(findings["FOREARM_LIFT"], "S3 16.7% · S4 66.7% of attempts")
        self.assertEqual(findings["Best / worst"], "best S2 91.7% · worst S4 33.3%")
        self.assertEqual([r["Session"] for r in result["table"]], ["S2", "S3 (viewed)", "S4"])
        self.assertEqual(result["table"][0]["Good"], "11/12 (92%)")   # numbers come from the record
        self.assertEqual(result["session_labels"]["S4"], "se20261011_100000")

    def test_claude_agent_uses_history_and_conversation(self):
        sent = []

        class Messages:
            calls = 0

            def create(self, **kw):
                sent.append(json.dumps(kw["messages"], default=str))
                Messages.calls += 1
                if Messages.calls == 1:
                    return SimpleNamespace(content=[
                        SimpleNamespace(type="tool_use", id="a", name="find_fault", input={"code": "FOREARM_LIFT"}),
                        SimpleNamespace(type="tool_use", id="b", name="get_session_detail", input={"session": "S4"})])
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps({
                    "answer": "FOREARM_LIFT rose from 16.7% in S3 to 66.7% (67%) in S4, peak 78.5°.",
                    "findings": [{"label": "FOREARM_LIFT", "detail": "S3 16.7% -> S4 66.7%"}] * 6,
                    "table_sessions": ["s4", "S3", "S99"], "check": "Watch elbow position.",
                    "sessions_used": ["S3", "S4"], "knowledge_ids_used": []}))])

        earlier = [{"question": "How is the patient doing?", "answer": "Improving until S2.", "key_points": []}]
        fake = SimpleNamespace(Anthropic=lambda **_: SimpleNamespace(messages=Messages()))
        with mock.patch.object(ca, "_model2_provider", return_value=("claude", "k", "claude-x")), \
                mock.patch.dict(sys.modules, {"anthropic": fake}):
            result = ca.chat_answer(self.pdir, self.viewed, "and later?", earlier)
        self.assertEqual(result["generation_mode"], "AI agent (claude: claude-x, 2 tool calls)")
        self.assertEqual(result["sessions_used"], ["S3", "S4"])
        self.assertEqual(len(result["findings"]), 4)                  # capped
        self.assertEqual([r["Session"] for r in result["table"]], ["S3 (viewed)", "S4"])   # S99 dropped
        self.assertEqual(result["table"][1]["Good"], "4/12 (33%)")    # filled from the record, not the model
        self.assertEqual(result["check"], "Watch elbow position.")
        self.assertEqual(result["unverified_numbers"], ["78.5"])      # S4's peak is 80.0: invented, flagged
        first = json.loads(sent[0])
        self.assertEqual(first[0], {"role": "user", "content": "How is the patient doing?"})
        self.assertEqual(first[1]["role"], "assistant")
        self.assertIn("viewing S3", first[2]["content"])
        self.assertIn("66.7", sent[1])                # S4's data went back to the model
        everything = " ".join(sent)
        for secret in (PATIENT, "se2026", "2026-10", "20261011"):
            self.assertNotIn(secret, everything)

    def test_provider_failure_falls_back(self):
        with mock.patch.object(ca, "_model2_provider", return_value=("gemini", "k", "m")), \
                mock.patch.object(pa, "_run_gemini", side_effect=RuntimeError("429 quota")):
            result = ca.chat_answer(self.pdir, self.viewed, "Is fatigue building up?")
        self.assertEqual(result["generation_mode"], "Local evidence summary")
        self.assertIn("quota", result["provider_error"])
        fatigue = next(f["detail"] for f in result["findings"] if f["label"] == "Fatigue (end)")
        self.assertEqual(fatigue, "S1 20.0 → S2 15.0 → S3 30.0 → S4 60.0")


if __name__ == "__main__":
    unittest.main()

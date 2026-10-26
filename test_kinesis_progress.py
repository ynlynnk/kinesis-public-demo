"""AI progress report: rule-based ground truth, risk flags and the function-calling loops."""

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
    for name in ("agent.knowledge_retriever", "agent.kinesis_session_rag", "agent.kinesis_progress_agent"):
        sys.modules.pop(name, None)

from agent import kinesis_progress_agent as pa  # noqa: E402

PATIENT = "Jane_Doe"


def _report(good: int, attempts: int, angle: float, fatigue: float | None, faults=None, planned=12):
    faults = faults or {}
    fault_list = [c for c, n in faults.items() for _ in range(n)]
    reps = []
    for i in range(attempts):
        counted = i < good
        reps.append({"index": i + 1, "counted": counted, "faults": [] if counted else [fault_list[i - good]]
                     if i - good < len(fault_list) else ["EARLY_DOWN"], "peak_deg": angle,
                     "muscle_pct_rvc": 60.0, "lift_s": 1.5, "hold_s": 1.0, "lower_s": 2.0})
    return {
        "session": {"exercise": "Bicep curls", "target_deg": 90.0, "personal_rom_deg": 104.0,
                    "planned": {"sets": 1, "reps_per_set": planned}},
        "totals": {"attempts": attempts, "good_reps": good, "not_counted": attempts - good,
                   "good_rate": good / attempts if attempts else None, "avg_peak_angle_deg": angle},
        "faults": [{"code": c, "tip": "t", "reps": n} for c, n in faults.items()],
        "fatigue": ({"final_score": fatigue, "zone": "red" if fatigue > 70 else "amber" if fatigue >= 40
                     else "green", "source": "counted",
                     "trend": [{"rep": 1, "score": 0.0}, {"rep": 2, "score": fatigue}]}
                    if fatigue is not None else {"source": None}),
        "reps": reps,
    }


class GroundTruthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdir = Path(self.tmp.name) / PATIENT

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, stamp: str, report: dict, events: str = "") -> str:
        sid = f"se{stamp}"
        path = self.pdir / sid
        path.mkdir(parents=True)
        (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
        if events:
            (path / "events.csv").write_text("t,event,arg\n" + events, encoding="utf-8")
        return sid

    def test_improving_patient_with_timeframe_and_no_flags(self):
        self.add("20261001_100000", _report(5, 12, 85, 20))
        self.add("20261004_100000", _report(7, 12, 88, 18))
        self.add("20261008_100000", _report(9, 12, 91, 15))
        sid = self.add("20261011_100000", _report(11, 12, 94, 12))
        self.add("20261015_100000", _report(2, 12, 60, 90))          # later sessions are ignored
        truth = pa.ground_truth(self.pdir, sid)
        ov = truth["overview"]
        self.assertEqual((ov["session_number"], ov["sessions_to_date"]), (4, 4))
        self.assertEqual((ov["rehab_day"], ov["rehab_span_days"], ov["days_since_previous"]), (10, 10, 3))
        self.assertEqual(ov["sessions_per_week"], 2.8)
        self.assertEqual(ov["metrics"]["good_rate_pct"]["direction_recent"], "improving")
        self.assertEqual(ov["rule_trajectory"], "improving")
        self.assertEqual(truth["risk"]["risk_tag"], "NO FLAGS")

    def test_declining_session_is_tagged_high_risk(self):
        for i, day in enumerate(("01", "04", "08")):
            self.add(f"202610{day}_100000", _report(10, 12, 92, 15 + 10 * i))
        sid = self.add("20261011_100000", _report(3, 12, 78, 75, {"NO_MUSCLE": 5, "FOREARM_LIFT": 4}),
                       events="10.0,session,pause:emergency\n")
        risk = pa.ground_truth(self.pdir, sid)["risk"]
        codes = {f["code"]: f["level"] for f in risk["flags"]}
        self.assertEqual(codes["QUALITY_DROP"], "high")
        self.assertEqual(codes["ROM_LOSS"], "high")
        self.assertEqual(codes["FATIGUE_HIGH"], "high")
        self.assertEqual(codes["FATIGUE_RISING"], "medium")
        self.assertEqual(codes["EMERGENCY_STOP"], "high")
        self.assertEqual(codes["LOW_MUSCLE_USE"], "medium")
        self.assertEqual(codes["COMPENSATION"], "medium")
        self.assertEqual(risk["risk_tag"], "HIGH RISK")
        self.assertTrue(risk["at_risk"])

    def test_two_medium_flags_make_at_risk_and_gaps_are_low(self):
        self.add("20261001_100000", _report(10, 12, 92, 20))
        sid = self.add("20261012_100000", _report(10, 12, 92, 50, {}, planned=40))
        risk = pa.ground_truth(self.pdir, sid)["risk"]
        codes = {f["code"]: f["level"] for f in risk["flags"]}
        self.assertEqual(codes, {"FATIGUE_HIGH": "medium", "SESSION_CUT_SHORT": "medium", "ADHERENCE_GAP": "low"})
        self.assertEqual(risk["risk_tag"], "AT RISK")

    def test_first_session_has_insufficient_data(self):
        sid = self.add("20261001_100000", _report(6, 12, 90, 10))
        truth = pa.ground_truth(self.pdir, sid)
        self.assertEqual(truth["overview"]["rule_trajectory"], "insufficient_data")
        report = pa.local_report(truth)
        self.assertEqual(report["trajectory"], "insufficient_data")


class FullReportTest(unittest.TestCase):
    def test_record_and_ai_summary_are_added_and_escaped(self):
        import kinesis_reports as kr

        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / PATIENT
            for stamp, good in (("20261001_100000", 6), ("20261008_100000", 10)):
                (pdir / f"se{stamp}").mkdir(parents=True)
                (pdir / f"se{stamp}" / "report.json").write_text(json.dumps(_report(good, 12, 90, 20)),
                                                                 encoding="utf-8")
            truth = pa.ground_truth(pdir, "se20261008_100000")
        app_html = "<html><body><main><h1>Session report</h1><div class='tiles'></div><h2>Whole session</h2></main>"
        ai = {"generation_mode": "AI reasoning (gemini)", "generated_at": "2026-10-26T10:00:00",
              "ai": {"headline": "Improving <script>alert(1)</script>", "trajectory": "improving",
                     "confidence": "medium", "reasoning": "Good-rep rate 50.0 -> 83.3.",
                     "metric_progress": [{"metric": "Good-rep rate", "direction": "improving", "evidence": "+33"}],
                     "fatigue_assessment": "Stable.", "additional_concerns": [],
                     "suggested_review_actions": ["Progress the load."]}}
        full = kr.full_report_html(app_html, truth, ai)
        self.assertLess(full.index("Rehabilitation record"), full.index("<h2>Whole session"))
        self.assertIn("Session 2 of 2", full)
        self.assertIn("10/12 (83%)", full)
        self.assertLess(full.index("Progress by metric"), full.index("AI progress summary"))
        self.assertIn("aria-label='Good-repetition rate (%) by session'", full)
        self.assertIn("improving</b> (+33.3 vs recent sessions", full)
        self.assertIn('<tr class="kin-cur"><td>S2 (this)</td><td>08 Oct 2026 10:00</td>', full)
        self.assertIn('<b class="tag">MONITOR</b>', full)            # 7-day gap: low flag
        self.assertIn("Long gap since the previous session", full)
        self.assertIn("Progress the load.", full)
        self.assertNotIn("<script>", full)
        self.assertIn("&lt;script&gt;", full)
        self.assertIn("not been generated", kr.full_report_html(app_html, truth, None))


class FunctionCallingTest(unittest.TestCase):
    FINAL = {"trajectory": "declining", "confidence": "medium", "headline": "Declining after S3.",
             "reasoning": "Good-rep rate fell from 83.3 to 25.0 [S].", "metric_progress": [],
             "fatigue_assessment": "Rising.", "risk_comment": "HIGH RISK kept.", "additional_concerns": [],
             "suggested_review_actions": ["Check load."], "citations": []}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdir = Path(self.tmp.name) / PATIENT
        for i, (stamp, good) in enumerate((("20261001_100000", 8), ("20261004_100000", 10),
                                            ("20261008_100000", 3))):
            path = self.pdir / f"se{stamp}"
            path.mkdir(parents=True)
            (path / "report.json").write_text(json.dumps(_report(good, 12, 92, 20 + 5 * i)), encoding="utf-8")
        self.sid = "se20261008_100000"

    def tearDown(self):
        self.tmp.cleanup()

    def test_claude_tool_loop(self):
        sent = []

        class Messages:
            calls = 0

            def create(self, **kw):
                sent.append(json.dumps(kw["messages"], default=str))
                Messages.calls += 1
                if Messages.calls == 1:
                    return SimpleNamespace(content=[
                        SimpleNamespace(type="tool_use", id="a", name="get_progress_overview", input={}),
                        SimpleNamespace(type="tool_use", id="b", name="get_risk_flags", input={})])
                if Messages.calls == 2:
                    return SimpleNamespace(content=[
                        SimpleNamespace(type="tool_use", id="c", name="get_session_detail", input={"session": "S3"}),
                        SimpleNamespace(type="tool_use", id="d", name="get_fatigue_history", input={})])
                return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(FunctionCallingTest.FINAL))])

        fake = SimpleNamespace(Anthropic=lambda **_: SimpleNamespace(messages=Messages()))
        with mock.patch.object(pa, "_model2_provider", return_value=("claude", "k", "claude-x")), \
                mock.patch.dict(sys.modules, {"anthropic": fake}):
            report = pa.generate_progress_report(self.pdir, self.sid)
        self.assertEqual(report["generation_mode"], "AI reasoning (claude: claude-x, 4 tool calls)")
        self.assertEqual([c["tool"] for c in report["tool_calls"]],
                         ["get_progress_overview", "get_risk_flags", "get_session_detail", "get_fatigue_history"])
        self.assertEqual(report["ai"]["trajectory"], "declining")
        self.assertEqual(report["ground_truth"]["risk"]["risk_tag"], "HIGH RISK")
        everything = " ".join(sent)
        for secret in (PATIENT, "se2026", "2026-10", self.tmp.name.replace("\\", "\\\\")):
            self.assertNotIn(secret, everything)

    def test_gemini_tool_loop(self):
        from google.genai import types

        sent = []

        class Models:
            calls = 0

            def generate_content(self, model, contents, config):
                sent.append(json.dumps([c.model_dump(mode="json") for c in contents]))
                Models.calls += 1
                if Models.calls == 1:
                    calls = [SimpleNamespace(name="get_progress_overview", args={}),
                             SimpleNamespace(name="list_session_history", args={})]
                    content = types.Content(role="model", parts=[types.Part.from_text(text="calling")])
                    return SimpleNamespace(function_calls=calls, candidates=[SimpleNamespace(content=content)])
                return SimpleNamespace(function_calls=None, text=json.dumps(FunctionCallingTest.FINAL))

        with mock.patch.object(pa, "_model2_provider", return_value=("gemini", "k", "gemini-x")), \
                mock.patch("google.genai.Client", lambda **_: SimpleNamespace(models=Models())):
            report = pa.generate_progress_report(self.pdir, self.sid)
        self.assertEqual(report["generation_mode"], "AI reasoning (gemini: gemini-x, 2 tool calls)")
        self.assertIn("Good-repetition rate", " ".join(sent))   # tool results went back to the model
        self.assertNotIn(PATIENT, " ".join(sent))

    def test_provider_failure_keeps_ground_truth(self):
        with mock.patch.object(pa, "_model2_provider", return_value=("gemini", "k", "m")), \
                mock.patch.object(pa, "_run_gemini", side_effect=RuntimeError("429 quota")):
            report = pa.generate_progress_report(self.pdir, self.sid)
        self.assertEqual(report["generation_mode"], "Local rule-based report")
        self.assertIn("quota", report["provider_error"])
        self.assertEqual(report["ai"]["trajectory"], report["ground_truth"]["overview"]["rule_trajectory"])


if __name__ == "__main__":
    unittest.main()

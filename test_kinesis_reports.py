"""Tests for sessions handed over by the KINESIS patient app (report/ folder) and their RAG layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

# agent/test_query_planner.py installs a stub retriever (always []) in sys.modules for its
# own pure-Python check; it is discovered first. These tests need the real retriever.
if not hasattr(sys.modules.get("agent.knowledge_retriever"), "KNOWLEDGE_ROOT"):
    sys.modules.pop("agent.knowledge_retriever", None)
    sys.modules.pop("agent.kinesis_session_rag", None)

import kinesis_reports as kr  # noqa: E402
from agent import kinesis_session_rag as rag  # noqa: E402


def _report(good: int, attempts: int, faults: dict[str, int] | None = None, zone: str = "green") -> dict:
    reps = [
        {
            "index": i + 1, "set": 1, "counted": i < good,
            "faults": [] if i < good else list((faults or {"EARLY_DOWN": 1})),
            "t_start": 10.0 + 6 * i, "t_end": 14.0 + 6 * i,
            "peak_deg": 91.0, "lift_s": 1.5, "hold_s": 1.0, "lower_s": 2.0, "muscle_pct_rvc": 60.0,
        }
        for i in range(attempts)
    ]
    return {
        "session": {"exercise": "Bicep curls", "target_deg": 90.0, "personal_rom_deg": 104.0,
                    "load_kg": 3.0, "muscle_reference_calibrated": True},
        "totals": {"attempts": attempts, "good_reps": good, "not_counted": attempts - good,
                   "good_rate": good / attempts if attempts else None,
                   "avg_peak_angle_deg": 91.0 if attempts else None},
        "faults": [{"code": c, "tip": "tip", "reps": n} for c, n in (faults or {}).items()],
        "criteria": [{"code": "EARLY_DOWN", "tip": "Pause at the top", "rule": "hold >= 0.75 s",
                      "enabled": True}],
        "fatigue": {"final_score": 12.0, "zone": zone, "source": "counted", "trend": [{"rep": 1, "score": 0.0}]},
        "reps": reps,
    }


class ReportStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _publish(self, patient: str, session: str, report: dict) -> Path:
        path = self.root / patient / session
        path.mkdir(parents=True)
        (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (path / "manifest.json").write_text(json.dumps({"exercise": "Bicep curls"}), encoding="utf-8")
        return path

    def test_lists_sessions_and_tracks_review_status(self):
        self._publish("yk", "se20261025_100000", _report(0, 0))
        self._publish("yk", "se20261026_100000", _report(8, 10, {"EARLY_DOWN": 2}))
        (self.root / "yk" / "notes").mkdir()                     # not a session folder
        self.assertEqual(kr.list_patients(self.root), ["yk"])
        sessions = kr.list_sessions("yk", self.root)
        self.assertEqual([s["session_id"] for s in sessions], ["se20261025_100000", "se20261026_100000"])
        self.assertEqual(sessions[1]["review_status"], "awaiting_review")

        kr.save_review("yk", "se20261026_100000", "follow_up", "Check the elbow.", "Dr A", self.root)
        self.assertEqual(kr.list_sessions("yk", self.root)[1]["review_status"], "reviewed")
        kr.save_patient_log("yk", "se20261026_100000", {"title": "T", "summary": "S"}, True, self.root)
        self.assertEqual(kr.list_sessions("yk", self.root)[1]["review_status"], "released")
        self.assertTrue((self.root / "yk" / "se20261026_100000" / "patient_log.md").exists())

    def test_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            kr.session_dir("..", "se1", self.root)
        with self.assertRaises(ValueError):
            kr.session_dir("yk", "../../etc", self.root)

    def test_raw_semg_is_cropped_to_the_exercise_window(self):
        path = self._publish("yk", "se1", _report(1, 1))
        lines = ["t,adc"] + [f"{t / 1000:.3f},{2048 + (t % 7)}" for t in range(0, 20000)]
        (path / "emg.csv").write_text("\n".join(lines), encoding="utf-8")
        raw = kr.raw_semg_envelope(path, t0=5.0, t1=15.0, bins=50)
        self.assertEqual(len(raw), 50)
        self.assertGreaterEqual(raw["t_s"].min(), 0)
        self.assertLessEqual(raw["t_s"].max(), 10.0)
        self.assertTrue((raw["max"] >= raw["min"]).all())


class SessionRagTest(unittest.TestCase):
    def test_facts_ignore_earlier_sessions_without_reps(self):
        facts = rag.session_facts(_report(8, 10), [{"good_rate_percent": math.nan}])
        self.assertEqual(facts["history"], {"earlier_sessions": 0})
        facts = rag.session_facts(_report(8, 10), [{"good_rate_percent": 50.0, "fatigue_score": 9.0}])
        self.assertEqual(facts["history"]["previous_good_rate_percent"], 50.0)

    def test_retrieval_reaches_the_section_of_each_frequent_fault(self):
        facts = rag.session_facts(_report(5, 10, {"FOREARM_LIFT": 3, "EARLY_DOWN": 2}))
        sections = [k["section_title"] for k in rag.retrieve("Why were reps not counted?", facts)]
        self.assertIn("Compensation checks", sections)
        self.assertIn("Tempo and control checks", sections)
        fatigue = [k["section_title"] for k in rag.retrieve("Explain the fatigue score", facts)]
        self.assertEqual(fatigue[0], "The fatigue score")

    def test_local_answer_when_no_provider(self):
        with mock.patch.object(rag, "_model2_provider", return_value=("local", "", "Local")):
            result = rag.answer_specialist_question(_report(8, 10, {"EARLY_DOWN": 2}), "Summarise", [])
        self.assertEqual(result["generation_mode"], "Local evidence summary")
        self.assertTrue(any("8 of 10" in p for p in result["key_points"]))
        self.assertTrue(result["knowledge"])

    def test_provider_failure_falls_back_and_never_sends_identifiers(self):
        sent = {}

        def fake_call(provider, key, model, system, payload, tokens):
            sent.update(payload)
            raise RuntimeError("quota")

        report = _report(8, 10)
        with mock.patch.object(rag, "_model2_provider", return_value=("claude", "k", "m")), \
                mock.patch.object(rag, "_generate_model_text", side_effect=fake_call):
            result = rag.answer_specialist_question(report, "Summarise", [])
        self.assertEqual(result["generation_mode"], "Local evidence summary")
        self.assertIn("quota", result["provider_error"])
        text = json.dumps(sent)
        self.assertNotIn("yk", text)
        self.assertNotIn("se2026", text)

    def test_patient_log_keeps_review_and_compares_only_real_history(self):
        review = {"status_label": "Physiotherapist follow-up needed", "level": "red", "notes": "Call me."}
        with mock.patch.object(rag, "_model2_provider", return_value=("local", "", "Local")):
            log = rag.draft_patient_log(_report(4, 10, {"DROP_FAST": 6}), review, [{"good_rate_percent": math.nan}])
            better = rag.draft_patient_log(_report(9, 10), review, [{"good_rate_percent": 50.0}])
        self.assertEqual(log["status_label"], review["status_label"])
        self.assertEqual(log["specialist_note"], "Call me.")
        self.assertIn("first recorded session", log["progress"])
        self.assertIn(rag.PATIENT_TIPS["DROP_FAST"], log["tips"])
        self.assertIn("More of your repetitions", better["progress"])


if __name__ == "__main__":
    unittest.main()

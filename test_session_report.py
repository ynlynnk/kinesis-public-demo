import unittest

from session_report import REPORT_SECTIONS, build_standard_session_report


class SessionReportTests(unittest.TestCase):
    def test_report_always_contains_the_same_sections(self):
        summary = {
            "patient_id": "P001",
            "session_id": "S001",
            "question": "Summarise this session.",
            "generation_mode": "Local session summary",
            "movement": {"result": "Matched", "explanation": "Stable."},
            "fatigue": {
                "result": "Not detected",
                "onset": "Not detected",
                "explanation": "No sustained pattern.",
            },
            "session_metrics": {
                "window_count": 10,
                "movement_mismatch_window_count": 2,
                "movement_consistency": 0.8,
                "fatigue_pattern_window_count": 1,
                "fatigue_pattern_window_fraction": 0.1,
            },
            "session_comparison": "Similar to the previous session.",
            "physiotherapist_review": {"status": "Awaiting review"},
            "evidence_items": [],
            "important_note": "Not a diagnosis.",
        }

        report = build_standard_session_report(summary)
        sections = tuple(
            key
            for key in report
            if key not in {"report_version", "generated_at"}
        )
        self.assertEqual(sections, REPORT_SECTIONS)
        self.assertEqual(
            report["movement_finding"]["movement_consistency_percent"],
            80.0,
        )
        self.assertFalse(
            report["movement_finding"]["repetitions"]["available"]
        )


if __name__ == "__main__":
    unittest.main()

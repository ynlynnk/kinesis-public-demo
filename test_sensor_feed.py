import unittest
from pathlib import Path

from sensor_feed import (
    load_sensor_frame,
    prepare_sensor_frame,
    summarize_sensor_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DATA = PROJECT_ROOT / "data" / "nexus_testing.csv"


class SensorFeedTests(unittest.TestCase):
    def test_nexus_snapshot_is_loaded_and_summarized(self):
        frame, source, is_live = load_sensor_frame("", TEST_DATA)
        prepared = prepare_sensor_frame(frame)
        summary = summarize_sensor_frame(prepared)

        self.assertEqual(source, "Bundled Nexus connectivity-test snapshot")
        self.assertFalse(is_live)
        self.assertEqual(summary["row_count"], 12)
        self.assertEqual(summary["session_count"], 2)
        self.assertEqual(summary["sequence_gaps"], 0)
        self.assertTrue(summary["quaternion_valid"])
        self.assertFalse(summary["nexus_input_ready"])
        self.assertFalse(summary["cadence_ready"])
        self.assertEqual(summary["nexus_window_rows"], 256)
        self.assertEqual(summary["complete_nexus_windows"], 0)
        self.assertFalse(summary["checkpoint_available"])
        self.assertAlmostEqual(
            summary["median_interval_seconds"],
            4.59,
            places=2,
        )

        readiness = {
            row["Requirement"]: row
            for row in summary["readiness"]
        }
        self.assertEqual(readiness["sEMG input"]["Status"], "Ready")
        self.assertEqual(
            readiness["Recorded-row cadence"]["Status"],
            "Not ready",
        )
        self.assertEqual(readiness["Orientation input"]["Status"], "Ready")
        self.assertEqual(
            readiness["CNN checkpoint file"]["Status"],
            "Not ready",
        )

    def test_orientation_fields_are_created(self):
        frame, _, _ = load_sensor_frame("", TEST_DATA)
        prepared = prepare_sensor_frame(frame)

        for column in ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]:
            self.assertIn(column, prepared.columns)
            self.assertTrue(prepared[column].notna().all())

    def test_missing_required_column_is_rejected(self):
        frame, _, _ = load_sensor_frame("", TEST_DATA)
        with self.assertRaisesRegex(ValueError, "Quat Z"):
            prepare_sensor_frame(frame.drop(columns=["Quat Z"]))


if __name__ == "__main__":
    unittest.main()

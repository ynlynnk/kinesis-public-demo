import unittest

import numpy as np
import pandas as pd

from firebase_sensor_adapter import firebase_payload_to_frame
from realtime_input_contract import assess_realtime_sensor_frame
from sensor_feed import prepare_sensor_frame


def make_batch(duration_seconds: float = 13.0) -> dict:
    semg_count = int(500 * duration_seconds)
    imu_count = int(50 * duration_seconds)
    semg = [
        {
            "seq": index,
            "device_time_us": index * 2000,
            "raw": 1000.0 + 20.0 * np.sin(index / 17.0),
        }
        for index in range(semg_count)
    ]
    imu = [
        {
            "device_time_us": index * 20000,
            "quat_w": 1.0,
            "quat_x": 0.0,
            "quat_y": 0.0,
            "quat_z": 0.0,
            "calib": 3,
        }
        for index in range(imu_count)
    ]
    return {
        "recording_id": "R001",
        "participant_id": "P001",
        "server_time_iso": "2026-10-23T10:00:00+08:00",
        "semg_rate_hz": 500,
        "imu_rate_hz": 50,
        "semg_samples": semg,
        "imu_samples": imu,
    }


class RealtimeInputContractTests(unittest.TestCase):
    def test_dense_firebase_batch_passes_input_contract(self):
        frame = firebase_payload_to_frame(make_batch())
        prepared = prepare_sensor_frame(frame)
        assessment = assess_realtime_sensor_frame(prepared)

        self.assertEqual(len(frame), 6500)
        self.assertTrue(assessment["ready"])
        self.assertAlmostEqual(
            assessment["observed_semg_rate_hz"], 500.0, places=2
        )
        self.assertEqual(assessment["signal_quality"]["status"], "Good")
        self.assertEqual(
            assessment["signal_quality"]["metrics"]["missing_sequence_count"],
            0,
        )

    def test_sparse_firebase_summary_is_rejected(self):
        batch = make_batch()
        batch["semg_samples"] = batch["semg_samples"][:1]
        with self.assertRaisesRegex(ValueError, "sparse summary"):
            firebase_payload_to_frame(batch)

    def test_server_timestamp_requires_timezone(self):
        batch = make_batch()
        batch["server_time_iso"] = "2026-10-23T10:00:00"
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            firebase_payload_to_frame(batch)

    def test_mixed_recordings_are_rejected(self):
        first = make_batch()
        second = make_batch()
        second["recording_id"] = "R002"
        with self.assertRaisesRegex(ValueError, "only one recording"):
            firebase_payload_to_frame([first, second])

    def test_sequence_gap_blocks_inference(self):
        frame = firebase_payload_to_frame(make_batch())
        frame.loc[5000:, "Seq"] += 1
        assessment = assess_realtime_sensor_frame(prepare_sensor_frame(frame))

        self.assertFalse(assessment["ready"])
        checks = {item["check"]: item for item in assessment["checks"]}
        self.assertFalse(checks["Sequence continuity"]["passed"])
        self.assertEqual(assessment["signal_quality"]["status"], "Invalid")
        self.assertIn(
            "Sequence continuity",
            assessment["signal_quality"]["blocking_issues"],
        )

    def test_low_imu_calibration_is_reported_for_review(self):
        batch = make_batch()
        for sample in batch["imu_samples"][:10]:
            sample["calib"] = 1
        frame = firebase_payload_to_frame(batch)
        assessment = assess_realtime_sensor_frame(prepare_sensor_frame(frame))

        self.assertTrue(assessment["ready"])
        self.assertEqual(
            assessment["signal_quality"]["status"], "Needs review"
        )
        self.assertTrue(assessment["signal_quality"]["warnings"])


if __name__ == "__main__":
    unittest.main()


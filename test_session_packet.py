import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from firebase_sensor_adapter import firebase_payload_to_frame
from realtime_input_contract import assess_realtime_sensor_frame
from sensor_feed import prepare_sensor_frame
from session_packet import (
    build_repetition_summary,
    build_session_packet,
    save_session_packet,
)
from test_realtime_input_contract import make_batch


class SessionPacketTests(unittest.TestCase):
    def setUp(self):
        self.sensor_data = prepare_sensor_frame(
            firebase_payload_to_frame(make_batch())
        )
        self.assessment = assess_realtime_sensor_frame(self.sensor_data)

    def test_unavailable_repetitions_are_not_invented(self):
        summary = build_repetition_summary()
        self.assertFalse(summary["available"])
        self.assertIsNone(summary["total_repetitions"])

    def test_repetition_counts_are_validated(self):
        with self.assertRaisesRegex(ValueError, "0 <= correct <= total"):
            build_repetition_summary(8, 9)

    def test_packet_keeps_repetitions_separate_from_windows(self):
        model1 = {
            "prediction_released": False,
            "release_reasons": ["Device validation pending"],
            "window_predictions": [
                {"movement_incorrect_probability": 0.2},
                {"movement_incorrect_probability": 0.7},
            ],
        }
        packet = build_session_packet(
            self.sensor_data,
            self.assessment,
            model1,
            patient_id="P001",
            session_id="S001",
            muscle_group="Forearm",
            exercise="Pronation and supination",
            total_repetitions=10,
            correct_repetitions=8,
        )

        self.assertEqual(packet["repetition_summary"]["correct_repetitions"], 8)
        self.assertEqual(packet["movement_window_summary"]["matching_windows"], 1)
        self.assertIn("not counted exercise repetitions", packet["movement_window_summary"]["note"])
        self.assertEqual(packet["signal_quality"]["status"], "Good")

    def test_packet_can_be_saved_for_local_history(self):
        packet = build_session_packet(
            self.sensor_data,
            self.assessment,
            None,
            patient_id="P001",
            session_id="S002",
            muscle_group="Forearm",
            exercise="Pronation and supination",
        )
        with TemporaryDirectory() as directory:
            output = save_session_packet(packet, directory)
            saved = json.loads(Path(output).read_text(encoding="utf-8"))
        self.assertEqual(saved["packet_type"], "kinesis.completed_session")
        self.assertEqual(saved["storage"]["future_transport"], "firebase")


if __name__ == "__main__":
    unittest.main()

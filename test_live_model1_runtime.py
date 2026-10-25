import unittest
from pathlib import Path

from firebase_sensor_adapter import firebase_payload_to_frame
from live_model1_runtime import (
    RealtimeInputNotReady,
    predict_live_model1_session,
)
from sensor_feed import prepare_sensor_frame
from test_realtime_input_contract import make_batch


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models"


class LiveModel1RuntimeTests(unittest.TestCase):
    def test_dense_forearm_session_runs_both_real_checkpoints(self):
        sensor_data = prepare_sensor_frame(
            firebase_payload_to_frame(make_batch())
        )
        result = predict_live_model1_session(
            sensor_data,
            "Forearm",
            MODEL_ROOT / "kinesis_forearm_movement_cnn.pt",
            MODEL_ROOT / "kinesis_forearm_fatigue_cnn.pt",
        )

        self.assertIn(result["movement"]["display_label"], {
            "Neutral", "Pronation", "Supination", "Uncertain"
        })
        self.assertIn(result["fatigue"]["display_label"], {
            "Low-fatigue phase", "Fatigue-like phase", "Uncertain"
        })
        self.assertFalse(result["device_validated"])
        self.assertFalse(result["prediction_released"])

    def test_bad_cadence_is_rejected_before_inference(self):
        batch = make_batch()
        for index, sample in enumerate(batch["semg_samples"]):
            sample["device_time_us"] = index * 5000
        batch["semg_rate_hz"] = 500
        with self.assertRaisesRegex(ValueError, "measured sEMG cadence"):
            firebase_payload_to_frame(batch)


if __name__ == "__main__":
    unittest.main()


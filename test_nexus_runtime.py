import unittest
from pathlib import Path

import pandas as pd

from nexus_runtime import predict_latest_nexus_window
from sensor_feed import prepare_sensor_frame


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DATA = PROJECT_ROOT / "data" / "nexus_testing.csv"
MODEL_FILE = (
    PROJECT_ROOT / "artifacts" / "models" / "kinesis_nexus_cnn.pt"
)


class NexusRuntimeTests(unittest.TestCase):
    def test_trained_preview_predicts_latest_window(self):
        sensor_data = prepare_sensor_frame(pd.read_csv(TEST_DATA))
        result = predict_latest_nexus_window(sensor_data, str(MODEL_FILE))

        self.assertEqual(result["training_data_kind"], "synthetic")
        self.assertEqual(result["window_rows"], 8)
        self.assertEqual(result["window_end_sequence"], 7)
        self.assertGreaterEqual(result["fatigue_probability"], 0.0)
        self.assertLessEqual(result["fatigue_probability"], 1.0)
        self.assertGreaterEqual(
            result["movement_incorrect_probability"],
            0.0,
        )
        self.assertLessEqual(
            result["movement_incorrect_probability"],
            1.0,
        )
        self.assertFalse(result["validated_on_real_data"])
        self.assertFalse(result["input_within_training_range"])
        self.assertGreater(result["input_max_absolute_z_score"], 3.0)
        self.assertTrue(result["output_saturated"])
        self.assertFalse(result["prediction_released"])


if __name__ == "__main__":
    unittest.main()

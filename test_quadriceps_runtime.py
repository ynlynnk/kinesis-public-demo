import unittest
from pathlib import Path

import numpy as np

from quadriceps_runtime import (
    FATIGUE_CLASS_NAMES,
    MOVEMENT_CLASS_NAMES,
    load_quadriceps_fatigue_model,
    load_quadriceps_movement_model,
    predict_quadriceps_fatigue_window,
    predict_quadriceps_movement_window,
)
from synthetic_live_demo import create_demo_buffer


PROJECT_ROOT = Path(__file__).resolve().parent
MOVEMENT_MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "kinesis_quadriceps_movement_cnn.pt"
)
FATIGUE_MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "kinesis_quadriceps_fatigue_cnn.pt"
)


class QuadricepsRuntimeTests(unittest.TestCase):
    def test_checkpoints_use_real_labelled_quadriceps_data(self):
        _, movement = load_quadriceps_movement_model(
            str(MOVEMENT_MODEL_FILE)
        )
        _, fatigue = load_quadriceps_fatigue_model(str(FATIGUE_MODEL_FILE))

        for checkpoint in (movement, fatigue):
            self.assertEqual(
                checkpoint["training_data_kind"],
                "real_public_semg",
            )
            self.assertTrue(checkpoint["trained_on_labelled_data"])
            self.assertTrue(checkpoint["validated_on_public_real_data"])
            self.assertFalse(checkpoint["validated_on_kinesis_data"])
            self.assertEqual(checkpoint["muscle_group"], "Quadriceps")

        movement_training = set(movement["training_subjects"])
        movement_validation = set(movement["validation_subjects"])
        movement_testing = set(movement["test_metrics"]["test_subjects"])
        self.assertFalse(movement_training & movement_validation)
        self.assertFalse(movement_training & movement_testing)
        self.assertFalse(movement_validation & movement_testing)

        fatigue_training = set(fatigue["training_subjects"])
        fatigue_testing = set(fatigue["test_metrics"]["test_subjects"])
        self.assertFalse(fatigue_training & fatigue_testing)

    def test_dashboard_buffer_runs_through_both_quadriceps_cnns(self):
        state = create_demo_buffer("Sustained contraction")
        signal = np.asarray(state["signal"], dtype=np.float32)
        quaternions = np.asarray(
            state["imu_quaternion"],
            dtype=np.float32,
        )

        movement = predict_quadriceps_movement_window(
            signal[-256:],
            quaternions[-64:],
            str(MOVEMENT_MODEL_FILE),
        )
        fatigue = predict_quadriceps_fatigue_window(
            signal[-512:],
            str(FATIGUE_MODEL_FILE),
        )

        self.assertIn(movement["label"], MOVEMENT_CLASS_NAMES)
        self.assertIn(fatigue["label"], FATIGUE_CLASS_NAMES)
        self.assertAlmostEqual(
            sum(movement["probabilities"].values()),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            sum(fatigue["probabilities"].values()),
            1.0,
            places=5,
        )
        self.assertGreaterEqual(fatigue["fatigue_score"], 0.0)
        self.assertLessEqual(fatigue["fatigue_score"], 1.0)

    def test_movement_runtime_rejects_an_incomplete_imu_window(self):
        with self.assertRaisesRegex(ValueError, "64 quaternion samples"):
            predict_quadriceps_movement_window(
                np.zeros(256, dtype=np.float32),
                np.tile(
                    np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    (63, 1),
                ),
                str(MOVEMENT_MODEL_FILE),
            )


if __name__ == "__main__":
    unittest.main()

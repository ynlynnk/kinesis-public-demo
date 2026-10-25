import unittest
from pathlib import Path

from forearm_fatigue_runtime import (
    CLASS_NAMES,
    load_fatigue_verification_samples,
    load_forearm_fatigue_model,
    predict_normalized_fatigue_window,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = (
    PROJECT_ROOT / "artifacts" / "models" / "kinesis_forearm_fatigue_cnn.pt"
)
SAMPLES_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "kinesis_fatigue_verification_samples.npz"
)


class ForearmFatigueRuntimeTests(unittest.TestCase):
    def test_checkpoint_uses_real_data_and_subject_split(self):
        _, checkpoint = load_forearm_fatigue_model(str(MODEL_FILE))
        self.assertEqual(checkpoint["training_data_kind"], "real_public_semg")
        self.assertEqual(checkpoint["label_kind"], "protocol_phase_proxy")
        self.assertTrue(checkpoint["validated_on_public_real_data"])
        self.assertFalse(checkpoint["validated_on_kinesis_data"])
        self.assertGreater(checkpoint["test_metrics"]["accuracy"], 0.5)

        training = set(checkpoint["training_subjects"])
        validation = set(checkpoint["validation_subjects"])
        testing = set(checkpoint["test_metrics"]["test_subjects"])
        self.assertFalse(training & validation)
        self.assertFalse(training & testing)
        self.assertFalse(validation & testing)

    def test_bundled_held_out_examples_run_through_cnn(self):
        examples = load_fatigue_verification_samples(str(SAMPLES_FILE))
        self.assertEqual(len(examples["windows"]), len(CLASS_NAMES))
        for window, expected_index in zip(
            examples["windows"], examples["labels"]
        ):
            result = predict_normalized_fatigue_window(window, str(MODEL_FILE))
            self.assertEqual(result["label"], CLASS_NAMES[int(expected_index)])
            self.assertAlmostEqual(
                sum(result["probabilities"].values()), 1.0, places=5
            )
            self.assertGreaterEqual(result["fatigue_score"], 0.0)
            self.assertLessEqual(result["fatigue_score"], 1.0)


if __name__ == "__main__":
    unittest.main()

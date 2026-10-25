import unittest
from pathlib import Path

from forearm_movement_runtime import (
    CLASS_NAMES,
    load_forearm_movement_model,
    load_forearm_verification_samples,
    predict_normalized_forearm_window,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = (
    PROJECT_ROOT / "artifacts" / "models" / "kinesis_forearm_movement_cnn.pt"
)
SAMPLES_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "kinesis_forearm_verification_samples.npz"
)


class ForearmMovementRuntimeTests(unittest.TestCase):
    def test_checkpoint_uses_real_labelled_data_and_subject_split(self):
        _, checkpoint = load_forearm_movement_model(str(MODEL_FILE))

        self.assertEqual(checkpoint["training_data_kind"], "real_public_semg")
        self.assertTrue(checkpoint["trained_on_labelled_data"])
        self.assertTrue(checkpoint["validated_on_public_real_data"])
        self.assertFalse(checkpoint["validated_on_kinesis_data"])
        self.assertGreater(checkpoint["test_metrics"]["accuracy"], 1 / 3)

        training = set(checkpoint["training_subjects"])
        validation = set(checkpoint["validation_subjects"])
        testing = set(checkpoint["test_metrics"]["test_subjects"])
        self.assertFalse(training & validation)
        self.assertFalse(training & testing)
        self.assertFalse(validation & testing)

    def test_bundled_held_out_examples_run_through_the_cnn(self):
        examples = load_forearm_verification_samples(str(SAMPLES_FILE))
        self.assertEqual(len(examples["windows"]), len(CLASS_NAMES))

        for window, expected_index in zip(
            examples["windows"],
            examples["labels"],
        ):
            result = predict_normalized_forearm_window(
                window,
                str(MODEL_FILE),
            )
            self.assertEqual(result["label"], CLASS_NAMES[int(expected_index)])
            self.assertAlmostEqual(
                sum(result["probabilities"].values()),
                1.0,
                places=5,
            )


if __name__ == "__main__":
    unittest.main()

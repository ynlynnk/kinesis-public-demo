import unittest

import numpy as np

from synthetic_live_demo import (
    BUFFER_SAMPLES,
    DEVICE_BUFFER_SAMPLES,
    IMU_BUFFER_SAMPLES,
    SAMPLE_RATE_HZ,
    STEP_SAMPLES,
    advance_demo_buffer,
    create_demo_buffer,
    generate_synthetic_imu,
    generate_synthetic_semg,
)
from model1_signal_pipeline import DEVICE_SEMG_RATE_HZ


class SyntheticLiveDemoTests(unittest.TestCase):
    def test_generator_is_repeatable_and_finite(self):
        first = generate_synthetic_semg(0, 100, "Rhythmic contraction")
        second = generate_synthetic_semg(0, 100, "Rhythmic contraction")

        np.testing.assert_allclose(first, second)
        self.assertTrue(np.isfinite(first).all())

    def test_complete_buffer_is_ready_for_both_cnns(self):
        state = create_demo_buffer("Low activity")

        self.assertEqual(len(state["signal"]), BUFFER_SAMPLES)
        self.assertEqual(len(state["raw_semg"]), DEVICE_BUFFER_SAMPLES)
        self.assertEqual(len(state["filtered_semg"]), DEVICE_BUFFER_SAMPLES)
        self.assertEqual(len(state["imu_quaternion"]), IMU_BUFFER_SAMPLES)
        self.assertEqual(state["next_sample"], BUFFER_SAMPLES)
        self.assertEqual(state["elapsed_seconds"], 0.0)

    def test_generated_imu_contains_unit_quaternions(self):
        quaternions = generate_synthetic_imu(0, 100)
        norms = np.linalg.norm(quaternions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_advance_keeps_rolling_window_size(self):
        state = create_demo_buffer("Sustained contraction")
        advanced = advance_demo_buffer(state)

        self.assertEqual(len(advanced["signal"]), BUFFER_SAMPLES)
        self.assertEqual(len(advanced["raw_semg"]), DEVICE_BUFFER_SAMPLES)
        self.assertEqual(len(advanced["imu_quaternion"]), IMU_BUFFER_SAMPLES)
        self.assertEqual(
            advanced["next_sample"],
            BUFFER_SAMPLES
            + int(round(STEP_SAMPLES * SAMPLE_RATE_HZ / DEVICE_SEMG_RATE_HZ)),
        )
        self.assertAlmostEqual(advanced["elapsed_seconds"], 1.0)
        self.assertFalse(
            np.array_equal(state["signal"], advanced["signal"])
        )


if __name__ == "__main__":
    unittest.main()

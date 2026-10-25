import unittest

import numpy as np

from model1_signal_pipeline import (
    DEVICE_SEMG_RATE_HZ,
    EMA_ALPHA,
    MODEL_SEMG_RATE_HZ,
    ema_filter,
    normalize_to_baseline,
    prepare_semg_for_model,
    resample_signal,
)


class Model1SignalPipelineTests(unittest.TestCase):
    def test_ema_uses_documented_alpha(self):
        filtered, final_value = ema_filter([0.0, 1.0, 1.0], EMA_ALPHA)
        np.testing.assert_allclose(filtered, [0.0, 0.25, 0.4375])
        self.assertAlmostEqual(final_value, 0.4375)

    def test_500_hz_second_becomes_200_model_samples(self):
        source = np.linspace(-1.0, 1.0, DEVICE_SEMG_RATE_HZ)
        resampled = resample_signal(
            source,
            DEVICE_SEMG_RATE_HZ,
            MODEL_SEMG_RATE_HZ,
        )
        self.assertEqual(len(resampled), MODEL_SEMG_RATE_HZ)

    def test_prepared_signal_is_finite_and_clipped(self):
        source = np.sin(
            2 * np.pi * 35 * np.arange(DEVICE_SEMG_RATE_HZ)
            / DEVICE_SEMG_RATE_HZ
        )
        prepared = prepare_semg_for_model(
            source,
            baseline_mean=0.0,
            baseline_standard_deviation=0.1,
        )
        self.assertEqual(len(prepared), MODEL_SEMG_RATE_HZ)
        self.assertTrue(np.isfinite(prepared).all())
        self.assertLessEqual(float(np.abs(prepared).max()), 12.0)

    def test_baseline_normalisation_rejects_invalid_scale(self):
        normalized = normalize_to_baseline([1.0, 2.0], 1.0, 0.0)
        np.testing.assert_allclose(normalized, [0.0, 12.0])


if __name__ == "__main__":
    unittest.main()

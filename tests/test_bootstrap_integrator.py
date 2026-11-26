"""Regression tests for short-history forecast assimilation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from apartment_thermal import predict_from_combined_data


def combined_data_with_history(history_hours: int) -> pd.DataFrame:
    """Build hourly deployed-schema rows with history and a future forecast."""
    initial_time = pd.Timestamp("2026-01-03T16:00:00Z")
    past_times = pd.date_range(
        initial_time - pd.Timedelta(hours=history_hours),
        initial_time,
        freq="h",
    )
    future_times = pd.date_range(initial_time, "2026-01-04T08:00:00Z", freq="h")

    # Indoor temperatures deliberately run warmer than the fixed model so the
    # bootstrap integrator has a repeatable residual to assimilate.
    elapsed = np.arange(len(past_times), dtype=float)
    indoor = pd.DataFrame(
        {
            "recorded_at": past_times,
            "sensor": "Living Room",
            "temperature_f": 72.0 + 0.03 * elapsed,
            "valid": True,
            "shortwave_radiation_w_m2": 0.0,
        }
    )
    outside_times = past_times.union(future_times)
    outside = pd.DataFrame(
        {
            "recorded_at": outside_times,
            "sensor": "Outside",
            "temperature_f": 60.0,
            "valid": True,
            "shortwave_radiation_w_m2": 0.0,
        }
    )
    return pd.concat([indoor, outside], ignore_index=True)


class BootstrapIntegratorTests(unittest.TestCase):
    def test_short_history_reanchors_without_integrating(self) -> None:
        forecast = predict_from_combined_data(combined_data_with_history(8))

        self.assertTrue(
            forecast.error_integrator_status.startswith(
                "waiting_for_bootstrap_history"
            )
        )
        self.assertEqual(forecast.error_integrator_correction_f_per_hour, 0.0)
        self.assertEqual(
            forecast.curve.iloc[0]["predicted_indoor_f"],
            forecast.initial_temperature_f,
        )

    def test_bootstrap_integrator_runs_without_batch_history(self) -> None:
        forecast = predict_from_combined_data(combined_data_with_history(18))

        self.assertNotEqual(forecast.bias_status, "applied")
        self.assertEqual(
            forecast.error_integrator_status,
            "applied_bootstrap",
        )
        self.assertGreater(forecast.integrator_observations_used, 0)
        self.assertGreaterEqual(forecast.integrator_history_hours, 12.0)
        self.assertLessEqual(
            abs(forecast.error_integrator_correction_f_per_hour),
            0.2,
        )
        self.assertEqual(
            forecast.curve.iloc[0]["predicted_indoor_f"],
            forecast.initial_temperature_f,
        )

    def test_normal_integrator_runs_after_batch_becomes_available(self) -> None:
        forecast = predict_from_combined_data(combined_data_with_history(80))

        self.assertEqual(forecast.bias_status, "applied")
        self.assertEqual(
            forecast.error_integrator_status,
            "applied_after_batch",
        )
        self.assertGreater(forecast.integrator_observations_used, 0)


if __name__ == "__main__":
    unittest.main()

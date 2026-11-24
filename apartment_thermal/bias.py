"""Recent-history bias assimilation for the fixed historical model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import ModelCParameters, simulate_model_c


@dataclass(frozen=True)
class BiasCorrectionResult:
    raw_correction_f_per_hour: float
    correction_f_per_hour: float
    corrected_parameters: ModelCParameters
    daily_windows: int
    baseline_rmse_f: float
    corrected_rmse_f: float


@dataclass(frozen=True)
class ErrorIntegratorResult:
    correction_f_per_hour: float
    corrected_parameters: ModelCParameters
    observations_used: int
    baseline_one_step_rmse_f: float
    integrated_one_step_rmse_f: float


def _recent_windows(
    indoor: pd.Series,
    end: pd.Timestamp,
    lookback_days: int,
    minimum_points_per_day: int,
) -> list[pd.Series]:
    start = end - pd.Timedelta(days=lookback_days)
    anchors = pd.date_range(start.ceil("D"), end.floor("D"), freq="D")
    windows = []
    for anchor in anchors:
        observed = indoor.loc[
            (indoor.index >= anchor) &
            (indoor.index < min(anchor + pd.Timedelta(days=1), end))
        ]
        if len(observed) >= minimum_points_per_day:
            windows.append(observed)
    return windows


def estimate_recent_bias(
    indoor: pd.Series,
    weather: pd.DataFrame,
    base_parameters: ModelCParameters,
    *,
    end: pd.Timestamp | None = None,
    lookback_days: int = 7,
    half_life_days: float = 3.0,
    maximum_absolute_correction_f_per_hour: float = 0.75,
    minimum_daily_windows: int = 3,
    minimum_points_per_day: int = 24,
    shrinkage_factor: float = 0.5,
) -> BiasCorrectionResult:
    """Estimate one recent background-gain correction with fixed physics.

    Each recent day is independently initialized from its first measurement.
    Recent days receive more weight, and robust loss limits the influence of
    fan/window events. Tau and solar gain remain fixed at historical values.
    """
    end = pd.Timestamp(end if end is not None else indoor.index.max())
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    windows = _recent_windows(indoor, end, lookback_days, minimum_points_per_day)
    if len(windows) < minimum_daily_windows:
        raise ValueError(
            f"Bias correction needs {minimum_daily_windows} complete days; "
            f"found {len(windows)}"
        )
    newest_day = max(window.index[0] for window in windows)

    def residuals(correction: np.ndarray) -> np.ndarray:
        parameters = replace(
            base_parameters,
            background_gain_f_per_hour=(
                base_parameters.background_gain_f_per_hour + float(correction[0])
            ),
        )
        chunks = []
        for observed in windows:
            age_days = (newest_day - observed.index[0]).total_seconds() / 86400.0
            weight = 0.5 ** (age_days / half_life_days)
            predicted = simulate_model_c(
                observed.index, observed.iloc[0], weather, parameters,
            )
            chunks.append(np.sqrt(weight) * (predicted[1:] - observed.to_numpy()[1:]))
        return np.concatenate(chunks)

    baseline_error = residuals(np.array([0.0]))
    optimized = least_squares(
        residuals,
        x0=[0.0],
        bounds=(
            [-maximum_absolute_correction_f_per_hour],
            [maximum_absolute_correction_f_per_hour],
        ),
        loss="soft_l1",
        f_scale=0.5,
    )
    raw_correction = float(optimized.x[0])
    correction = raw_correction * shrinkage_factor
    corrected = replace(
        base_parameters,
        background_gain_f_per_hour=base_parameters.background_gain_f_per_hour + correction,
    )
    corrected_error = residuals(np.array([correction]))
    return BiasCorrectionResult(
        raw_correction_f_per_hour=raw_correction,
        correction_f_per_hour=correction,
        corrected_parameters=corrected,
        daily_windows=len(windows),
        baseline_rmse_f=float(np.sqrt(np.mean(baseline_error**2))),
        corrected_rmse_f=float(np.sqrt(np.mean(corrected_error**2))),
    )


def integrate_recent_errors(
    indoor: pd.Series,
    weather: pd.DataFrame,
    base_parameters: ModelCParameters,
    *,
    end: pd.Timestamp | None = None,
    lookback_days: int = 7,
    integral_gain_per_hour: float = 0.02,
    leak_factor_per_hour: float = 0.98,
    maximum_absolute_correction_f_per_hour: float = 0.5,
    residual_clip_f: float = 3.0,
    maximum_gap_hours: float = 2.0,
) -> ErrorIntegratorResult:
    """Run a leaky, anti-windup integral controller over recent errors.

    Every interval starts from the actual prior indoor measurement. The
    resulting integral state adjusts background heat rate, not temperature,
    so today's forecast still begins exactly at the latest measured value.
    """
    end = pd.Timestamp(end if end is not None else indoor.index.max())
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    recent = indoor.loc[
        (indoor.index >= end - pd.Timedelta(days=lookback_days)) &
        (indoor.index <= end)
    ].dropna()
    if len(recent) < 3:
        raise ValueError("Insufficient recent measurements for error integration")
    correction = 0.0
    baseline_errors = []
    integrated_errors = []
    used = 0
    for previous_time, current_time in zip(recent.index[:-1], recent.index[1:]):
        dt_hours = (current_time - previous_time).total_seconds() / 3600.0
        if dt_hours <= 0 or dt_hours > maximum_gap_hours:
            continue
        times = pd.DatetimeIndex([previous_time, current_time])
        initial = float(recent.loc[previous_time])
        actual = float(recent.loc[current_time])
        baseline_prediction = simulate_model_c(
            times, initial, weather, base_parameters,
        )[-1]
        active_parameters = replace(
            base_parameters,
            background_gain_f_per_hour=(
                base_parameters.background_gain_f_per_hour + correction
            ),
        )
        integrated_prediction = simulate_model_c(
            times, initial, weather, active_parameters,
        )[-1]
        baseline_errors.append(baseline_prediction - actual)
        integrated_error = integrated_prediction - actual
        integrated_errors.append(integrated_error)
        controller_error = np.clip(-integrated_error, -residual_clip_f, residual_clip_f)
        correction *= leak_factor_per_hour ** dt_hours
        proposed = correction + integral_gain_per_hour * controller_error * dt_hours
        correction = float(np.clip(
            proposed,
            -maximum_absolute_correction_f_per_hour,
            maximum_absolute_correction_f_per_hour,
        ))
        used += 1
    if used == 0:
        raise ValueError("No recent measurement intervals passed gap checks")
    corrected_parameters = replace(
        base_parameters,
        background_gain_f_per_hour=base_parameters.background_gain_f_per_hour + correction,
    )
    return ErrorIntegratorResult(
        correction_f_per_hour=correction,
        corrected_parameters=corrected_parameters,
        observations_used=used,
        baseline_one_step_rmse_f=float(np.sqrt(np.mean(np.square(baseline_errors)))),
        integrated_one_step_rmse_f=float(np.sqrt(np.mean(np.square(integrated_errors)))),
    )

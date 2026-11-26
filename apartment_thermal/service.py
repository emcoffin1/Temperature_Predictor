"""High-level API intended for generator scripts and scheduled jobs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .model import (
    DEFAULT_SENSOR,
    ModelCParameters,
    forecast_curve,
    load_indoor_readings,
    load_default_parameters,
    load_parameters,
    normalize_weather,
)
from .bias import estimate_recent_bias, integrate_recent_errors
from .config import load_config


@dataclass(frozen=True)
class IndoorForecast:
    curve: pd.DataFrame
    minimum_f: float
    maximum_f: float
    minimum_time: pd.Timestamp
    maximum_time: pd.Timestamp
    initial_temperature_f: float
    generated_from: pd.Timestamp
    bias_correction_f_per_hour: float = 0.0
    bias_status: str = "not_requested"
    recent_baseline_rmse_f: float | None = None
    recent_corrected_rmse_f: float | None = None
    forecast_mode: str = "model_c_full"
    input_fallbacks: tuple[str, ...] = ()
    error_integrator_correction_f_per_hour: float = 0.0
    error_integrator_status: str = "not_requested"
    integrator_baseline_one_step_rmse_f: float | None = None
    integrator_active_one_step_rmse_f: float | None = None
    integrator_observations_used: int = 0
    integrator_history_hours: float = 0.0

    def hourly_curve(
        self,
        timezone: str = "America/Los_Angeles",
    ) -> pd.DataFrame:
        """Interpolate the internal curve onto local hourly clock boundaries."""
        local_start = self.generated_from.tz_convert(timezone)
        next_midnight = local_start.floor("D") + pd.DateOffset(days=1)
        first_hour = local_start.ceil("h")
        local_times = pd.date_range(
            first_hour,
            next_midnight,
            freq="h",
            inclusive="left",
        )
        utc_times = local_times.tz_convert("UTC")
        utc_times = utc_times[utc_times <= self.curve.index.max()]
        if utc_times.empty:
            utc_times = pd.DatetimeIndex([self.generated_from])
            local_times = utc_times.tz_convert(timezone)
        else:
            local_times = utc_times.tz_convert(timezone)
        source_ns = self.curve.index.view("int64")
        target_ns = utc_times.view("int64")
        hourly = pd.DataFrame(
            {
                "predicted_indoor_f": np.interp(
                    target_ns,
                    source_ns,
                    self.curve["predicted_indoor_f"],
                ),
                "outdoor_f": np.interp(
                    target_ns,
                    source_ns,
                    self.curve["outdoor_f"],
                ),
            },
            index=local_times,
        )
        hourly.index.name = "local_time"
        return hourly

    def summary(self, timezone: str = "America/Los_Angeles") -> dict[str, object]:
        return {
            "generated_from": self.generated_from.isoformat(),
            "initial_temperature_f": self.initial_temperature_f,
            "minimum_f": self.minimum_f,
            "minimum_time": self.minimum_time.tz_convert(timezone).isoformat(),
            "maximum_f": self.maximum_f,
            "maximum_time": self.maximum_time.tz_convert(timezone).isoformat(),
            "timezone": timezone,
            "bias_correction_f_per_hour": self.bias_correction_f_per_hour,
            "bias_status": self.bias_status,
            "recent_baseline_rmse_f": self.recent_baseline_rmse_f,
            "recent_corrected_rmse_f": self.recent_corrected_rmse_f,
            "forecast_mode": self.forecast_mode,
            "input_fallbacks": list(self.input_fallbacks),
            "error_integrator_correction_f_per_hour": (
                self.error_integrator_correction_f_per_hour
            ),
            "error_integrator_status": self.error_integrator_status,
            "integrator_baseline_one_step_rmse_f": (
                self.integrator_baseline_one_step_rmse_f
            ),
            "integrator_active_one_step_rmse_f": (
                self.integrator_active_one_step_rmse_f
            ),
            "integrator_observations_used": self.integrator_observations_used,
            "integrator_history_hours": self.integrator_history_hours,
        }


def predict_from_combined_data(
    data_or_path: pd.DataFrame | Path,
    parameters: ModelCParameters | Path | None = None,
    *,
    sensor: str = DEFAULT_SENSOR,
    timezone: str = "America/Los_Angeles",
    maximum_sensor_age_hours: float = 3.0,
    config: dict[str, object] | Path | None = None,
) -> IndoorForecast:
    """Forecast from one file/query result using the deployed Parquet schema."""
    data = (
        pd.read_parquet(data_or_path)
        if isinstance(data_or_path, Path)
        else data_or_path.copy()
    )
    required = {"recorded_at", "sensor", "temperature_f"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Combined data is missing columns: {sorted(missing)}")
    data["recorded_at"] = pd.to_datetime(data["recorded_at"], utc=True)
    living_rows = data.loc[data["sensor"].eq(sensor)]
    if living_rows.empty:
        raise ValueError(f"No rows found for {sensor!r}")
    initial_time = living_rows["recorded_at"].max()
    past = data.loc[data["recorded_at"] <= initial_time]
    future_outside = data.loc[
        data["sensor"].eq("Outside") &
        data["recorded_at"].ge(initial_time)
    ]
    recent_outside = past.loc[past["sensor"].eq("Outside")]
    return predict_remaining_local_day(
        indoor_data=past,
        weather_forecast=future_outside,
        parameters=parameters,
        sensor=sensor,
        timezone=timezone,
        maximum_sensor_age_hours=maximum_sensor_age_hours,
        recent_indoor_data=past,
        recent_weather_history=recent_outside,
        config=config,
    )


def predict_remaining_local_day(
    indoor_data: pd.DataFrame | Path,
    weather_forecast: pd.DataFrame | Path,
    parameters: ModelCParameters | Path | None = None,
    *,
    sensor: str = DEFAULT_SENSOR,
    timezone: str = "America/Los_Angeles",
    maximum_sensor_age_hours: float = 3.0,
    recent_indoor_data: pd.DataFrame | Path | None = None,
    recent_weather_history: pd.DataFrame | Path | None = None,
    config: dict[str, object] | Path | None = None,
) -> IndoorForecast:
    """Predict from the latest indoor reading through local midnight.

    This is the primary integration function for a job run around 1 a.m. It
    consumes the full hourly forecast curve, not only its daily min/max.
    """
    readings = load_indoor_readings(indoor_data, sensor)
    initial_time = readings.index[-1]
    initial_temperature = float(readings.iloc[-1])
    local_initial = initial_time.tz_convert(timezone)
    next_midnight = local_initial.floor("D") + pd.DateOffset(days=1)
    remaining_hours = max(
        1,
        math.ceil((next_midnight - local_initial).total_seconds() / 3600.0),
    )
    input_fallbacks = []
    try:
        future = normalize_weather(weather_forecast)
        input_fallbacks.extend(future.attrs.get("fallbacks", []))
        available = future.index[future.index >= initial_time]
        if available.empty:
            raise ValueError("Forecast data ends before the latest sensor reading")
        sensor_gap = available[0] - initial_time
        if sensor_gap > pd.Timedelta(hours=maximum_sensor_age_hours):
            raise ValueError(
                f"Latest {sensor} reading is {sensor_gap} before forecast data begins"
            )
    except ValueError as error:
        # With no usable future outdoor temperature, persistence is the only
        # defensible fallback. Do not invent a weather curve.
        times = pd.date_range(
            initial_time.ceil("h"), next_midnight.tz_convert("UTC"), freq="h",
        ).insert(0, initial_time).drop_duplicates().sort_values()
        curve = pd.DataFrame(
            {
                "predicted_indoor_f": np.full(len(times), initial_temperature),
                "outdoor_f": np.full(len(times), np.nan),
            },
            index=times,
        ).rename_axis("recorded_at")
        return IndoorForecast(
            curve=curve,
            minimum_f=initial_temperature,
            maximum_f=initial_temperature,
            minimum_time=times[0],
            maximum_time=times[0],
            initial_temperature_f=initial_temperature,
            generated_from=initial_time,
            bias_status="not_applied_persistence_fallback",
            forecast_mode="indoor_persistence",
            input_fallbacks=(f"unusable_forecast_data: {error}",),
        )
    if parameters is None:
        model_parameters = load_default_parameters()
    elif isinstance(parameters, Path):
        model_parameters = load_parameters(parameters)
    else:
        model_parameters = parameters
    bias_correction = 0.0
    bias_status = "not_requested"
    baseline_rmse = None
    corrected_rmse = None
    integrator_correction = 0.0
    integrator_status = "not_requested"
    integrator_baseline_rmse = None
    integrator_active_rmse = None
    integrator_observations_used = 0
    integrator_history_hours = 0.0
    if recent_indoor_data is not None or recent_weather_history is not None:
        if recent_indoor_data is None or recent_weather_history is None:
            bias_status = "fallback: incomplete recent-history inputs"
            integrator_status = "fallback: incomplete recent-history inputs"
        else:
            runtime_config = (
                load_config(config)
                if isinstance(config, Path) or config is None
                else config
            )
            bias_config = runtime_config["bias_correction"]
            integrator_config = runtime_config["error_integrator"]
            try:
                recent_readings = load_indoor_readings(recent_indoor_data, sensor)
                interval = str(runtime_config["model"]["resample_interval"])
                recent_readings = (
                    recent_readings.resample(interval).mean()
                    .interpolate(limit=2, limit_area="inside").dropna()
                )
                recent_weather = normalize_weather(
                    recent_weather_history, require_radiation=True,
                )
                recent_readings = recent_readings.loc[
                    recent_readings.index.to_series().between(
                        recent_weather.index.min(), recent_weather.index.max()
                    )
                ]
                if len(recent_readings) >= 2:
                    integrator_history_hours = float(
                        (recent_readings.index[-1] - recent_readings.index[0])
                        .total_seconds() / 3600.0
                    )
            except ValueError as error:
                bias_status = f"fallback: {error}"
                integrator_status = f"fallback: {error}"
            else:
                batch_applied = False
                if not bool(bias_config["enabled"]):
                    bias_status = "disabled_by_config"
                else:
                    try:
                        assimilated = estimate_recent_bias(
                            recent_readings,
                            recent_weather,
                            model_parameters,
                            end=initial_time,
                            lookback_days=int(bias_config["lookback_days"]),
                            half_life_days=float(bias_config["half_life_days"]),
                            maximum_absolute_correction_f_per_hour=float(
                                bias_config["maximum_absolute_correction_f_per_hour"]
                            ),
                            minimum_daily_windows=int(
                                bias_config["minimum_daily_windows"]
                            ),
                            minimum_points_per_day=int(
                                bias_config["minimum_points_per_day"]
                            ),
                            shrinkage_factor=float(bias_config["shrinkage_factor"]),
                        )
                        model_parameters = assimilated.corrected_parameters
                        bias_correction = assimilated.correction_f_per_hour
                        baseline_rmse = assimilated.baseline_rmse_f
                        corrected_rmse = assimilated.corrected_rmse_f
                        bias_status = "applied"
                        batch_applied = True
                    except ValueError as error:
                        bias_status = f"fallback: {error}"

                if not bool(integrator_config["enabled"]):
                    integrator_status = "disabled_by_config"
                else:
                    minimum_history_hours = float(
                        integrator_config.get("bootstrap_minimum_history_hours", 12.0)
                    )
                    if integrator_history_hours < minimum_history_hours:
                        integrator_status = (
                            "waiting_for_bootstrap_history: "
                            f"{integrator_history_hours:.1f}/"
                            f"{minimum_history_hours:.1f} hours"
                        )
                    else:
                        gain = float(integrator_config["integral_gain_per_hour"])
                        correction_limit = float(
                            integrator_config[
                                "maximum_absolute_correction_f_per_hour"
                            ]
                        )
                        mode = "after_batch"
                        if not batch_applied:
                            gain *= float(
                                integrator_config.get(
                                    "bootstrap_gain_multiplier", 0.5,
                                )
                            )
                            correction_limit = min(
                                correction_limit,
                                float(
                                    integrator_config.get(
                                        "bootstrap_maximum_absolute_correction_f_per_hour",
                                        0.2,
                                    )
                                ),
                            )
                            mode = "bootstrap"
                        try:
                            integrated = integrate_recent_errors(
                                recent_readings,
                                recent_weather,
                                model_parameters,
                                end=initial_time,
                                lookback_days=int(
                                    integrator_config.get(
                                        "lookback_days",
                                        bias_config["lookback_days"],
                                    )
                                ),
                                integral_gain_per_hour=gain,
                                leak_factor_per_hour=float(
                                    integrator_config["leak_factor_per_hour"]
                                ),
                                maximum_absolute_correction_f_per_hour=(
                                    correction_limit
                                ),
                                residual_clip_f=float(
                                    integrator_config["residual_clip_f"]
                                ),
                                maximum_gap_hours=float(
                                    integrator_config["maximum_gap_hours"]
                                ),
                            )
                            model_parameters = integrated.corrected_parameters
                            integrator_correction = integrated.correction_f_per_hour
                            integrator_baseline_rmse = (
                                integrated.baseline_one_step_rmse_f
                            )
                            integrator_active_rmse = (
                                integrated.integrated_one_step_rmse_f
                            )
                            integrator_observations_used = integrated.observations_used
                            integrator_status = f"applied_{mode}"
                        except ValueError as error:
                            integrator_status = f"fallback: {error}"
    curve = forecast_curve(
        initial_time,
        initial_temperature,
        future,
        model_parameters,
        remaining_hours,
    )
    curve = curve.loc[curve.index <= next_midnight.tz_convert("UTC")]
    if curve.empty:
        raise ValueError("Weather forecast has no points before local midnight")
    minimum_time = curve["predicted_indoor_f"].idxmin()
    maximum_time = curve["predicted_indoor_f"].idxmax()
    return IndoorForecast(
        curve=curve,
        minimum_f=float(curve.loc[minimum_time, "predicted_indoor_f"]),
        maximum_f=float(curve.loc[maximum_time, "predicted_indoor_f"]),
        minimum_time=minimum_time,
        maximum_time=maximum_time,
        initial_temperature_f=initial_temperature,
        generated_from=initial_time,
        bias_correction_f_per_hour=bias_correction,
        bias_status=bias_status,
        recent_baseline_rmse_f=baseline_rmse,
        recent_corrected_rmse_f=corrected_rmse,
        forecast_mode=(
            "model_c_no_solar"
            if "missing_radiation_assumed_zero" in input_fallbacks
            else "model_c_uv_proxy"
            if "uv_index_converted_to_radiation_proxy" in input_fallbacks
            else "model_c_full"
        ),
        input_fallbacks=tuple(input_fallbacks),
        error_integrator_correction_f_per_hour=integrator_correction,
        error_integrator_status=integrator_status,
        integrator_baseline_one_step_rmse_f=integrator_baseline_rmse,
        integrator_active_one_step_rmse_f=integrator_active_rmse,
        integrator_observations_used=integrator_observations_used,
        integrator_history_hours=integrator_history_hours,
    )

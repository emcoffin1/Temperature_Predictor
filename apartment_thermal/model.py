"""Reusable Living Room thermal model selected by rolling backtests.

The model uses outdoor temperature, global horizontal shortwave radiation, and
a constant background term:

    dTin/dt = (Tout - Tin) / tau + k_shortwave * (GHI / 100 W/m^2) + q0

All timestamps are UTC internally. Temperatures are degrees Fahrenheit and
time is measured in hours.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


DEFAULT_SENSOR = "Living Room"
DEFAULT_INTERVAL = "30min"
UV_PROXY_W_M2_PER_INDEX = 119.91596898101388


@dataclass(frozen=True)
class ModelCParameters:
    tau_hours: float
    shortwave_gain_f_per_hour_per_100w: float
    background_gain_f_per_hour: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.tau_hours,
            self.shortwave_gain_f_per_hour_per_100w,
            self.background_gain_f_per_hour,
        ])

    @classmethod
    def from_array(cls, values: np.ndarray) -> "ModelCParameters":
        return cls(*(float(value) for value in values))


@dataclass(frozen=True)
class FitResult:
    parameters: ModelCParameters
    daily_windows: int
    cost: float
    optimizer_evaluations: int


def _normalize_time(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "recorded_at" not in frame and "time" in frame:
        frame = frame.rename(columns={"time": "recorded_at"})
    if "recorded_at" not in frame:
        raise ValueError("Data has no recorded_at/time column")
    frame["recorded_at"] = pd.to_datetime(frame["recorded_at"], utc=True)
    return frame


class MissingOutdoorTemperatureError(ValueError):
    """Raised when no usable outdoor-temperature field is available."""


def normalize_weather(
    data_or_path: pd.DataFrame | Path,
    *,
    require_radiation: bool = False,
) -> pd.DataFrame:
    """Normalize provider-agnostic outdoor temperature and solar radiation.

    Missing forecast radiation degrades to zero solar gain and is reported in
    ``weather.attrs['fallbacks']``. Training and bias assimilation may require
    radiation explicitly.
    """
    data = pd.read_parquet(data_or_path) if isinstance(data_or_path, Path) else data_or_path
    data = _normalize_time(data)
    temperature_column = next(
        (
            name for name in (
                "outdoor_f", "temperature_f",
            ) if name in data
        ),
        None,
    )
    radiation_column = next(
        (
            name for name in (
                "shortwave_w_m2", "shortwave_radiation_w_m2",
            )
            if name in data
        ),
        None,
    )
    if temperature_column is None:
        raise MissingOutdoorTemperatureError(
            "No outdoor temperature column is available"
        )
    fallbacks = []
    if radiation_column is None and "uv_index" in data:
        data = data.copy()
        radiation_column = "shortwave_w_m2"
        data[radiation_column] = (
            pd.to_numeric(data["uv_index"], errors="coerce")
            * UV_PROXY_W_M2_PER_INDEX
        )
        fallbacks.append("uv_index_converted_to_radiation_proxy")
    if radiation_column is None:
        if require_radiation:
            raise ValueError("No solar/shortwave radiation column is available")
        data = data.copy()
        radiation_column = "shortwave_w_m2"
        data[radiation_column] = 0.0
        fallbacks.append("missing_radiation_assumed_zero")
    weather = data[["recorded_at", temperature_column, radiation_column]].rename(
        columns={temperature_column: "outdoor_f", radiation_column: "shortwave_w_m2"},
    )
    weather[["outdoor_f", "shortwave_w_m2"]] = weather[
        ["outdoor_f", "shortwave_w_m2"]
    ].apply(pd.to_numeric, errors="coerce")
    weather = (
        weather.dropna().sort_values("recorded_at").drop_duplicates("recorded_at")
        .set_index("recorded_at")
    )
    weather["shortwave_w_m2"] = weather["shortwave_w_m2"].clip(lower=0)
    if len(weather) < 2:
        raise ValueError("Weather contains fewer than two usable timestamps")
    weather.attrs["fallbacks"] = fallbacks
    return weather


def load_training_data(
    data_or_path: pd.DataFrame | Path,
    sensor: str = DEFAULT_SENSOR,
    interval: str = DEFAULT_INTERVAL,
) -> tuple[pd.Series, pd.DataFrame]:
    """Load a combined parquet and return resampled indoor and outdoor inputs."""
    data = (
        pd.read_parquet(data_or_path)
        if isinstance(data_or_path, Path)
        else data_or_path.copy()
    )
    data = _normalize_time(data)
    required = {"sensor", "temperature_f"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    if "valid" in data:
        data = data.loc[data["valid"].fillna(False)]
    data = data.sort_values("recorded_at").drop_duplicates(["sensor", "recorded_at"])
    indoor_rows = data.loc[
        data["sensor"].eq(sensor), ["recorded_at", "temperature_f"],
    ]
    if indoor_rows.empty:
        raise ValueError(f"No indoor readings found for {sensor!r}")
    indoor = (
        indoor_rows.set_index("recorded_at")["temperature_f"].astype(float)
        .resample(interval).mean().interpolate(limit=2, limit_area="inside").dropna()
    )
    weather = normalize_weather(
        data.loc[data["sensor"].eq("Outside")], require_radiation=True,
    )
    indoor = indoor.loc[
        indoor.index.to_series().between(weather.index.min(), weather.index.max())
    ]
    if len(indoor) < 24:
        raise ValueError("Insufficient overlapping indoor and weather data")
    return indoor, weather


def load_indoor_readings(
    data_or_path: pd.DataFrame | Path,
    sensor: str = DEFAULT_SENSOR,
) -> pd.Series:
    """Load valid raw indoor readings for production forecast initialization."""
    data = (
        pd.read_parquet(data_or_path)
        if isinstance(data_or_path, Path)
        else data_or_path.copy()
    )
    data = _normalize_time(data)
    if not {"sensor", "temperature_f"}.issubset(data):
        raise ValueError("Sensor data requires sensor and temperature_f columns")
    if "valid" in data:
        data = data.loc[data["valid"].fillna(False)]
    readings = (
        data.loc[data["sensor"].eq(sensor), ["recorded_at", "temperature_f"]]
        .dropna().sort_values("recorded_at").drop_duplicates("recorded_at")
        .set_index("recorded_at")["temperature_f"].astype(float)
    )
    if readings.empty:
        raise ValueError(f"No valid readings found for {sensor!r}")
    return readings


def simulate_model_c(
    times: pd.DatetimeIndex,
    initial_temperature_f: float,
    weather: pd.DataFrame,
    parameters: ModelCParameters,
) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if times.empty or not times.is_monotonic_increasing:
        raise ValueError("Simulation times must be nonempty and increasing")
    t0 = times[0]
    evaluation_hours = (times - t0).total_seconds().to_numpy() / 3600.0
    weather_hours = (weather.index - t0).total_seconds().to_numpy() / 3600.0
    outdoor = weather["outdoor_f"].to_numpy()
    shortwave_100w = weather["shortwave_w_m2"].to_numpy() / 100.0

    def derivative(time: float, state: np.ndarray) -> list[float]:
        outdoor_now = np.interp(time, weather_hours, outdoor)
        solar_now = np.interp(time, weather_hours, shortwave_100w)
        rate = (outdoor_now - state[0]) / parameters.tau_hours
        rate += parameters.shortwave_gain_f_per_hour_per_100w * solar_now
        rate += parameters.background_gain_f_per_hour
        return [rate]

    if evaluation_hours[-1] == 0:
        return np.array([initial_temperature_f])
    solution = solve_ivp(
        derivative,
        (0.0, evaluation_hours[-1]),
        [initial_temperature_f],
        t_eval=evaluation_hours,
        max_step=0.25,
        rtol=2e-6,
        atol=2e-6,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0]


def _daily_windows(
    indoor: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Series]:
    anchors = pd.date_range(start.ceil("h"), end - pd.Timedelta(days=1), freq="24h")
    windows = []
    for anchor in anchors:
        window = indoor.loc[
            (indoor.index >= anchor) &
            (indoor.index <= anchor + pd.Timedelta(days=1))
        ]
        if len(window) >= 24:
            windows.append(window)
    return windows


def fit_model_c(
    indoor: pd.Series,
    weather: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> FitResult:
    """Fit independent daily windows to limit long-horizon accumulated drift."""
    windows = _daily_windows(indoor, start, end)
    if not windows:
        raise ValueError("No complete daily fitting windows in the requested history")

    def residuals(values: np.ndarray) -> np.ndarray:
        parameters = ModelCParameters.from_array(values)
        errors = []
        for observed in windows:
            predicted = simulate_model_c(
                observed.index, observed.iloc[0], weather, parameters,
            )
            errors.append(predicted[1:] - observed.to_numpy()[1:])
        return np.concatenate(errors)

    optimized = least_squares(
        residuals,
        x0=[24.0, 0.1, 0.0],
        bounds=([1.0, 0.0, -2.0], [240.0, 2.0, 2.0]),
        x_scale=[24.0, 0.1, 0.2],
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=300,
    )
    if not optimized.success:
        raise RuntimeError(optimized.message)
    return FitResult(
        parameters=ModelCParameters.from_array(optimized.x),
        daily_windows=len(windows),
        cost=float(optimized.cost),
        optimizer_evaluations=int(optimized.nfev),
    )


def fit_recent_model(
    indoor: pd.Series,
    weather: pd.DataFrame,
    end: pd.Timestamp,
    history_days: float = 7.0,
) -> FitResult:
    start = max(indoor.index.min(), end - pd.Timedelta(days=history_days))
    return fit_model_c(indoor, weather, start, end)


def fit_historical_model(
    indoor: pd.Series,
    weather: pd.DataFrame,
) -> FitResult:
    """Fit stable production parameters from every usable historical day."""
    return fit_model_c(indoor, weather, indoor.index.min(), indoor.index.max())


def forecast_curve(
    initial_time: pd.Timestamp,
    initial_temperature_f: float,
    future_weather: pd.DataFrame,
    parameters: ModelCParameters,
    hours: int = 24,
) -> pd.DataFrame:
    initial_time = pd.Timestamp(initial_time)
    initial_time = (
        initial_time.tz_localize("UTC")
        if initial_time.tz is None
        else initial_time.tz_convert("UTC")
    )
    end = min(initial_time + pd.Timedelta(hours=hours), future_weather.index.max())
    future_times = future_weather.loc[
        (future_weather.index >= initial_time) & (future_weather.index <= end)
    ].index
    times = future_times.insert(0, initial_time).drop_duplicates().sort_values()
    if len(times) < 2:
        raise ValueError("Future weather does not cover the requested forecast")
    predicted = simulate_model_c(
        times, initial_temperature_f, future_weather, parameters,
    )
    outdoor = np.interp(
        times.view("int64"), future_weather.index.view("int64"),
        future_weather["outdoor_f"],
    )
    return pd.DataFrame(
        {"predicted_indoor_f": predicted, "outdoor_f": outdoor}, index=times,
    ).rename_axis("recorded_at")


def save_parameters(parameters: ModelCParameters, path: Path, **metadata: object) -> None:
    payload = {"model": "C-shortwave", **asdict(parameters), **metadata}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def load_parameters(path: Path) -> ModelCParameters:
    payload = json.loads(path.read_text())
    return _parameters_from_payload(payload)


def load_default_parameters() -> ModelCParameters:
    """Load the ten-week historical model bundled with the package."""
    payload = json.loads(
        files("apartment_thermal").joinpath("default_model.json").read_text()
    )
    return _parameters_from_payload(payload)


def _parameters_from_payload(payload: dict[str, object]) -> ModelCParameters:
    return ModelCParameters(
        tau_hours=float(payload["tau_hours"]),
        shortwave_gain_f_per_hour_per_100w=float(
            payload["shortwave_gain_f_per_hour_per_100w"]
        ),
        background_gain_f_per_hour=float(payload["background_gain_f_per_hour"]),
    )

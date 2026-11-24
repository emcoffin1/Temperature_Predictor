"""Fit, validate, and use physics-based apartment temperature models.

Compare progressively richer thermal models:
    .venv/bin/python apartment_temp_pred.py --data temps/fake_data_wk46.parquet

Also forecast from a future hourly weather file:
    .venv/bin/python apartment_temp_pred.py --forecast-weather temps/weather-history-2026-W47.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ModelSpec:
    label: str
    parameters: tuple[str, ...]
    initial: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    scale: tuple[float, ...]


MODELS = {
    "A": ModelSpec("passive", ("tau",), (24.0,), (1.0,), (240.0,), (24.0,)),
    "B": ModelSpec(
        "passive + background", ("tau", "q0"), (24.0, 0.0),
        (1.0, -2.0), (240.0, 2.0), (24.0, 0.2),
    ),
    "C": ModelSpec(
        "passive + background + solar", ("tau", "k_solar", "q0"),
        (24.0, 0.05, 0.0), (1.0, 0.0, -2.0),
        (240.0, 2.0, 2.0), (24.0, 0.05, 0.2),
    ),
    "E": ModelSpec(
        "passive + background + cloud-adjusted solar",
        ("tau", "k_solar", "k_cloud", "q0"),
        (24.0, 0.05, -0.05, 0.0), (1.0, 0.0, -2.0, -2.0),
        (240.0, 2.0, 0.0, 2.0), (24.0, 0.05, 0.05, 0.2),
    ),
    "D": ModelSpec(
        "passive + background + solar + wind", ("tau", "kw", "k_solar", "q0"),
        (24.0, 0.001, 0.05, 0.0), (1.0, 0.0, 0.0, -2.0),
        (240.0, 0.02, 2.0, 2.0), (24.0, 0.001, 0.05, 0.2),
    ),
}


def _utc(values: pd.Series | pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def normalize_weather(data_or_path: pd.DataFrame | Path) -> pd.DataFrame:
    """Normalize combined-schema or Open-Meteo weather into model inputs."""
    data = pd.read_parquet(data_or_path) if isinstance(data_or_path, Path) else data_or_path.copy()
    if "recorded_at" not in data and "time" in data:
        data = data.rename(columns={"time": "recorded_at"})
    if "recorded_at" not in data:
        raise ValueError("Weather data has no recorded_at/time column")

    rename = {}
    if "temperature_f" not in data and "temperature_2m" in data:
        rename["temperature_2m"] = "temperature_f"
    if "wind_speed_mph" not in data and "wind_speed_10m" in data:
        rename["wind_speed_10m"] = "wind_speed_mph"
    if "cloud_fraction" not in data:
        if "cloud_cover_percent" in data:
            rename["cloud_cover_percent"] = "cloud_fraction"
        elif "cloud_cover" in data:
            rename["cloud_cover"] = "cloud_fraction"
    # UV is preferred when available. Irradiance is supported, but a model must
    # be trained and forecast with the same solar-input units.
    solar_from_irradiance = False
    if "solar_input" not in data:
        if "uv_index" in data:
            rename["uv_index"] = "solar_input"
        elif "shortwave_radiation" in data:
            rename["shortwave_radiation"] = "solar_input"
            solar_from_irradiance = True
    data = data.rename(columns=rename)

    needed = ["temperature_f", "wind_speed_mph", "solar_input", "cloud_fraction"]
    for column in needed:
        if column not in data:
            if column in {"wind_speed_mph", "solar_input", "cloud_fraction"}:
                data[column] = 0.0
            else:
                raise ValueError(f"Weather data has no {column!r} or recognized alias")
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if solar_from_irradiance:
        # Approximate a UV-like 0-10 scale when a forecast provides W/m^2 only.
        # Prefer requesting uv_index in future forecast collection instead.
        data["solar_input"] = data["solar_input"] / 100.0
    # Source schemas store total cloud cover as 0-100 percent.
    data["cloud_fraction"] = data["cloud_fraction"] / 100.0

    data["recorded_at"] = pd.to_datetime(data["recorded_at"], utc=True)
    weather = (
        data[["recorded_at", *needed]]
        .dropna(subset=["recorded_at", "temperature_f"])
        .sort_values("recorded_at").drop_duplicates("recorded_at")
        .set_index("recorded_at")
    )
    weather[["wind_speed_mph", "solar_input", "cloud_fraction"]] = (
        weather[["wind_speed_mph", "solar_input", "cloud_fraction"]]
        .interpolate(limit_direction="both").clip(lower=0)
    )
    weather["cloud_fraction"] = weather["cloud_fraction"].clip(upper=1)
    return weather.dropna()


def load_measurements(path: Path, sensor: str, interval: str) -> tuple[pd.Series, pd.DataFrame]:
    data = pd.read_parquet(path)
    required = {"recorded_at", "sensor", "temperature_f"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    data = data.copy()
    data["recorded_at"] = pd.to_datetime(data["recorded_at"], utc=True)
    if "valid" in data:
        data = data.loc[data["valid"].fillna(False)]
    data = data.sort_values("recorded_at").drop_duplicates(["sensor", "recorded_at"])

    frame = data.loc[data["sensor"].eq(sensor), ["recorded_at", "temperature_f"]]
    if frame.empty:
        available = ", ".join(sorted(data["sensor"].dropna().unique()))
        raise ValueError(f"No readings for {sensor!r}; available sensors: {available}")
    indoor = (
        frame.set_index("recorded_at")["temperature_f"].astype(float)
        .resample(interval).mean().interpolate(limit=2, limit_area="inside").dropna()
    )
    weather = normalize_weather(data.loc[data["sensor"].eq("Outside")])
    indoor = indoor.loc[indoor.index.to_series().between(weather.index.min(), weather.index.max())]
    if len(indoor) < 20 or len(weather) < 2:
        raise ValueError("Insufficient overlapping indoor and weather observations")
    return indoor, weather


def parameter_dict(model: str, values: Iterable[float]) -> dict[str, float]:
    return dict(zip(MODELS[model].parameters, values, strict=True))


def simulate(
    times: pd.DatetimeIndex, initial_temp: float, weather: pd.DataFrame,
    model: str, values: Iterable[float],
) -> np.ndarray:
    """Integrate one forecast window from a measured initial temperature."""
    times = _utc(times)
    if len(times) == 0:
        return np.array([], dtype=float)
    if not times.is_monotonic_increasing:
        raise ValueError("Simulation times must be increasing")
    params = parameter_dict(model, values)
    t0 = times[0]
    t_eval = (times - t0).total_seconds().to_numpy(dtype=float) / 3600.0
    weather_hours = (weather.index - t0).total_seconds().to_numpy(dtype=float) / 3600.0
    tout = weather["temperature_f"].to_numpy(dtype=float)
    wind = weather["wind_speed_mph"].to_numpy(dtype=float)
    solar = weather["solar_input"].to_numpy(dtype=float)
    cloud = weather["cloud_fraction"].to_numpy(dtype=float)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        outdoor = np.interp(t, weather_hours, tout)
        exchange = (outdoor - y[0]) / params["tau"]
        if "kw" in params:
            exchange += params["kw"] * np.interp(t, weather_hours, wind) * (outdoor - y[0])
        gain = params.get("q0", 0.0)
        solar_now = np.interp(t, weather_hours, solar)
        gain += params.get("k_solar", 0.0) * solar_now
        # Test whether cloud cover modifies solar gain beyond the UV input.
        gain += params.get("k_cloud", 0.0) * solar_now * np.interp(t, weather_hours, cloud)
        return [exchange + gain]

    if t_eval[-1] == 0:
        return np.array([initial_temp])
    solution = solve_ivp(
        rhs, (0.0, t_eval[-1]), [initial_temp], t_eval=t_eval,
        max_step=0.25, rtol=2e-6, atol=2e-6,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0]


def forecast_windows(indoor: pd.Series, start: pd.Timestamp, end: pd.Timestamp, hours: int = 24):
    """Yield independent windows, each initialized from a real observation."""
    anchors = pd.date_range(start.ceil("h"), end, freq=f"{hours}h")
    for anchor in anchors:
        window = indoor.loc[(indoor.index >= anchor) & (indoor.index <= anchor + pd.Timedelta(hours=hours))]
        if len(window) >= max(6, hours * 2):
            yield window


def fit_model(
    model: str, indoor: pd.Series, weather: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp,
):
    spec = MODELS[model]
    windows = list(forecast_windows(indoor, start, end - pd.Timedelta(hours=24)))
    if not windows:
        raise ValueError("Not enough training data for a complete fitting window")

    def residuals(values: np.ndarray) -> np.ndarray:
        chunks = []
        for observed in windows:
            predicted = simulate(observed.index, observed.iloc[0], weather, model, values)
            # Exclude the forced initial point. Residuals remain in F so the
            # robust-loss scale below also has a physical interpretation.
            chunks.append(predicted[1:] - observed.to_numpy()[1:])
        return np.concatenate(chunks)

    result = least_squares(
        residuals, x0=spec.initial, bounds=(spec.lower, spec.upper),
        x_scale=spec.scale, loss="soft_l1", f_scale=0.5, max_nfev=300,
    )
    return result, len(windows)


def score_forecasts(
    model: str, values: Iterable[float], indoor: pd.Series,
    weather: pd.DataFrame, start: pd.Timestamp, horizons: Iterable[int],
) -> pd.DataFrame:
    rows = []
    last_time = min(indoor.index.max(), weather.index.max())
    for horizon in horizons:
        origins = pd.date_range(
            start.ceil("6h"), last_time - pd.Timedelta(hours=horizon), freq="6h",
        )
        errors, baseline_errors, used = [], [], 0
        for origin in origins:
            observed = indoor.loc[
                (indoor.index >= origin) &
                (indoor.index <= origin + pd.Timedelta(hours=horizon))
            ]
            if len(observed) < 3:
                continue
            predicted = simulate(observed.index, observed.iloc[0], weather, model, values)
            errors.extend(predicted[1:] - observed.to_numpy()[1:])
            baseline_errors.extend(observed.iloc[0] - observed.to_numpy()[1:])
            used += 1
        if not errors:
            continue
        error, baseline = np.asarray(errors), np.asarray(baseline_errors)
        rows.append({
            "model": model, "horizon_h": horizon,
            "rmse_f": np.sqrt(np.mean(error**2)), "mae_f": np.mean(np.abs(error)),
            "bias_f": np.mean(error), "baseline_rmse_f": np.sqrt(np.mean(baseline**2)),
            "origins": used,
        })
    return pd.DataFrame(rows)


def near_bounds(model: str, values: np.ndarray) -> list[str]:
    spec, warnings = MODELS[model], []
    for name, value, lower, upper in zip(spec.parameters, values, spec.lower, spec.upper, strict=True):
        span = upper - lower
        if value - lower <= 0.01 * span:
            warnings.append(f"{name} near lower bound")
        elif upper - value <= 0.01 * span:
            warnings.append(f"{name} near upper bound")
    return warnings


def choose_model(scores: pd.DataFrame) -> str:
    """Prefer the simplest model within 0.15 F of the best mean RMSE."""
    means = scores.groupby("model")["rmse_f"].mean().sort_values()
    threshold = means.iloc[0] + 0.15
    return next(model for model in MODELS if model in means.index and means[model] <= threshold)


def print_parameters(model: str, values: np.ndarray) -> None:
    units = {
        "tau": "hr", "kw": "1/(hr*mph)",
        "k_solar": "F/(hr*solar-unit)",
        "k_cloud": "F/(hr*solar-unit*cloud-fraction)", "q0": "F/hr",
    }
    text = ", ".join(
        f"{name}={value:.5g} {units[name]}"
        for name, value in zip(MODELS[model].parameters, values, strict=True)
    )
    print(f"Model {model}: {text}")
    warnings = near_bounds(model, values)
    if warnings:
        print("  identifiability warning: " + "; ".join(warnings))


def plot_validation(
    indoor: pd.Series, weather: pd.DataFrame, cutoff: pd.Timestamp,
    model: str, values: np.ndarray, output: Path,
) -> None:
    observed = indoor.loc[indoor.index >= cutoff]
    predicted = simulate(observed.index, observed.iloc[0], weather, model, values)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, height_ratios=(3, 1),
    )
    ax1.plot(observed.index, observed, label="Measured indoor", color="tab:blue")
    ax1.plot(observed.index, predicted, label=f"Model {model}", color="tab:red")
    ax1.plot(weather.index, weather["temperature_f"], label="Outdoor", color="tab:green", alpha=0.4)
    ax1.set_ylabel("Temperature (F)")
    ax1.legend()
    ax1.grid(alpha=0.25)
    ax2.plot(observed.index, predicted - observed.to_numpy(), color="tab:purple")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Error (F)")
    ax2.set_xlabel("UTC")
    ax2.grid(alpha=0.25)
    fig.suptitle("Chronological holdout (metrics use independent forecast starts)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    indoor, weather = load_measurements(args.data, args.sensor, args.interval)
    start, end = indoor.index.min(), indoor.index.max()
    cutoff = start + (end - start) * args.train_fraction
    # Reserve an extra day beyond the longest horizon so 48-hour validation
    # has several independent origins rather than zero or one.
    validation_needed = pd.Timedelta(hours=max(args.horizons) + 24)
    cutoff = min(cutoff.floor(args.interval), end - validation_needed)
    print(f"Data: {start} to {end}; chronological split: {cutoff}")

    fits, score_frames = {}, []
    for model in MODELS:
        result, window_count = fit_model(model, indoor, weather, start, cutoff)
        fits[model] = result
        print_parameters(model, result.x)
        scaled_jacobian = result.jac * np.asarray(MODELS[model].scale)
        condition = np.linalg.cond(scaled_jacobian)
        print(
            f"  optimizer success={result.success}, cost={result.cost:.4f}, "
            f"daily windows={window_count}, scaled-Jacobian condition={condition:.2e}"
        )
        score_frames.append(
            score_forecasts(model, result.x, indoor, weather, cutoff, args.horizons)
        )

    scores = pd.concat(score_frames, ignore_index=True)
    print("\nChronological validation (initialized from measured temperature):")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    selected = choose_model(scores)
    print(f"\nSelected Model {selected} ({MODELS[selected].label}); simplest within 0.15 F of best mean RMSE.")
    plot_validation(indoor, weather, cutoff, selected, fits[selected].x, args.plot)
    print(f"Validation plot: {args.plot}")

    recent_start = max(start, end - pd.Timedelta(days=args.history_days))
    final, window_count = fit_model(selected, indoor, weather, recent_start, end)
    print(f"\nFinal Model {selected}, {window_count} daily windows from latest {args.history_days:g} days:")
    print_parameters(selected, final.x)

    if args.forecast_weather is not None:
        future = normalize_weather(args.forecast_weather)
        forecast_start = max(indoor.index.max(), future.index.min())
        forecast_end = min(
            forecast_start + pd.Timedelta(hours=args.forecast_hours), future.index.max(),
        )
        forecast_times = future.loc[
            (future.index >= forecast_start) & (future.index <= forecast_end)
        ].index
        if len(forecast_times) < 2:
            raise ValueError("Future weather does not cover the requested forecast period")
        simulation_times = forecast_times.insert(0, indoor.index.max()).drop_duplicates().sort_values()
        prediction = simulate(simulation_times, indoor.iloc[-1], future, selected, final.x)
        forecast = pd.DataFrame({
            "predicted_indoor_f": prediction,
            "outdoor_f": np.interp(
                simulation_times.view("int64"), future.index.view("int64"),
                future["temperature_f"],
            ),
        }, index=simulation_times).loc[forecast_times.min():]
        forecast.index.name = "recorded_at"
        forecast.to_csv(args.forecast_output)
        low_time = forecast["predicted_indoor_f"].idxmin()
        high_time = forecast["predicted_indoor_f"].idxmax()
        print(
            f"\nIndoor forecast: {forecast.predicted_indoor_f.min():.2f} F at {low_time}, "
            f"{forecast.predicted_indoor_f.max():.2f} F at {high_time}"
        )
        print(
            f"Weather range: {forecast.outdoor_f.min():.2f} to {forecast.outdoor_f.max():.2f} F; "
            f"curve saved to {args.forecast_output}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("temps/fake_data_wk46.parquet"))
    parser.add_argument("--sensor", default="Bedroom")
    parser.add_argument("--interval", default="10min")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 24, 48])
    parser.add_argument("--history-days", type=float, default=7.0)
    parser.add_argument("--forecast-weather", type=Path)
    parser.add_argument("--forecast-hours", type=int, default=24)
    parser.add_argument("--forecast-output", type=Path, default=Path("indoor_temperature_forecast.csv"))
    parser.add_argument("--plot", type=Path, default=Path("apartment_thermal_validation.png"))
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction <= 0.9:
        parser.error("--train-fraction must be between 0.5 and 0.9")
    return args


if __name__ == "__main__":
    run(parse_args())

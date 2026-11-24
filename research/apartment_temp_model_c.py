"""Backtest or forecast with the selected Living Room Model C.

The reusable implementation lives in thermal_model.py. This file is the CLI
for development, scheduled fitting, and producing CSV/PNG forecast artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apartment_thermal.model import (
    DEFAULT_INTERVAL,
    DEFAULT_SENSOR,
    fit_recent_model,
    forecast_curve,
    load_training_data,
    normalize_weather,
    save_parameters,
    simulate_model_c,
)


def print_fit(fit) -> None:
    parameters = fit.parameters
    print(
        f"Model C-shortwave: tau={parameters.tau_hours:.3f} hr, "
        f"shortwave gain={parameters.shortwave_gain_f_per_hour_per_100w:.5f} "
        "F/(hr per 100 W/m^2), "
        f"background={parameters.background_gain_f_per_hour:.5f} F/hr"
    )
    print(
        f"Fitting windows={fit.daily_windows}, cost={fit.cost:.3f}, "
        f"optimizer evaluations={fit.optimizer_evaluations}"
    )


def print_range(frame: pd.DataFrame, timezone: str) -> None:
    low_time = frame["predicted_indoor_f"].idxmin()
    high_time = frame["predicted_indoor_f"].idxmax()
    print(
        f"Predicted minimum: {frame.loc[low_time, 'predicted_indoor_f']:.2f} F at "
        f"{low_time.tz_convert(timezone)}"
    )
    print(
        f"Predicted maximum: {frame.loc[high_time, 'predicted_indoor_f']:.2f} F at "
        f"{high_time.tz_convert(timezone)}"
    )


def save_plot(frame: pd.DataFrame, path: Path, title: str, timezone: str) -> None:
    display = frame.copy()
    display.index = display.index.tz_convert(timezone)
    fig, ax = plt.subplots(figsize=(13, 6))
    if "measured_indoor_f" in display:
        ax.plot(display.index, display["measured_indoor_f"], label="Measured Living Room")
    ax.plot(display.index, display["predicted_indoor_f"], label="Model C-shortwave", color="tab:red")
    ax.plot(display.index, display["outdoor_f"], label="Outdoor", color="tab:green", alpha=0.45)
    ax.set(title=title, xlabel=timezone, ylabel="Temperature (F)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_backtest(args: argparse.Namespace, indoor: pd.Series, weather: pd.DataFrame) -> None:
    if args.target_day is None:
        # Select the last complete local day rather than an arbitrary UTC day.
        target_local = (
            indoor.index.max().tz_convert(args.timezone).floor("D")
            - pd.DateOffset(days=1)
        )
    else:
        target_local = pd.Timestamp(args.target_day)
        target_local = (
            target_local.tz_localize(args.timezone)
            if target_local.tz is None
            else target_local.tz_convert(args.timezone)
        ).floor("D")
    target = target_local.tz_convert("UTC")
    target_end = (target_local + pd.DateOffset(days=1)).tz_convert("UTC")
    observed = indoor.loc[
        (indoor.index >= target) & (indoor.index < target_end)
    ]
    if len(observed) < 24:
        raise ValueError(f"Insufficient readings to backtest {target_local.date()}")
    training_end = observed.index[0] - pd.Timedelta(microseconds=1)
    fit = fit_recent_model(indoor, weather, training_end, args.history_days)
    print_fit(fit)
    predicted = simulate_model_c(
        observed.index, observed.iloc[0], weather, fit.parameters,
    )
    error = predicted - observed.to_numpy()
    print(
        f"Backtest {target_local.date()}: RMSE={np.sqrt(np.mean(error**2)):.3f} F, "
        f"MAE={np.mean(np.abs(error)):.3f} F, bias={np.mean(error):+.3f} F"
    )
    outdoor = np.interp(
        observed.index.view("int64"), weather.index.view("int64"), weather["outdoor_f"],
    )
    frame = pd.DataFrame({
        "predicted_indoor_f": predicted,
        "measured_indoor_f": observed.to_numpy(),
        "outdoor_f": outdoor,
        "error_f": error,
    }, index=observed.index).rename_axis("recorded_at")
    frame.to_csv(args.curve_output)
    print_range(frame, args.timezone)
    save_plot(
        frame, args.plot_output,
        f"Living Room backtest: {target_local.date()}", args.timezone,
    )
    save_parameters(
        fit.parameters, args.parameters_output,
        sensor=args.sensor, fitted_through=training_end,
        history_days=args.history_days, solar_input="shortwave_radiation",
    )


def run_forecast(args: argparse.Namespace, indoor: pd.Series, weather: pd.DataFrame) -> None:
    fit = fit_recent_model(indoor, weather, indoor.index.max(), args.history_days)
    print_fit(fit)
    future = normalize_weather(args.forecast_weather)
    available = future.index[future.index >= indoor.index.max()]
    if available.empty:
        raise ValueError("Forecast weather ends before the latest indoor reading")
    gap = available[0] - indoor.index.max()
    if gap > pd.Timedelta(hours=3):
        raise ValueError(f"Sensor reading is stale by {gap}; refresh sensor data")
    frame = forecast_curve(
        indoor.index.max(), indoor.iloc[-1], future, fit.parameters, args.forecast_hours,
    )
    frame.to_csv(args.curve_output)
    print_range(frame, args.timezone)
    save_plot(frame, args.plot_output, "Living Room forecast", args.timezone)
    save_parameters(
        fit.parameters, args.parameters_output,
        sensor=args.sensor, fitted_through=indoor.index.max(),
        history_days=args.history_days, solar_input="shortwave_radiation",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("temps/combined_latest_10_weeks.parquet"),
    )
    parser.add_argument("--sensor", default=DEFAULT_SENSOR)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--history-days", type=float, default=7.0)
    parser.add_argument(
        "--target-day", type=pd.Timestamp,
        help="Local calendar date to backtest (default: last complete local day)",
    )
    parser.add_argument("--forecast-weather", type=Path)
    parser.add_argument("--forecast-hours", type=int, default=24)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    parser.add_argument("--curve-output", type=Path, default=Path("model_c_curve.csv"))
    parser.add_argument("--plot-output", type=Path, default=Path("model_c_curve.png"))
    parser.add_argument(
        "--parameters-output", type=Path, default=Path("model_c_parameters.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indoor, weather = load_training_data(args.data, args.sensor, args.interval)
    if args.forecast_weather:
        run_forecast(args, indoor, weather)
    else:
        run_backtest(args, indoor, weather)
    print(
        f"Curve: {args.curve_output}\nPlot: {args.plot_output}\n"
        f"Parameters: {args.parameters_output}"
    )


if __name__ == "__main__":
    main()

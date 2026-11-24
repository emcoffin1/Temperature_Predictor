"""Walk Model C-shortwave through every usable day in the ten-week dataset.

Each local calendar day is predicted strictly from the preceding seven days.
The first week is therefore calibration history; every later complete day is
an out-of-sample prediction. Outputs include every 30-minute prediction, daily
metrics and fitted parameters, and a full-history plot.
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
    load_training_data,
    simulate_model_c,
)


def local_day_bounds(day: pd.Timestamp, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    local = pd.Timestamp(day)
    local = local.tz_localize(timezone) if local.tz is None else local.tz_convert(timezone)
    local = local.floor("D")
    return local.tz_convert("UTC"), (local + pd.DateOffset(days=1)).tz_convert("UTC")


def walk_forward(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    indoor, weather = load_training_data(args.data, args.sensor, args.interval)
    first_local = indoor.index.min().tz_convert(args.timezone).floor("D")
    last_local = indoor.index.max().tz_convert(args.timezone).floor("D") - pd.DateOffset(days=1)
    first_target = first_local + pd.DateOffset(days=args.history_days)
    target_days = pd.date_range(first_target, last_local, freq="D")
    prediction_frames, metric_rows = [], []

    for number, target_local in enumerate(target_days, start=1):
        start, end = local_day_bounds(target_local, args.timezone)
        observed = indoor.loc[(indoor.index >= start) & (indoor.index < end)]
        if len(observed) < args.minimum_points:
            print(
                f"Skip {target_local.date()}: only {len(observed)} observations",
                flush=True,
            )
            continue
        training_end = observed.index[0] - pd.Timedelta(microseconds=1)
        try:
            fit = fit_recent_model(
                indoor, weather, training_end, history_days=args.history_days,
            )
        except ValueError as error:
            print(f"Skip {target_local.date()}: {error}", flush=True)
            continue
        predicted = simulate_model_c(
            observed.index, observed.iloc[0], weather, fit.parameters,
        )
        actual = observed.to_numpy()
        error = predicted - actual
        outdoor = np.interp(
            observed.index.view("int64"), weather.index.view("int64"),
            weather["outdoor_f"],
        )
        prediction_frames.append(pd.DataFrame({
            "target_local_day": target_local.date().isoformat(),
            "measured_indoor_f": actual,
            "predicted_indoor_f": predicted,
            "outdoor_f": outdoor,
            "error_f": error,
        }, index=observed.index))
        metric_rows.append({
            "target_local_day": target_local.date().isoformat(),
            "observations": len(observed),
            "rmse_f": np.sqrt(np.mean(error**2)),
            "mae_f": np.mean(np.abs(error)),
            "bias_f": np.mean(error),
            "minimum_error_f": np.min(predicted) - np.min(actual),
            "maximum_error_f": np.max(predicted) - np.max(actual),
            "tau_hours": fit.parameters.tau_hours,
            "shortwave_gain_f_per_hour_per_100w": (
                fit.parameters.shortwave_gain_f_per_hour_per_100w
            ),
            "background_gain_f_per_hour": fit.parameters.background_gain_f_per_hour,
            "training_windows": fit.daily_windows,
        })
        print(
            f"{number:02d}/{len(target_days)} {target_local.date()}: "
            f"RMSE={metric_rows[-1]['rmse_f']:.3f} F",
            flush=True,
        )

    if not prediction_frames:
        raise ValueError("No usable walk-forward target days")
    predictions = pd.concat(prediction_frames).sort_index()
    predictions.index.name = "recorded_at"
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(args.predictions_output)
    metrics.to_csv(args.metrics_output, index=False)
    return predictions, metrics


def print_summary(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    error = predictions["error_f"].to_numpy()
    print("\nFull walk-forward summary")
    print(f"Predicted local days: {len(metrics)}")
    print(f"Predicted 30-minute points: {len(predictions):,}")
    print(f"Point-weighted RMSE: {np.sqrt(np.mean(error**2)):.3f} F")
    print(f"Point-weighted MAE: {np.mean(np.abs(error)):.3f} F")
    print(f"Point-weighted bias: {np.mean(error):+.3f} F")
    print(f"Median daily RMSE: {metrics.rmse_f.median():.3f} F")
    print(f"90th-percentile daily RMSE: {metrics.rmse_f.quantile(0.9):.3f} F")
    worst = metrics.loc[metrics.rmse_f.idxmax()]
    print(f"Worst day: {worst.target_local_day}, RMSE={worst.rmse_f:.3f} F")


def save_plot(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path,
    timezone: str,
) -> None:
    display = predictions.copy()
    display.index = display.index.tz_convert(timezone)
    metric_times = pd.to_datetime(metrics["target_local_day"]).dt.tz_localize(timezone)
    fig, (temperature_ax, error_ax, daily_ax) = plt.subplots(
        3, 1, figsize=(18, 11), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    # Plot each target day separately so sensor outages remain visible gaps
    # rather than being joined by misleading diagonal lines.
    for number, (_, day) in enumerate(display.groupby("target_local_day")):
        temperature_ax.plot(
            day.index, day["measured_indoor_f"],
            label="Measured Living Room" if number == 0 else None,
            linewidth=1, color="tab:blue",
        )
        temperature_ax.plot(
            day.index, day["predicted_indoor_f"],
            label="Predicted" if number == 0 else None,
            linewidth=1, alpha=0.85, color="tab:orange",
        )
        error_ax.plot(
            day.index, day["error_f"], color="tab:purple", linewidth=0.8,
        )
    temperature_ax.set_ylabel("Temperature (F)")
    temperature_ax.set_title("Living Room Model C-shortwave: full walk-forward history")
    temperature_ax.legend()
    error_ax.axhline(0, color="black", linewidth=0.8)
    error_ax.set_ylabel("Error (F)")
    daily_ax.plot(metric_times, metrics["rmse_f"], color="tab:red", marker=".")
    daily_ax.set(ylabel="Daily RMSE (F)", xlabel=timezone)
    for axis in (temperature_ax, error_ax, daily_ax):
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("temps/combined_latest_10_weeks.parquet"),
    )
    parser.add_argument("--sensor", default=DEFAULT_SENSOR)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    parser.add_argument("--minimum-points", type=int, default=36)
    parser.add_argument(
        "--predictions-output", type=Path,
        default=Path("full_history_predictions.csv"),
    )
    parser.add_argument(
        "--metrics-output", type=Path,
        default=Path("full_history_daily_metrics.csv"),
    )
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("full_history_backtest.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, metrics = walk_forward(args)
    print_summary(predictions, metrics)
    save_plot(predictions, metrics, args.plot_output, args.timezone)
    print(
        f"\nPredictions: {args.predictions_output}\n"
        f"Daily metrics: {args.metrics_output}\nPlot: {args.plot_output}"
    )


if __name__ == "__main__":
    main()

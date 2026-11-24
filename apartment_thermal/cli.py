"""Command-line entry points for training and scheduled forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import (
    DEFAULT_INTERVAL,
    DEFAULT_SENSOR,
    fit_historical_model,
    load_training_data,
    save_parameters,
)
from .service import predict_from_combined_data, predict_remaining_local_day


def train(args: argparse.Namespace) -> None:
    indoor, weather = load_training_data(args.data, args.sensor, args.interval)
    fit = fit_historical_model(indoor, weather)
    save_parameters(
        fit.parameters,
        args.output,
        sensor=args.sensor,
        trained_from=indoor.index.min(),
        trained_through=indoor.index.max(),
        daily_windows=fit.daily_windows,
        fit_cost=fit.cost,
        solar_input="shortwave_radiation",
    )
    print(
        f"Trained C-shortwave on {fit.daily_windows} daily windows; "
        f"parameters saved to {args.output}"
    )
    print(
        f"tau={fit.parameters.tau_hours:.3f} hr, "
        f"shortwave_gain={fit.parameters.shortwave_gain_f_per_hour_per_100w:.5f}, "
        f"background={fit.parameters.background_gain_f_per_hour:.5f} F/hr"
    )


def forecast(args: argparse.Namespace) -> None:
    if args.combined_data is not None:
        result = predict_from_combined_data(
            args.combined_data,
            args.model,
            sensor=args.sensor,
            timezone=args.timezone,
            maximum_sensor_age_hours=args.maximum_sensor_age_hours,
            config=args.config,
        )
    else:
        if args.sensor_data is None or args.weather is None:
            raise SystemExit(
                "forecast requires --combined-data, or both --sensor-data and --weather"
            )
        result = predict_remaining_local_day(
            args.sensor_data,
            args.weather,
            args.model,
            sensor=args.sensor,
            timezone=args.timezone,
            maximum_sensor_age_hours=args.maximum_sensor_age_hours,
            recent_indoor_data=args.recent_sensor_data,
            recent_weather_history=args.recent_weather,
            config=args.config,
        )
    hourly = result.hourly_curve(args.timezone)
    hourly.to_csv(args.output)
    if args.detailed_output is not None:
        result.curve.to_csv(args.detailed_output)
    summary = result.summary(args.timezone)
    if args.summary_output:
        payload = {
            **summary,
            "hourly_forecast": [
                {
                    "local_time": timestamp.isoformat(),
                    "predicted_indoor_f": float(row.predicted_indoor_f),
                    "outdoor_f": (
                        None if pd.isna(row.outdoor_f) else float(row.outdoor_f)
                    ),
                }
                for timestamp, row in hourly.iterrows()
            ],
        }
        args.summary_output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Hourly forecast saved to {args.output}")
    if args.detailed_output is not None:
        print(f"Detailed integration curve saved to {args.detailed_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apartment-thermal")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="Fit production parameters from combined historical data",
    )
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, default=Path("model_c_parameters.json"))
    train_parser.add_argument("--sensor", default=DEFAULT_SENSOR)
    train_parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    train_parser.set_defaults(handler=train)

    forecast_parser = commands.add_parser(
        "forecast", help="Predict from the latest sensor reading through local midnight",
    )
    forecast_parser.add_argument(
        "--combined-data", type=Path,
        help="One SQL export/Parquet using the deployed fake_data schema",
    )
    forecast_parser.add_argument("--sensor-data", type=Path)
    forecast_parser.add_argument("--weather", type=Path)
    forecast_parser.add_argument(
        "--model", type=Path,
        help="Optional parameter JSON; defaults to the model bundled in the package",
    )
    forecast_parser.add_argument("--output", type=Path, default=Path("indoor_forecast.csv"))
    forecast_parser.add_argument(
        "--detailed-output", type=Path,
        help="Optional internal-resolution diagnostic curve",
    )
    forecast_parser.add_argument("--summary-output", type=Path)
    forecast_parser.add_argument("--sensor", default=DEFAULT_SENSOR)
    forecast_parser.add_argument(
        "--recent-sensor-data", type=Path,
        help="Optional seven-day sensor export used for bias correction",
    )
    forecast_parser.add_argument(
        "--recent-weather", type=Path,
        help="Optional matching seven-day actual weather used for bias correction",
    )
    forecast_parser.add_argument(
        "--config", type=Path,
        help="Optional configuration JSON; defaults to the bundled configuration",
    )
    forecast_parser.add_argument("--timezone", default="America/Los_Angeles")
    forecast_parser.add_argument("--maximum-sensor-age-hours", type=float, default=3.0)
    forecast_parser.set_defaults(handler=forecast)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

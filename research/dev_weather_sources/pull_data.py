"""Download hourly Open-Meteo inputs for apartment temperature prediction."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


# Long Beach Airport (KLGB), matching data_view.py and the training data.
LATITUDE = 33.817930
LONGITUDE = -118.151891

HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "uv_index",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "precipitation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("forecast", "history"), default="forecast",
        help="Use the live forecast API or historical-forecast API",
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--latitude", type=float, default=LATITUDE)
    parser.add_argument("--longitude", type=float, default=LONGITUDE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.days <= 14:
        parser.error("--days must be between 1 and 14")
    return args


def main() -> None:
    args = parse_args()
    end_date = args.start_date + timedelta(days=args.days - 1)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        if args.mode == "forecast"
        else "https://historical-forecast-api.open-meteo.com/v1/forecast"
    )
    params = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "start_date": args.start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_FIELDS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "cell_selection": "land",
    }
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError(f"Invalid Open-Meteo response: {payload}")

    weather = pd.DataFrame(payload["hourly"])
    weather["time"] = pd.to_datetime(weather["time"], utc=True)
    weather = weather.rename(columns={"time": "recorded_at"})
    output = args.output or Path(
        f"temps/weather-{args.mode}-{args.start_date.isoformat()}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(output, index=False)

    print(f"Saved {len(weather)} hourly rows to {output}")
    print(
        f"API grid: {payload.get('latitude')}, {payload.get('longitude')}; "
        f"UTC range: {weather.recorded_at.min()} to {weather.recorded_at.max()}"
    )


if __name__ == "__main__":
    main()

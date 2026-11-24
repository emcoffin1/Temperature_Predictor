"""Build a multi-week sensor/weather training dataset without altering originals.

By default this selects the latest ten *complete* temperature week files,
downloads matching Open-Meteo historical-forecast weather, interpolates it to
the Outside sensor timestamps, and writes one combined parquet file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests


LATITUDE = 33.817930
LONGITUDE = -118.151891

WEATHER_MAPPING = {
    "temperature_2m": "temperature_f",
    "cloud_cover": "cloud_cover_percent",
    "wind_speed_10m": "wind_speed_mph",
    "uv_index": "uv_index",
    "uv_index_clear_sky": "uv_index_clear_sky",
    "relative_humidity_2m": "relative_humidity_percent",
    "dew_point_2m": "dew_point_f",
    "shortwave_radiation": "shortwave_radiation_w_m2",
    "direct_radiation": "direct_radiation_w_m2",
    "diffuse_radiation": "diffuse_radiation_w_m2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("temps"))
    parser.add_argument("--weeks", type=int, default=10)
    parser.add_argument("--latitude", type=float, default=LATITUDE)
    parser.add_argument("--longitude", type=float, default=LONGITUDE)
    parser.add_argument(
        "--output", type=Path, default=Path("temps/combined_latest_10_weeks.parquet"),
    )
    return parser.parse_args()


def complete_week_files(data_dir: Path, count: int) -> list[Path]:
    candidates = []
    pattern = re.compile(r"temperatures-(\d{4})-W(\d{2})\.parquet$")
    for path in data_dir.glob("temperatures-*-W*.parquet"):
        match = pattern.match(path.name)
        if not match:
            continue
        timestamps = pd.to_datetime(
            pd.read_parquet(path, columns=["recorded_at"])["recorded_at"], utc=True,
        )
        # Allow minor collection gaps at either end, but reject partial weeks.
        if timestamps.max() - timestamps.min() >= pd.Timedelta(days=6, hours=12):
            candidates.append((int(match.group(1)), int(match.group(2)), path))
    selected = [item[2] for item in sorted(candidates)[-count:]]
    if len(selected) < count:
        raise ValueError(f"Requested {count} complete weeks; found {len(selected)}")
    return selected


def download_weather(start: pd.Timestamp, end: pd.Timestamp, latitude: float, longitude: float) -> pd.DataFrame:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.date().isoformat(),
        # Include the following midnight/hour so readings late on the final
        # sensor day can be interpolated instead of extrapolated.
        "end_date": (end + pd.Timedelta(days=1)).date().isoformat(),
        "hourly": ",".join(WEATHER_MAPPING),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "cell_selection": "land",
    }
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError(f"Invalid Open-Meteo response: {payload}")
    weather = pd.DataFrame(payload["hourly"]).rename(
        columns={"time": "recorded_at", **WEATHER_MAPPING},
    )
    weather["recorded_at"] = pd.to_datetime(weather["recorded_at"], utc=True)
    for column in WEATHER_MAPPING.values():
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    print(
        f"Open-Meteo grid: {payload.get('latitude')}, {payload.get('longitude')}; "
        f"elevation {payload.get('elevation')} m"
    )
    return weather.sort_values("recorded_at").drop_duplicates("recorded_at")


def main() -> None:
    args = parse_args()
    files = complete_week_files(args.data_dir, args.weeks)
    print("Selected sensor files:")
    for path in files:
        print(f"  {path}")

    sensor_data = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    sensor_data["recorded_at"] = pd.to_datetime(sensor_data["recorded_at"], utc=True)
    sensor_data = (
        sensor_data.sort_values("recorded_at")
        .drop_duplicates(["sensor", "recorded_at"], keep="last")
        .reset_index(drop=True)
    )
    start, end = sensor_data.recorded_at.min(), sensor_data.recorded_at.max()
    weather = download_weather(start, end, args.latitude, args.longitude)

    weather_columns = list(WEATHER_MAPPING.values())
    weather_indexed = weather.set_index("recorded_at")[weather_columns]
    outside_mask = sensor_data["sensor"].eq("Outside")
    outside_times = pd.DatetimeIndex(sensor_data.loc[outside_mask, "recorded_at"])
    if outside_times.empty:
        raise ValueError("Selected files have no Outside sensor readings")

    timeline = weather_indexed.index.union(outside_times).sort_values()
    interpolated = (
        weather_indexed.reindex(timeline)
        .interpolate(method="time", limit_area="inside")
        .reindex(outside_times)
    )
    if interpolated.isna().any().any():
        missing = interpolated.isna().sum()
        raise ValueError(f"Weather does not cover all Outside timestamps:\n{missing}")

    combined = sensor_data.copy()
    for column in weather_columns:
        if column not in combined:
            combined[column] = np.nan
    # Use the same modeled outdoor-temperature source in training and forecasts.
    combined.loc[outside_mask, "temperature_f"] = interpolated["temperature_f"].to_numpy()
    for column in weather_columns:
        if column == "temperature_f":
            continue
        combined.loc[outside_mask, column] = interpolated[column].to_numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(
        f"Saved {len(combined):,} rows covering {start} through {end} to {args.output}"
    )


if __name__ == "__main__":
    main()


from pathlib import Path
import numpy as np
import pandas as pd
import requests

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "temps"

INPUT_FILE = DATA_DIR / "temperatures-2026-W46.parquet"
OUTPUT_FILE = DATA_DIR / "fake_data_wk46.parquet"

# Long Beach Airport (KLGB)
LATITUDE = 33.817930
LONGITUDE = -118.151891

# Week 46, UTC
START = pd.Timestamp("2026-11-09 00:00:26+00:00")
END = pd.Timestamp("2026-11-16 00:00:00+00:00")

# Open-Meteo field -> your database field
WEATHER_MAPPING = {
    "temperature_2m": "temperature_f",
    "cloud_cover": "cloud_cover_percent",
    "wind_speed_10m": "wind_speed_mph",
    "uv_index": "uv_index",
    "relative_humidity_2m": "relative_humidity_percent",
    "dew_point_2m": "dew_point_f",
}

WEATHER_COLUMNS = [
    "cloud_cover_percent",
    "wind_speed_mph",
    "uv_index",
    "relative_humidity_percent",
    "dew_point_f",
]

OUTPUT_COLUMNS = [
    "recorded_at",
    "sensor",
    "temperature_f",
    "valid",
    *WEATHER_COLUMNS,
]


# ==========================================
# 1. LOAD ORIGINAL DATA
# ==========================================

actual = pd.read_parquet(INPUT_FILE)

actual["recorded_at"] = pd.to_datetime(
    actual["recorded_at"], utc=True
)

actual = actual.loc[
    (actual["recorded_at"] >= START)
    & (actual["recorded_at"] < END)
].copy()

actual = actual.sort_values(
    "recorded_at"
).reset_index(drop=True)

if actual.empty:
    raise ValueError("No measurements found for week 46.")

outside_mask = actual["sensor"].eq("Outside")

if not outside_mask.any():
    raise ValueError("No Outside sensor readings found.")

print(f"Loaded {len(actual)} sensor measurements")
print(f"Outside readings: {outside_mask.sum()}")


# ==========================================
# 2. PULL HISTORICAL WEATHER
# ==========================================

URL = (
    "https://historical-forecast-api.open-meteo.com"
    "/v1/forecast"
)

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2026-11-09",
    "end_date": "2026-11-16",
    "hourly": ",".join(WEATHER_MAPPING.keys()),
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "timezone": "UTC",
    "cell_selection": "land",
}

response = requests.get(
    URL, params=params, timeout=45
)

response.raise_for_status()
weather_json = response.json()

if "hourly" not in weather_json:
    raise ValueError(f"Invalid API response: {weather_json}")

print("\nWeather model location:")
print(
    weather_json.get("latitude"),
    weather_json.get("longitude")
)

weather = pd.DataFrame(weather_json["hourly"])

weather = weather.rename(
    columns={
        "time": "recorded_at",
        **WEATHER_MAPPING,
    }
)

weather["recorded_at"] = pd.to_datetime(
    weather["recorded_at"], utc=True
)

weather = (
    weather.sort_values("recorded_at")
    .drop_duplicates("recorded_at")
    .set_index("recorded_at")
)

# Convert weather values to floats
for column in WEATHER_MAPPING.values():
    if column not in weather.columns:
        raise ValueError(f"Missing API field: {column}")

    weather[column] = pd.to_numeric(
        weather[column], errors="coerce"
    )


# ==========================================
# 3. INTERPOLATE WEATHER DATA
# ==========================================

outside_times = pd.DatetimeIndex(
    actual.loc[outside_mask, "recorded_at"]
)

# Make sure historical data covers the readings
if (
    outside_times.min() < weather.index.min()
    or outside_times.max() > weather.index.max()
):
    raise ValueError(
        "Weather data does not cover all sensor timestamps."
    )

timeline = weather.index.union(
    outside_times
).sort_values()

interpolated = (
    weather[list(WEATHER_MAPPING.values())]
    .reindex(timeline)
    .interpolate(
        method="time",
        limit_area="inside"
    )
    .reindex(outside_times)
)

if interpolated.isna().any().any():
    raise ValueError(
        "Missing interpolated weather values:\n"
        f"{interpolated.isna().sum()}"
    )

# Keep physical quantities within valid ranges
interpolated["cloud_cover_percent"] = (
    interpolated["cloud_cover_percent"].clip(0, 100)
)

interpolated["relative_humidity_percent"] = (
    interpolated["relative_humidity_percent"].clip(0, 100)
)

interpolated["uv_index"] = (
    interpolated["uv_index"].clip(lower=0)
)

interpolated["wind_speed_mph"] = (
    interpolated["wind_speed_mph"].clip(lower=0)
)


# ==========================================
# 4. FORMAT TO MATCH ACTUAL SCHEMA
# ==========================================

combined = actual.copy()

# Add weather columns, NULL for indoor sensors
for column in WEATHER_COLUMNS:
    combined[column] = np.nan

# Replace Outside temperatures with Open-Meteo
combined.loc[outside_mask, "temperature_f"] = (
    interpolated["temperature_f"].to_numpy()
)

# Populate weather fields only for Outside
for column in WEATHER_COLUMNS:
    combined.loc[outside_mask, column] = (
        interpolated[column].to_numpy()
    )

# Match the actual database column order
combined = combined[OUTPUT_COLUMNS]


# ==========================================
# 5. SAVE NEW PARQUET
# ==========================================

combined.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)
print("Rows:", len(combined))

print("\nColumns:")
print(combined.columns)

print("\nOutside readings:")
print(
    combined.loc[outside_mask].head().to_string(
        index=False
    )
)

print("\nIndoor readings:")
print(
    combined.loc[~outside_mask].head().to_string(
        index=False
    )
)

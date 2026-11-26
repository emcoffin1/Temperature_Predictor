"""Minimal example for adding the forecast to an existing generator payload."""

from apartment_thermal import predict_from_combined_data


def add_indoor_forecast(
    payload: dict,
    database_rows,
) -> dict:
    """Add the latest re-anchored forecast to a generator payload.

    Call this at the initial daily run and every two hours afterward. The
    caller should persist the initial daily payload separately if full-day
    forecast accuracy will be evaluated.
    """
    forecast = predict_from_combined_data(
        data_or_path=database_rows,
    )
    payload["indoor_temperature"] = forecast.summary()
    hourly = forecast.hourly_curve()
    payload["indoor_temperature_curve"] = [
        {"time": timestamp.isoformat(), "temperature_f": row.predicted_indoor_f}
        for timestamp, row in hourly.iterrows()
    ]
    return payload

# Apartment Thermal Forecast

Physics-based Living Room temperature forecasting for the apartment generator.
The production pipeline combines:

1. a fixed Model C trained from ten weeks of historical data;
2. a bounded batch correction estimated from the previous seven days;
3. a leaky error integrator over the same recent residuals; and
4. two-hour reforecasting anchored to the newest indoor measurement; and
5. future `Outside` rows through local midnight.

The package consumes the same schema as `fake_data_wk46.parquet`. Production
does not contact an external weather provider. The generator/database is the
source of measured indoor values, actual recent conditions, and future rows.

## Intent

The forecast answers:

- What will the Living Room temperature curve look like today?
- What indoor minimum and maximum are expected, and when?
- Has the apartment recently been consistently warmer or cooler than the
  historical model predicts?

This is a predictive effective thermal model. Its time constant and gains
should not yet be interpreted as uniquely identified physical building
properties.

## Historical Model C

The governing equation is:

```text
dTin/dt = (Tout - Tin) / tau
          + k_solar * (R / 100)
          + q0
          + b_batch
          + b_integral
```

Where:

- `Tin` is Living Room temperature in °F.
- `Tout` is the `Outside` temperature in °F.
- `tau` is the effective thermal time constant in hours.
- `R` is shortwave radiation in W/m².
- `k_solar` is solar gain in °F/hr per 100 W/m².
- `q0` is the historical background heat rate in °F/hr.
- `b_batch` is the seven-day batch correction in °F/hr.
- `b_integral` is the leaky residual-integrator correction in °F/hr.

The fixed parameters bundled in the package are:

```text
tau       = 13.8004865160 hr
k_solar   = 0.0678333531 °F/hr per 100 W/m²
q0        = 0.2218550437 °F/hr
```

They were fitted from 63 usable daily windows covering September 7 through
November 15, 2026. They do not update during the daily forecast.

## Seven-Day Batch Correction

For recent day `d`, the historical model is initialized from that day's first
actual Living Room measurement and integrated using actual `Outside`
temperature and solar input. Residuals are:

```text
e(t) = Tpredicted(t) - Tmeasured(t)
```

The optimizer estimates one bounded background correction:

```text
b_raw = argmin Σ wd * robust_loss(e(t; b))
```

Newer days receive exponentially greater weight:

```text
wd = 0.5 ** (age_days / 3)
```

Only half the estimated correction is applied:

```text
b_batch = 0.5 * b_raw
```

Safeguards:

- seven-day lookback;
- three-day weighting half-life;
- robust `soft_l1` loss;
- maximum raw correction of ±0.75 °F/hr;
- minimum three usable daily windows; and
- fallback to the historical model when recent history is insufficient.

`tau` and `k_solar` remain fixed during this correction.

## Leaky Error Integrator

After batch correction, a sequential residual integrator processes the same
recent measurements. Every interval starts from the previous actual indoor
measurement, predicts the next measurement, and updates a heat-rate state:

```text
r(k) = Tmeasured(k) - Tpredicted(k)

b_integral(k+1) = leak ** dt * b_integral(k)
                  + Ki * clip(r(k), -3, 3) * dt
```

Production settings:

```text
Ki                         = 0.02
leak per hour              = 0.98
residual clipping          = ±3 °F
integral correction limit  = ±0.5 °F/hr
maximum accepted data gap  = 2 hr
anti-windup                = enabled
```

The integral state adjusts background heat rate. It does not offset the latest
measured indoor temperature, so every forecast still starts exactly at the
latest real Living Room reading.

The integrator operates independently of the batch correction. This matters
during initial deployment, because the batch estimator needs multiple usable
daily windows while the integrator can begin adapting sooner.

### Short-history bootstrap

Correction is staged according to available overlapping indoor and weather
history:

```text
less than 12 hours  re-anchor only; do not integrate
12 hours onward     bootstrap integrator at 50% gain, capped at +/-0.2 F/hr
batch available     normal integrator gain and +/-0.5 F/hr cap
```

The 12-hour delay prevents a fan, open window, cooking event, or a handful of
noisy readings from producing an aggressive initial correction. Bootstrap
mode is reported as `error_integrator_status="applied_bootstrap"`; after a
successful batch correction it is reported as `applied_after_batch`.

## Validation

All correction modes were tested in a 56-day walk-forward evaluation. Each
target day used only information available before that day.

| Mode | Mean daily RMSE | Median RMSE | 90th percentile |
|---|---:|---:|---:|
| Historical Model C | 1.325 °F | 1.225 °F | 2.397 °F |
| Batch correction | 1.232 °F | **1.142 °F** | 1.973 °F |
| Error integrator | 1.224 °F | 1.160 °F | 1.996 °F |
| Batch + integrator | **1.197 °F** | 1.155 °F | **1.961 °F** |

Batch plus integrator reduced mean RMSE by 0.128 °F and reduced the worst
tested day from 3.340 °F to 2.802 °F. Its paired improvement over the
historical model had `p=0.012`.

Detailed results and visualization:

- `research/results/error_integrator_comparison.csv`
- `research/results/error_integrator_comparison.png`

## Daily and Two-Hour Process

A job run around 1:00 a.m. performs the following:

1. Load the fixed ten-week Model C parameters.
2. Select the latest valid `Living Room` measurement.
3. Select up to seven days of past `Living Room` and `Outside` rows.
4. Estimate the bounded batch correction when enough daily history exists.
5. Run the leaky error integrator independently, using bootstrap limits until
   the batch correction becomes available.
6. Select future `Outside` rows from the latest measurement through local
   midnight.
7. Integrate the corrected ODE over the complete future curve.
8. Return the predicted curve, minimum, maximum, correction diagnostics, and
   explicit fallback status.

The generator should repeat this process every two hours. Every run starts at
the latest actual Living Room temperature, so accumulated forecast offset is
not carried into the new curve. Scheduling belongs to the calling generator or
systemd timer; the library itself does not start background jobs.

For evaluation, retain both outputs:

- the immutable forecast generated near 1:00 a.m.; and
- the latest two-hour updated forecast.

This preserves a fair full-day accuracy measurement while still serving the
best currently available prediction.

The forecast uses the complete time series. It does not reduce the day's input
to only outdoor minimum and maximum temperatures.

## Input Schema

The combined SQL query or Parquet uses the deployed `fake_data` schema:

| Column | Required | Description |
|---|---:|---|
| `recorded_at` | Yes | UTC timestamp |
| `sensor` | Yes | `Living Room`, `Outside`, or another sensor name |
| `temperature_f` | Yes | Indoor or outdoor temperature in °F |
| `valid` | Recommended | Whether the reading should be used |
| `cloud_cover_percent` | No | Retained but not used by production Model C |
| `wind_speed_mph` | No | Retained but not used by production Model C |
| `uv_index` | No | Solar fallback when shortwave is unavailable |
| `relative_humidity_percent` | No | Retained for future development |
| `dew_point_f` | No | Retained for future development |
| `shortwave_radiation_w_m2` | No | Preferred solar input when available |

Weather values belong on rows where `sensor == "Outside"`.

For one combined query/file:

- past `Living Room` rows provide recent indoor history;
- past `Outside` rows provide recent actual conditions;
- future `Outside` rows provide forecast conditions; and
- the newest valid `Living Room` row initializes the forecast.

## Input Fallbacks

Fallbacks are explicit in `forecast_mode` and `input_fallbacks`:

1. `model_c_full`: use `shortwave_radiation_w_m2`.
2. `model_c_uv_proxy`: if shortwave is missing, convert `uv_index` using:

   ```text
   Rproxy = uv_index * 119.915968981 W/m²
   ```

3. `model_c_no_solar`: if shortwave and UV are missing, use zero solar input.
   Recent bias correction is skipped so it cannot absorb missing solar data.
4. `indoor_persistence`: if usable future `Outside.temperature_f` is missing,
   hold the latest indoor temperature constant through midnight.

Incomplete seven-day history disables only recent correction. The fixed
historical Model C remains available.

## Outputs

`forecast.summary()` returns a JSON-compatible dictionary containing:

```json
{
  "generated_from": "2026-11-14T08:59:14+00:00",
  "initial_temperature_f": 70.37,
  "minimum_f": 67.91,
  "minimum_time": "2026-11-14T08:36:28-08:00",
  "maximum_f": 71.84,
  "maximum_time": "2026-11-14T17:06:42-08:00",
  "timezone": "America/Los_Angeles",
  "forecast_mode": "model_c_uv_proxy",
  "input_fallbacks": ["uv_index_converted_to_radiation_proxy"],
  "bias_correction_f_per_hour": 0.0417,
  "bias_status": "applied",
  "error_integrator_correction_f_per_hour": 0.0299,
  "error_integrator_status": "applied_after_batch",
  "integrator_observations_used": 322,
  "integrator_history_hours": 168.0
}
```

`forecast.hourly_curve()` is the generator-facing output. It is indexed by
local hourly clock boundaries and contains:

- `predicted_indoor_f`
- `outdoor_f`

Example:

```text
local_time                 predicted_indoor_f  outdoor_f
2026-11-14 01:00:00-08:00               70.35      61.18
2026-11-14 02:00:00-08:00               69.94      60.54
2026-11-14 03:00:00-08:00               69.56      59.92
```

The CLI writes the same rows to `indoor_forecast.csv`. When
`--summary-output` is supplied, the JSON contains both summary diagnostics and
an `hourly_forecast` array with `local_time`, `predicted_indoor_f`, and
`outdoor_f` for every remaining hour.

`forecast.curve` retains the internal-resolution diagnostic curve, indexed by
UTC `recorded_at`, with:

- `predicted_indoor_f`
- `outdoor_f`

## Configuration

[`apartment_thermal_config.json`](apartment_thermal_config.json) tracks:

- historical constants and fitting bounds;
- training dates and window count;
- batch-correction settings;
- error-integrator gains and limits;
- the deployed Parquet/SQL column mapping;
- schedule and timezone;
- quality requirements;
- fallback behavior; and
- output filenames.

The same configuration and fixed model are bundled in the wheel. Database
credentials are not stored in JSON. If the generator queries SQL directly, it
should read its connection from `APARTMENT_DATA_DB_URL` or its existing secret
management mechanism and pass the resulting DataFrame to this package.

## Generator Integration

```python
from apartment_thermal import predict_from_combined_data

# database_rows contains past measurements and future Outside rows using the
# deployed schema documented above.
forecast = predict_from_combined_data(database_rows)
hourly = forecast.hourly_curve()

payload["indoor_temperature"] = forecast.summary()
payload["indoor_temperature_curve"] = [
    {
        "time": timestamp.isoformat(),
        "temperature_f": row.predicted_indoor_f,
    }
    for timestamp, row in hourly.iterrows()
]
```

A complete example is in `examples/generator_integration.py`.

## Command-Line Forecast

```bash
apartment-thermal forecast \
  --combined-data apartment-current.parquet \
  --config apartment_thermal_config.json \
  --output indoor_forecast.csv \
  --summary-output indoor_forecast.json
```

`indoor_forecast.csv` contains one row per local hour. Add
`--detailed-output indoor_forecast_detailed.csv` only when the internal
integration timestamps are also needed for diagnostics.

Example cron entries that preserve the initial forecast and refresh the live
forecast every two hours:

```cron
0 1 * * * /path/to/.venv/bin/apartment-thermal forecast --combined-data /path/to/apartment-current.parquet --output /path/to/indoor_forecast_01am.csv --summary-output /path/to/indoor_forecast_01am.json
0 1 * * * /path/to/.venv/bin/apartment-thermal forecast --combined-data /path/to/apartment-current.parquet --output /path/to/indoor_forecast_latest.csv --summary-output /path/to/indoor_forecast_latest.json
0 3-23/2 * * * /path/to/.venv/bin/apartment-thermal forecast --combined-data /path/to/apartment-current.parquet --output /path/to/indoor_forecast_latest.csv --summary-output /path/to/indoor_forecast_latest.json
```

## Historical Retraining

The daily job does not change the fixed historical model. Retraining is an
explicit offline operation:

```bash
apartment-thermal train \
  --data combined_training_data.parquet \
  --output newer_parameters.json
```

Use the new parameters without rebuilding the package:

```bash
apartment-thermal forecast \
  --combined-data apartment-current.parquet \
  --model newer_parameters.json
```

Without `--model`, the package uses its bundled ten-week model.

## Installation and Deployment

For a Git-based server deployment, keep this repository beside the calling
application and install it into that application's virtual environment:

```bash
cd ~
git clone https://github.com/emcoffin1/Temperature_Predictor.git
cd ~/ESP_Temp_Display
.venv/bin/pip install -e ../Temperature_Predictor
```

Update it with:

```bash
cd ~/Temperature_Predictor
git pull
sudo systemctl restart temp-display.service
```

The installed package contains only deployable code, the fixed model, and its
default configuration. Local Parquets, research outputs, and development-only
external-weather acquisition scripts are not runtime dependencies.

## Repository Layout

- `apartment_thermal/`: deployable model, corrections, API, and CLI
- `apartment_thermal_config.json`: editable runtime configuration
- `examples/`: generator integration example
- `research/`: model comparisons, backtests, plots, and development-only data
  acquisition
- `temps/`: local development data, excluded from deployment/source control

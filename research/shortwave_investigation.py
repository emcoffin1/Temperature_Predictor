"""Compare UV, total shortwave, and direct radiation as Model C solar inputs.

Each target day is a genuine rolling backtest: parameters are fitted using only
the preceding seven days, then the full target-day indoor curve is predicted
from its first measured temperature. Global horizontal shortwave radiation is
scaled to units of 100 W/m^2 for numerical conditioning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from apartment_temp_pred import fit_model, load_measurements, simulate


MODEL = "C"
RADIATION_COLUMNS = {
    "shortwave": "shortwave_radiation_w_m2",
    "direct_radiation": "direct_radiation_w_m2",
}


def weather_variants(path: Path, sensor: str, interval: str) -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    indoor, uv_weather = load_measurements(path, sensor, interval)
    raw = pd.read_parquet(path)
    missing = set(RADIATION_COLUMNS.values()).difference(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}; rebuild it with combined_data.py")
    raw["recorded_at"] = pd.to_datetime(raw["recorded_at"], utc=True)
    outside = (
        raw.loc[
            raw["sensor"].eq("Outside"),
            ["recorded_at", *RADIATION_COLUMNS.values()],
        ]
        .dropna().sort_values("recorded_at").drop_duplicates("recorded_at")
        .set_index("recorded_at").astype(float)
    )
    variants = {"uv_index": uv_weather}
    for name, column in RADIATION_COLUMNS.items():
        radiation_weather = uv_weather.copy()
        radiation_weather["solar_input"] = np.interp(
            radiation_weather.index.view("int64"), outside.index.view("int64"),
            outside[column].to_numpy(),
        ) / 100.0
        variants[name] = radiation_weather
    return indoor, variants


def score_day(
    indoor: pd.Series,
    weather: pd.DataFrame,
    target: pd.Timestamp,
    history_days: int,
) -> dict[str, float] | None:
    end = target + pd.Timedelta(days=1)
    observed = indoor.loc[(indoor.index >= target) & (indoor.index < end)]
    if len(observed) < 24:
        return None
    training_end = observed.index[0] - pd.Timedelta(microseconds=1)
    training_start = training_end - pd.Timedelta(days=history_days)
    result, windows = fit_model(
        MODEL, indoor, weather, training_start, training_end,
    )
    predicted = simulate(observed.index, observed.iloc[0], weather, MODEL, result.x)
    actual = observed.to_numpy()
    error = predicted - actual
    baseline_error = observed.iloc[0] - actual
    actual_peak = observed.index[np.argmax(actual)]
    predicted_peak = observed.index[np.argmax(predicted)]
    return {
        "rmse_f": np.sqrt(np.mean(error**2)),
        "mae_f": np.mean(np.abs(error)),
        "bias_f": np.mean(error),
        "baseline_rmse_f": np.sqrt(np.mean(baseline_error**2)),
        "minimum_error_f": np.min(predicted) - np.min(actual),
        "maximum_error_f": np.max(predicted) - np.max(actual),
        "peak_time_error_h": (predicted_peak - actual_peak).total_seconds() / 3600.0,
        "tau_h": result.x[0],
        "solar_coefficient": result.x[1],
        "q0_f_per_h": result.x[2],
        "training_windows": windows,
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    indoor, variants = weather_variants(args.data, args.sensor, args.interval)
    first_target = indoor.index.min().ceil("D") + pd.Timedelta(days=args.history_days)
    last_target = indoor.index.max().floor("D")
    targets = pd.date_range(first_target, last_target, freq=f"{args.step_days}D")
    rows = []
    for number, target in enumerate(targets, start=1):
        print(f"Backtest {number}/{len(targets)}: {target.date()}", flush=True)
        for source, weather in variants.items():
            result = score_day(indoor, weather, target, args.history_days)
            if result is not None:
                rows.append({"target_day": target, "solar_source": source, **result})
    scores = pd.DataFrame(rows)
    if scores.empty:
        raise ValueError("No complete target days were available")
    scores.to_csv(args.output, index=False)
    return scores


def report(scores: pd.DataFrame) -> None:
    scores = scores.copy()
    scores["absolute_peak_time_error_h"] = scores["peak_time_error_h"].abs()
    summary = scores.groupby("solar_source").agg(
        days=("target_day", "count"),
        mean_rmse_f=("rmse_f", "mean"),
        median_rmse_f=("rmse_f", "median"),
        mean_mae_f=("mae_f", "mean"),
        mean_bias_f=("bias_f", "mean"),
        mean_minimum_error_f=("minimum_error_f", "mean"),
        mean_maximum_error_f=("maximum_error_f", "mean"),
        mean_peak_time_error_h=("peak_time_error_h", "mean"),
        mean_absolute_peak_time_error_h=("absolute_peak_time_error_h", "mean"),
        tau_mean_h=("tau_h", "mean"),
        tau_std_h=("tau_h", "std"),
        solar_coefficient_mean=("solar_coefficient", "mean"),
        solar_coefficient_std=("solar_coefficient", "std"),
    )
    pivot = scores.pivot(index="target_day", columns="solar_source", values="rmse_f").dropna()
    if "uv_index" in pivot:
        for source in (column for column in pivot if column != "uv_index"):
            differences = pivot[source] - pivot["uv_index"]
            wins = int((differences < 0).sum())
            delta = float(differences.mean())
            test = stats.ttest_rel(pivot[source], pivot["uv_index"])
            confidence = stats.t.interval(
                0.95, len(differences) - 1,
                loc=delta, scale=stats.sem(differences),
            )
            print(
                f"\n{source} wins {wins}/{len(pivot)} paired days; "
                f"mean RMSE change versus UV: {delta:+.3f} F\n"
                f"Paired-test p={test.pvalue:.3f}; 95% interval: "
                f"[{confidence[0]:+.3f}, {confidence[1]:+.3f}] F"
            )
    print("\nRolling daily backtest summary:")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))


def plot_scores(scores: pd.DataFrame, output: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for source, group in scores.groupby("solar_source"):
        group = group.sort_values("target_day")
        ax1.plot(group["target_day"], group["rmse_f"], marker="o", label=source)
        ax2.plot(group["target_day"], group["bias_f"], marker="o", label=source)
    ax1.set(ylabel="Daily RMSE (F)", title="Model C solar-input comparison")
    ax2.set(ylabel="Daily bias (F)", xlabel="Target day (UTC)")
    ax2.axhline(0, color="black", linewidth=0.8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("temps/combined_latest_10_weeks.parquet"),
    )
    parser.add_argument("--sensor", default="Living Room")
    parser.add_argument("--interval", default="30min")
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument(
        "--step-days", type=int, default=7,
        help="Spacing between target days; use 1 for every available day",
    )
    parser.add_argument("--output", type=Path, default=Path("shortwave_investigation.csv"))
    parser.add_argument("--plot", type=Path, default=Path("shortwave_investigation.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = run(args)
    report(scores)
    plot_scores(scores, args.plot)
    print(f"\nDetailed results: {args.output}\nPlot: {args.plot}")


if __name__ == "__main__":
    main()

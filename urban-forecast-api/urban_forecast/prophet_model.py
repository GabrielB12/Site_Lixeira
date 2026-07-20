import logging

import numpy as np
import pandas as pd
from prophet import Prophet

# Prophet/cmdstanpy are chatty by default; keep the walk-forward loop readable.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def compute_prophet(
    df: pd.DataFrame,
    threshold: float = 90,
    max_horizon_hours: float = 200,
    min_observations: int = 10,
):
    """
    Forecast the remaining time until `fill_percent` reaches `threshold`
    using Prophet, fitted on the recent 96h window.

    Follows the same input/output contract as compute_baseline and
    compute_regression so it can be dropped into the same evaluation loop.
    """
    if df.empty:
        return None

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values("created_at")

    cutoff = df["created_at"].max() - pd.Timedelta(hours=96)
    df = df[df["created_at"] >= cutoff]

    if len(df) < min_observations:
        return None

    deltas = df["created_at"].diff().dt.total_seconds().dropna() / 3600
    step_hours = deltas.median()

    if not np.isfinite(step_hours) or step_hours <= 0:
        return None

    last_fill = float(df["fill_percent"].iloc[-1])
    last_time = df["created_at"].iloc[-1]

    if last_fill >= threshold:
        return {
            "current_fill_level": last_fill,
            "average_rate": None,
            "remaining_hours": 0.0,
            "predicted_date": last_time.isoformat(),
        }

    prophet_df = df.rename(
        columns={"created_at": "ds", "fill_percent": "y"}
    )[["ds", "y"]].copy()
    # Prophet does not accept timezone-aware timestamps
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    try:
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
        )
        model.fit(prophet_df)
    except Exception:
        return None

    periods = int(max_horizon_hours / step_hours) + 1

    try:
        future = model.make_future_dataframe(
            periods=periods,
            freq=pd.Timedelta(hours=step_hours),
        )
        forecast = model.predict(future)
    except Exception:
        return None

    forecast = forecast[forecast["ds"] > prophet_df["ds"].max()]
    crossing = forecast[forecast["yhat"] >= threshold]

    if crossing.empty:
        return None

    crossing_time = crossing.iloc[0]["ds"]
    hours = (crossing_time - prophet_df["ds"].max()).total_seconds() / 3600

    return {
        "current_fill_level": last_fill,
        "average_rate": None,
        "remaining_hours": float(hours),
        "predicted_date": (
            last_time + pd.to_timedelta(hours, unit="h")
        ).isoformat(),
    }

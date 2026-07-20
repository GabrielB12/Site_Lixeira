import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def compute_arima(
    df: pd.DataFrame,
    threshold: float = 90,
    order: tuple = (1, 1, 0),
    max_horizon_hours: float = 200,
    min_observations: int = 10,
):
    """
    Forecast the remaining time until `fill_percent` reaches `threshold`
    using an ARIMA model fitted on the recent 96h window.

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

    # ARIMA (as implemented here) assumes roughly evenly spaced samples;
    # we estimate the sampling step from the data instead of hardcoding it.
    deltas = df["created_at"].diff().dt.total_seconds().dropna() / 3600
    step_hours = deltas.median()

    if not np.isfinite(step_hours) or step_hours <= 0:
        return None

    y = df["fill_percent"].values
    last_fill = float(y[-1])
    last_time = df["created_at"].iloc[-1]

    if last_fill >= threshold:
        return {
            "current_fill_level": last_fill,
            "average_rate": None,
            "remaining_hours": 0.0,
            "predicted_date": last_time.isoformat(),
        }

    # Note: for an I(1) series (d=1), ARIMA needs an explicit drift/trend
    # term ("trend='t'"), otherwise multi-step forecasts flatten out at the
    # last observed level and will never cross a threshold above it.
    fitted = None
    for candidate_order, candidate_trend in (
        (order, "t" if order[1] > 0 else "c"),
        ((1, 1, 0), "t"),
        ((1, 0, 0), "c"),
    ):
        try:
            fitted = ARIMA(
                y, order=candidate_order, trend=candidate_trend
            ).fit()
            break
        except Exception:
            continue

    if fitted is None:
        return None

    max_steps = int(max_horizon_hours / step_hours) + 1

    try:
        forecast = fitted.forecast(steps=max_steps)
    except Exception:
        return None

    crossing_step = None
    for i, value in enumerate(forecast, start=1):
        if value >= threshold:
            crossing_step = i
            break

    if crossing_step is None:
        # model never reaches the threshold within the horizon considered
        return None

    hours = crossing_step * step_hours

    return {
        "current_fill_level": last_fill,
        "average_rate": None,
        "remaining_hours": float(hours),
        "predicted_date": (
            last_time + pd.to_timedelta(hours, unit="h")
        ).isoformat(),
    }

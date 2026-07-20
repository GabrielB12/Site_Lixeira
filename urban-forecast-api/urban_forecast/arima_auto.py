import numpy as np
import pandas as pd
import pmdarima as pm


def compute_arima_auto(
    df: pd.DataFrame,
    threshold: float = 90,
    max_horizon_hours: float = 200,
    min_observations: int = 10,
):
    """
    Forecast the remaining time until `fill_percent` reaches `threshold`
    using an automatically order-selected ARIMA model (pmdarima.auto_arima),
    fitted on the recent 96h window.

    This mirrors compute_arima but replaces the fixed, largely untuned
    ARIMA(1,1,0)-with-drift specification with an order search driven by
    AICc, so that the ARIMA benchmark is not artificially handicapped by a
    minimal, hand-picked configuration.

    Follows the same input/output contract as compute_baseline,
    compute_regression and compute_arima so it can be dropped into the same
    evaluation loop.
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

    try:
        fitted = pm.auto_arima(
            y,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None, max_d=2,
            seasonal=False,
            trend="t",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
    except Exception:
        try:
            # Fallback: some short/degenerate series make a trend term
            # non-identifiable; retry without an explicit deterministic trend.
            fitted = pm.auto_arima(
                y,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=None, max_d=2,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
        except Exception:
            return None

    max_steps = int(max_horizon_hours / step_hours) + 1

    try:
        forecast = fitted.predict(n_periods=max_steps)
    except Exception:
        return None

    forecast = np.asarray(forecast)

    crossing_step = None
    for i, value in enumerate(forecast, start=1):
        if value >= threshold:
            crossing_step = i
            break

    if crossing_step is None:
        return None

    hours = crossing_step * step_hours

    return {
        "current_fill_level": last_fill,
        "average_rate": None,
        "remaining_hours": float(hours),
        "predicted_date": (
            last_time + pd.to_timedelta(hours, unit="h")
        ).isoformat(),
        "selected_order": getattr(fitted, "order", None),
    }
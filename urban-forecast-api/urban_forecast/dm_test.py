"""
Diebold-Mariano test for comparing forecast accuracy between two models,
with the Harvey, Leybourne & Newbold (1997) small-sample correction.

References
----------
Diebold FX, Mariano RS. Comparing Predictive Accuracy. Journal of Business &
Economic Statistics. 1995;13(3):253-263.
Harvey D, Leybourne S, Newbold P. Testing the equality of prediction mean
squared errors. International Journal of Forecasting. 1997;13(2):281-291.
"""

import numpy as np
from scipy import stats


def _autocovariance(d, lag):
    d = np.asarray(d)
    n = len(d)
    d_mean = d.mean()
    return np.sum((d[:n - lag] - d_mean) * (d[lag:] - d_mean)) / n


def diebold_mariano(actual, pred1, pred2, h=1, loss="MAE"):
    """
    Diebold-Mariano test comparing forecasts `pred1` and `pred2` against
    `actual`, with the Harvey-Leybourne-Newbold (HLN) small-sample
    correction and a t-distribution reference (df = n - 1), which is the
    standard, widely used variant of the test for finite samples.

    Parameters
    ----------
    actual, pred1, pred2 : array-like, same length, paired observations
    h : forecast horizon used for the Newey-West-style truncation lag
        (lag = h - 1). h=1 assumes approximately one-step-ahead /
        non-overlapping forecasts.
    loss : "MAE" or "MSE"

    Returns
    -------
    dict with DM statistic, HLN-corrected statistic, p-value, and mean loss
    differential (positive => pred1 has larger average loss than pred2,
    i.e. pred2 is more accurate).
    """
    actual = np.asarray(actual, dtype=float)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)

    e1 = actual - pred1
    e2 = actual - pred2

    if loss.upper() == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif loss.upper() == "MSE":
        d = e1 ** 2 - e2 ** 2
    else:
        raise ValueError("loss must be 'MAE' or 'MSE'")

    n = len(d)
    if n < 2:
        return {
            "n": n, "dm_stat": np.nan, "hln_stat": np.nan,
            "p_value": np.nan, "mean_diff": np.nan,
        }

    d_mean = d.mean()
    gamma0 = _autocovariance(d, 0)
    var_d = gamma0
    for lag in range(1, max(h - 1, 0) + 1):
        gamma = _autocovariance(d, lag)
        var_d += 2 * gamma

    var_d = var_d / n

    if var_d <= 0 or not np.isfinite(var_d):
        return {
            "n": n, "dm_stat": np.nan, "hln_stat": np.nan,
            "p_value": np.nan, "mean_diff": d_mean,
        }

    dm_stat = d_mean / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold (1997) small-sample correction
    correction = np.sqrt(
        (n + 1 - 2 * h + (h * (h - 1)) / n) / n
    )
    hln_stat = dm_stat * correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(hln_stat), df=n - 1))

    return {
        "n": n,
        "dm_stat": dm_stat,
        "hln_stat": hln_stat,
        "p_value": p_value,
        "mean_diff": d_mean,
    }
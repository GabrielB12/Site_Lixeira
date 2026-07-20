import itertools
import time

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from urban_forecast.baseline import compute_baseline
from urban_forecast.regression import compute_regression
from urban_forecast.arima_model import compute_arima
from urban_forecast.arima_auto import compute_arima_auto
from urban_forecast.prophet_model import compute_prophet

from urban_forecast.dm_test import diebold_mariano


THRESHOLD = 70

MODELS = {
    "Baseline": compute_baseline,
    "Regression": compute_regression,
    "ARIMA": compute_arima,
    "ARIMA_auto": compute_arima_auto,
    "Prophet": compute_prophet,
}

df = pd.read_csv(
    "experiments/v_distancias_com_fill_rows_100.csv"
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    utc=True
)

df = df.sort_values(["sensor_id", "created_at"])
df = df.drop_duplicates(subset=["sensor_id", "created_at"])

df["time_diff"] = (
    df.groupby("sensor_id")["created_at"]
    .diff()
    .dt.total_seconds()
)

df = df[
    (df["time_diff"].isna()) |
    (df["time_diff"] > 200)
]

print("\nSensores encontrados:")
print(df["sensor_id"].unique())

results = []

t_start = time.time()

for sensor_id in df["sensor_id"].unique():

    sensor_df = df[df["sensor_id"] == sensor_id].copy()
    sensor_df = sensor_df.sort_values("created_at")

    if len(sensor_df) < 8:
        continue

    print(f"\nProcessando {sensor_id} ({len(sensor_df)} leituras)")

    for i in range(5, len(sensor_df) - 1):

        threshold = sensor_df["threshold_percent"].iloc[0]
        train = sensor_df.iloc[:i]
        last_time = train["created_at"].iloc[-1]

        future_rows = sensor_df[
            (sensor_df["created_at"] > last_time) &
            (sensor_df["fill_percent"] >= threshold)
        ]

        if future_rows.empty:
            continue

        real_threshold_time = future_rows.iloc[0]["created_at"]
        real_hours = (
            real_threshold_time - last_time
        ).total_seconds() / 3600

        if real_hours <= 0 or real_hours > 72:
            continue

        for model_name, model_fn in MODELS.items():

            t0 = time.perf_counter()
            try:
                result = model_fn(train, threshold=threshold)
            except Exception as exc:
                print(f"  [{model_name}] failed at i={i}: {exc}")
                continue
            fit_seconds = time.perf_counter() - t0

            if result is None:
                continue

            pred_hours = result["remaining_hours"]

            if pred_hours <= 0:
                continue

            results.append({
                "model": model_name,
                "sensor_id": sensor_id,
                "train_size": i,
                "real_hours": real_hours,
                "predicted_hours": pred_hours,
                "fit_seconds": fit_seconds,
            })

elapsed = time.time() - t_start
print(f"\nTempo total de execucao: {elapsed:.1f}s")

results_df = pd.DataFrame(results)
print("\nTotal de previsoes:")
print(results_df.groupby("model").size())

# =========================
# GLOBAL METRICS
# =========================

summary = []

for model in results_df["model"].unique():

    model_df = results_df[results_df["model"] == model]

    mae = mean_absolute_error(
        model_df["real_hours"], model_df["predicted_hours"]
    )
    rmse = mean_squared_error(
        model_df["real_hours"], model_df["predicted_hours"]
    ) ** 0.5
    mape = mean_absolute_percentage_error(
        model_df["real_hours"], model_df["predicted_hours"]
    )

    summary.append({
        "Model": model,
        "N": len(model_df),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "MeanFitSeconds": model_df["fit_seconds"].mean(),
    })

summary_df = pd.DataFrame(summary)

print("\nMetricas globais (com ARIMA_auto):")
print(summary_df.to_string(index=False))

summary_df.to_csv(
    "experiments/results_with_dm_test_summary.csv",
    index=False
)
results_df.to_csv(
    "experiments/results_with_dm_test_raw.csv",
    index=False
)

# =========================
# DIEBOLD-MARIANO PAIRWISE TESTS
# =========================
# Pair forecasts on (sensor_id, train_size) so that both models in a
# comparison are being scored against exactly the same target observation.

pivot = results_df.pivot_table(
    index=["sensor_id", "train_size", "real_hours"],
    columns="model",
    values="predicted_hours",
    aggfunc="first",
).reset_index()

print(f"\nObservacoes pareadas disponiveis (todas as 5 colunas presentes): "
      f"{pivot.dropna().shape[0]} de {pivot.shape[0]}")

dm_rows = []

model_names = list(MODELS.keys())

for m1, m2 in itertools.combinations(model_names, 2):

    paired = pivot.dropna(subset=[m1, m2, "real_hours"])

    if len(paired) < 5:
        continue

    for loss in ("MAE", "MSE"):
        res = diebold_mariano(
            paired["real_hours"], paired[m1], paired[m2],
            h=1, loss=loss,
        )
        dm_rows.append({
            "model_1": m1,
            "model_2": m2,
            "loss": loss,
            "n_paired": res["n"],
            "mean_loss_diff (m1 - m2)": res["mean_diff"],
            "DM_stat": res["dm_stat"],
            "HLN_stat": res["hln_stat"],
            "p_value": res["p_value"],
            "significant_5pct": (
                res["p_value"] < 0.05 if pd.notna(res["p_value"]) else None
            ),
        })

dm_df = pd.DataFrame(dm_rows)
pd.set_option("display.width", 160)
print("\nTestes de Diebold-Mariano (HLN-corrigido) pareados:")
print(dm_df.to_string(index=False))

dm_df.to_csv("experiments/dm_test_results.csv", index=False)

print("\nOK - resultados salvos em:")
print(" - experiments/results_with_dm_test_summary.csv")
print(" - experiments/results_with_dm_test_raw.csv")
print(" - experiments/dm_test_results.csv")
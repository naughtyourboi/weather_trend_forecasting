from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_CSV_NAMES = [
    "GlobalWeatherRepository.csv",
    "Global Weather Repository.csv",
    "global_weather_repository.csv",
]

TARGET_CANDIDATES = ["temperature_celsius", "temperature_c", "temp_c", "temperature"]
TIME_CANDIDATES = ["last_updated", "lastupdated", "last_update", "date"]
LAT_CANDIDATES = ["latitude", "lat"]
LON_CANDIDATES = ["longitude", "lon", "lng"]
LOCATION_CANDIDATES = ["location_name", "city", "location"]

AIR_QUALITY_CANDIDATES = [
    "air_quality_PM2.5",
    "air_quality_PM10",
    "air_quality_Ozone",
    "air_quality_pm2.5",
    "air_quality_pm10",
    "air_quality_ozone",
]

FEATURE_CANDIDATES = [
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "hour",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm",
    "cloud",
    "cloudcover",
    "visibility_km",
    "uv_index",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "air_quality_Ozone",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def first_existing_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for candidate in candidates:
        resolved = find_column(df, [candidate])
        if resolved and resolved not in cols:
            cols.append(resolved)
    return cols


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def metrics_table(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced Weather Trend Analysis")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the weather CSV file. If omitted, common filenames in the current directory are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for outputs (plots, html, csv summaries).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting Advanced Weather Trend Analysis ===")

    # ------------------------------------------------------------------
    # 1. Load and clean data
    # ------------------------------------------------------------------
    print("\n[1/7] Reading and cleaning data...")
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = None
        for name in DEFAULT_CSV_NAMES:
            candidate = Path(name)
            if candidate.exists():
                csv_path = candidate
                break

    if csv_path is None or not csv_path.exists():
        print(
            "Error: Cannot find the weather CSV file. Provide --csv or place one of the expected filenames in the current directory."
        )
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    time_col = find_column(df, TIME_CANDIDATES)
    target_col = find_column(df, TARGET_CANDIDATES)
    lat_col = find_column(df, LAT_CANDIDATES)
    lon_col = find_column(df, LON_CANDIDATES)
    location_col = find_column(df, LOCATION_CANDIDATES)

    if time_col is None:
        print("Error: Could not find a timestamp column such as 'last_updated'.")
        return
    if target_col is None:
        print("Error: Could not find a temperature target column.")
        return

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, target_col]).copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    # Fill missing values: numeric -> median, categorical -> forward/back fill.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    cat_cols = [c for c in df.columns if c not in numeric_cols and c != time_col]
    if cat_cols:
        df[cat_cols] = df[cat_cols].ffill().bfill()

    # Time features for forecasting / temporal analysis.
    df["year"] = df[time_col].dt.year
    df["month"] = df[time_col].dt.month
    df["day"] = df[time_col].dt.day
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["dayofyear"] = df[time_col].dt.dayofyear

    # ------------------------------------------------------------------
    # 2. Temporal trend plots (uses lastupdated / last_updated)
    # ------------------------------------------------------------------
    print("[2/7] Generating time trend visualizations...")
    temporal_df = df.copy()
    temporal_df["date_only"] = temporal_df[time_col].dt.date
    daily_trend = temporal_df.groupby("date_only", as_index=False)[target_col].mean()
    daily_trend["date_only"] = pd.to_datetime(daily_trend["date_only"])

    plt.figure(figsize=(12, 5))
    plt.plot(daily_trend["date_only"], daily_trend[target_col], linewidth=1.5)
    plt.title("Daily Average Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Temperature (Celsius)")
    plt.grid(alpha=0.2)
    save_plot(out_dir / "temperature_trend.png")

    precip_col = find_column(df, ["precip_mm", "precipitation_mm", "precipitation", "rain_mm"])
    if precip_col is not None:
        daily_precip = temporal_df.groupby("date_only", as_index=False)[precip_col].mean()
        daily_precip["date_only"] = pd.to_datetime(daily_precip["date_only"])
        plt.figure(figsize=(12, 5))
        plt.plot(daily_precip["date_only"], daily_precip[precip_col], linewidth=1.5)
        plt.title("Daily Average Precipitation Trend")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (mm)")
        plt.grid(alpha=0.2)
        save_plot(out_dir / "precipitation_trend.png")

    # ------------------------------------------------------------------
    # 3. Advanced EDA: anomaly detection
    # ------------------------------------------------------------------
    print("[3/7] Performing anomaly detection (Isolation Forest)...")
    anomaly_features = [target_col]
    if precip_col is not None:
        anomaly_features.append(precip_col)
    humidity_col = find_column(df, ["humidity"])
    if humidity_col is not None and humidity_col not in anomaly_features:
        anomaly_features.append(humidity_col)

    anomaly_df = df[anomaly_features].copy()
    anomaly_df = anomaly_df.replace([np.inf, -np.inf], np.nan)
    anomaly_df = anomaly_df.fillna(anomaly_df.median(numeric_only=True))

    anomaly_model = IsolationForest(contamination=0.01, random_state=42)
    df["anomaly"] = anomaly_model.fit_predict(anomaly_df)

    x_col = target_col
    y_col = precip_col if precip_col is not None else target_col
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="anomaly",
        palette={1: "blue", -1: "red"},
        s=18,
        linewidth=0,
    )
    plt.title("Anomaly Detection: Temperature vs Precipitation")
    plt.xlabel("Temperature (Celsius)")
    plt.ylabel("Precipitation (mm)")
    plt.legend(title="Status", labels=["Normal", "Anomaly"])
    save_plot(out_dir / "anomaly_detection.png")

    # ------------------------------------------------------------------
    # 4. Environmental impact / correlation analysis
    # ------------------------------------------------------------------
    print("[4/7] Analyzing environmental impact and air quality correlation...")
    aq_cols = first_existing_columns(df, AIR_QUALITY_CANDIDATES)
    corr_cols = [c for c in aq_cols if c in df.columns]
    for extra in [target_col, humidity_col, find_column(df, ["wind_kph"]), precip_col]:
        if extra is not None and extra not in corr_cols:
            corr_cols.append(extra)

    corr_cols = [c for c in corr_cols if c is not None]
    if len(corr_cols) >= 2:
        corr_matrix = df[corr_cols].corr(numeric_only=True)
        plt.figure(figsize=(11, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Correlation Matrix: Air Quality & Weather Parameters")
        save_plot(out_dir / "environmental_impact.png")
        corr_matrix.to_csv(out_dir / "correlation_matrix.csv")

    # ------------------------------------------------------------------
    # 5. Spatial analysis
    # ------------------------------------------------------------------
    print("[5/7] Generating spatial analysis map...")
    if lat_col is not None and lon_col is not None:
        map_df = df.copy()
        map_df = map_df.dropna(subset=[lat_col, lon_col, target_col])

        fig = px.scatter_geo(
            map_df,
            lat=lat_col,
            lon=lon_col,
            color=target_col,
            hover_name=location_col if location_col is not None else None,
            size_max=12,
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Global Temperature Distribution",
        )
        fig.write_html(out_dir / "spatial_analysis_map.html")

    # ------------------------------------------------------------------
    # 6. Forecasting models with time-aware split
    # ------------------------------------------------------------------
    print("[6/7] Building multiple models and ensemble for temperature prediction...")

    model_features = [c for c in FEATURE_CANDIDATES if c in df.columns]
    # Add any useful numeric columns not already included, but exclude the target and obvious identifiers.
    extra_numeric = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in set(model_features + [target_col, "anomaly", "year"])
        and c not in {"latitude", "longitude"}
        and "fahrenheit" not in c.lower()
        and "feels" not in c.lower()
        and "temp" not in c.lower() 
    ]
    # Keep feature count manageable and exclude leakage-like columns.
    model_features = list(dict.fromkeys(model_features + extra_numeric))

    if len(model_features) < 3:
        print("Warning: Not enough features for modeling. Skipping model training.")
        return

    model_df = df[[time_col, target_col] + model_features].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=[target_col])

    # Time-based split: train / validation / test.
    n = len(model_df)
    if n < 50:
        print("Warning: Dataset is too small for a meaningful time-aware split. Skipping modeling.")
        return

    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    train_df = model_df.iloc[:train_end].copy()
    val_df = model_df.iloc[train_end:val_end].copy()
    test_df = model_df.iloc[val_end:].copy()

    X_train = train_df[model_features]
    y_train = train_df[target_col]
    X_val = val_df[model_features]
    y_val = val_df[target_col]
    X_test = test_df[model_features]
    y_test = test_df[target_col]

    # Impute missing values using only training data.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=model_features, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=model_features, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=model_features, index=X_test.index)

    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    rf_model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2)
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    models = {
        "Linear Regression": lr_pipeline,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
    }

    validation_predictions = {}
    test_predictions = {}
    results = []
    for name, model in models.items():
        model.fit(X_train_imp, y_train)
        val_pred = model.predict(X_val_imp)
        test_pred = model.predict(X_test_imp)
        validation_predictions[name] = val_pred
        test_predictions[name] = test_pred
        scores = metrics_table(y_test, test_pred)
        results.append({"Model": name, **scores})
        print(f"  - {name}: RMSE = {scores['RMSE']:.4f}, MAE = {scores['MAE']:.4f}, R2 = {scores['R2']:.4f}")

    # Weighted ensemble based on validation performance.
    val_scores = {
        name: np.sqrt(mean_squared_error(y_val, pred)) for name, pred in validation_predictions.items()
    }
    inv = {name: 1.0 / (rmse + 1e-8) for name, rmse in val_scores.items()}
    total = sum(inv.values())
    weights = {name: value / total for name, value in inv.items()}

    ensemble_test_pred = np.zeros(len(X_test_imp))
    for name, pred in test_predictions.items():
        ensemble_test_pred += weights[name] * pred

    ensemble_scores = metrics_table(y_test, ensemble_test_pred)
    results.append({"Model": "Weighted Ensemble", **ensemble_scores})
    print(
        f"  - Weighted Ensemble: RMSE = {ensemble_scores['RMSE']:.4f}, MAE = {ensemble_scores['MAE']:.4f}, R2 = {ensemble_scores['R2']:.4f}"
    )

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(out_dir / "model_results.csv", index=False)

    with open(out_dir / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Feature importance
    # ------------------------------------------------------------------
    print("\n[7/7] Calculating feature importance...")
    rf_model.fit(X_train_imp, y_train)
    rf_importance = pd.Series(rf_model.feature_importances_, index=model_features).sort_values(ascending=False)

    plt.figure(figsize=(11, 6))
    rf_importance.head(12).plot(kind="bar")
    plt.title("Feature Importance for Temperature Prediction (Random Forest)")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    save_plot(out_dir / "feature_importance.png")

    # Permutation importance on the hold-out test set to complement model-based importance.
    perm = permutation_importance(
        rf_model,
        X_test_imp,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="r2",
    )
    perm_importance = pd.Series(perm.importances_mean, index=model_features).sort_values(ascending=False)
    plt.figure(figsize=(11, 6))
    perm_importance.head(12).plot(kind="bar")
    plt.title("Permutation Importance for Temperature Prediction")
    plt.ylabel("Mean Importance (R2 decrease)")
    plt.xticks(rotation=45, ha="right")
    save_plot(out_dir / "permutation_importance.png")

    # ------------------------------------------------------------------
    # Summary files for README / report writing
    # ------------------------------------------------------------------
    summary = {
        "rows": int(len(df)),
        "features_used": model_features,
        "time_column": time_col,
        "target_column": target_col,
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "ensemble_weights": weights,
        "best_model": results_df.iloc[0]["Model"],
        "best_rmse": float(results_df.iloc[0]["RMSE"]),
        "best_r2": float(results_df.iloc[0]["R2"]),
    }
    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Analysis Complete! ===")
    print(f"Outputs saved to: {out_dir}")
    print("Generated files:")
    for filename in [
        "temperature_trend.png",
        "precipitation_trend.png",
        "anomaly_detection.png",
        "environmental_impact.png",
        "spatial_analysis_map.html",
        "model_results.csv",
        "feature_importance.png",
        "permutation_importance.png",
        "analysis_summary.json",
    ]:
        path = out_dir / filename
        if path.exists():
            print(f"  - {path.name}")


if __name__ == "__main__":
    main()


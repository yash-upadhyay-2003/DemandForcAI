from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", DATA_DIR / "final_sales_forecasting_model.pkl"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", DATA_DIR / "final_sales_feature_names.pkl"))
HISTORY_PATH = Path(os.getenv("HISTORY_PATH", DATA_DIR / "daily_sales_history.csv"))

EXPECTED_FEATURES = [
    "Lag_1", "Lag_7", "Lag_30",
    "Rolling_Mean_7", "Rolling_Mean_30", "Product_Lag_1"
]
MODEL_NAME = "Tuned Ridge Regression"
TEST_MAE = 2050.81
TEST_RMSE = 3656.24
TEST_R2 = 0.007

app = FastAPI(
    title="Daily Demand Forecasting API",
    description="Daily revenue forecasting API using the exported Tuned Ridge pipeline.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

_model = None
_feature_names = None
_history = None


def load_assets():
    global _model, _feature_names, _history
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
        if not FEATURES_PATH.exists():
            raise RuntimeError(f"Feature-name artifact not found: {FEATURES_PATH}")
        # Only load trusted, locally supplied pickle artifacts.
        with MODEL_PATH.open("rb") as f:
            _model = pickle.load(f)
        with FEATURES_PATH.open("rb") as f:
            _feature_names = pickle.load(f)
        if list(_feature_names) != EXPECTED_FEATURES:
            raise RuntimeError(
                f"Unexpected model feature order. Expected {EXPECTED_FEATURES}; got {_feature_names}"
            )
    if _history is None:
        if not HISTORY_PATH.exists():
            raise RuntimeError(f"Historical data file not found: {HISTORY_PATH}")
        hist = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
        # Accept the notebook's native export name Product_Count and normalize
        # it internally to ProductCount for the existing forecast feature logic.
        if "ProductCount" not in hist.columns and "Product_Count" in hist.columns:
            hist = hist.rename(columns={"Product_Count": "ProductCount"})
        required = {"Date", "Daily_Sales", "ProductCount"}
        if not required.issubset(hist.columns):
            raise RuntimeError(
                "History CSV must contain Date, Daily_Sales, and Product_Count "
                "(or ProductCount). "
                f"Found columns: {hist.columns.tolist()}"
            )
        hist = hist[["Date", "Daily_Sales", "ProductCount"]].copy()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
        hist["Daily_Sales"] = pd.to_numeric(hist["Daily_Sales"], errors="raise").fillna(0.0)
        hist["ProductCount"] = pd.to_numeric(hist["ProductCount"], errors="raise").fillna(0.0)
        hist = hist.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
        expected = pd.date_range(hist["Date"].min(), hist["Date"].max(), freq="D")
        if len(hist) != len(expected) or not hist["Date"].reset_index(drop=True).equals(pd.Series(expected, name="Date")):
            hist = hist.set_index("Date").reindex(expected).rename_axis("Date").reset_index()
            hist["Daily_Sales"] = hist["Daily_Sales"].fillna(0.0)
            hist["ProductCount"] = hist["ProductCount"].fillna(0.0)
        _history = hist
    return _model, _feature_names, _history


def make_forecast(horizon: int) -> pd.DataFrame:
    model, feature_names, hist = load_assets()
    sales_history = hist["Daily_Sales"].astype(float).tolist()
    product_history = hist["ProductCount"].astype(float).tolist()
    active = hist.loc[hist["Daily_Sales"] > 0]
    if not active.empty:
        ratios = active.loc[active["ProductCount"] > 0, "Daily_Sales"] / active.loc[active["ProductCount"] > 0, "ProductCount"]
        sales_per_product = float(ratios.mean()) if not ratios.empty else 1.0
    else:
        sales_per_product = 1.0
    sales_per_product = sales_per_product if np.isfinite(sales_per_product) and sales_per_product > 0 else 1.0

    future_dates = pd.date_range(hist["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    records: list[dict[str, Any]] = []
    for forecast_date in future_dates:
        features = pd.DataFrame([[
            sales_history[-1],
            sales_history[-7],
            sales_history[-30],
            float(np.mean(sales_history[-7:])),
            float(np.mean(sales_history[-30:])),
            product_history[-1],
        ]], columns=feature_names)
        predicted = max(0.0, float(model.predict(features)[0]))
        sales_history.append(predicted)
        product_history.append(predicted / sales_per_product if predicted > 0 else 0.0)
        records.append({"Date": forecast_date, "Forecast_Revenue": predicted})
    return pd.DataFrame(records)


@app.get("/")
def root():
    return {"message": "Daily Demand Forecasting API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    try:
        _, feature_names, hist = load_assets()
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "history_start": hist["Date"].min().date().isoformat(),
            "history_end": hist["Date"].max().date().isoformat(),
            "history_rows": int(len(hist)),
            "feature_count": len(feature_names),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metadata")
def metadata():
    try:
        model, feature_names, hist = load_assets()
        return {
            "model": MODEL_NAME,
            "pipeline_steps": ["StandardScaler", "Ridge(alpha=1000)"],
            "test_mae": TEST_MAE,
            "test_rmse": TEST_RMSE,
            "test_r2": TEST_R2,
            "feature_count": len(feature_names),
            "features": list(feature_names),
            "history_start": hist["Date"].min().date().isoformat(),
            "history_end": hist["Date"].max().date().isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/history")
def get_history():
    try:
        _, _, hist = load_assets()
        out = hist.copy()
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        return out.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/forecast")
def get_forecast(horizon: int = Query(default=90, ge=1, le=90)):
    try:
        forecast = make_forecast(horizon)
        forecast["Date"] = forecast["Date"].dt.strftime("%Y-%m-%d")
        return {
            "horizon": horizon,
            "forecast_start": forecast["Date"].iloc[0],
            "forecast_end": forecast["Date"].iloc[-1],
            "total_forecast_revenue": round(float(forecast["Forecast_Revenue"].sum()), 2),
            "average_daily_forecast_revenue": round(float(forecast["Forecast_Revenue"].mean()), 2),
            "daily_forecast": forecast.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/dashboard")
def dashboard(horizon: int = Query(default=90, ge=1, le=90)):
    try:
        _, _, hist = load_assets()
        forecast = make_forecast(horizon)
        historical = hist[["Date", "Daily_Sales"]].copy()
        historical = historical.rename(columns={"Daily_Sales": "Revenue"})
        historical["Series"] = "Historical"
        future = forecast.rename(columns={"Forecast_Revenue": "Revenue"}).copy()
        future["Series"] = "Forecast"
        combined = pd.concat(
            [historical[["Date", "Revenue", "Series"]], future[["Date", "Revenue", "Series"]]],
            ignore_index=True,
        )
        monthly = future.copy()
        monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
        monthly = monthly.groupby("Month", as_index=False)["Revenue"].sum()
        peak_row = hist.loc[hist["Daily_Sales"].idxmax()]
        return {
            "horizon": horizon,
            "history_start": hist["Date"].min().date().isoformat(),
            "history_end": hist["Date"].max().date().isoformat(),
            "forecast_start": forecast["Date"].min().date().isoformat(),
            "forecast_end": forecast["Date"].max().date().isoformat(),
            "total_historical_revenue": round(float(hist["Daily_Sales"].sum()), 2),
            "average_daily_historical_revenue": round(float(hist["Daily_Sales"].mean()), 2),
            "peak_sales_date": peak_row["Date"].date().isoformat(),
            "peak_sales_revenue": round(float(peak_row["Daily_Sales"]), 2),
            "total_forecast_revenue": round(float(forecast["Forecast_Revenue"].sum()), 2),
            "average_daily_forecast_revenue": round(float(forecast["Forecast_Revenue"].mean()), 2),
            "monthly_forecast": [
                {"Month": r["Month"], "Forecast_Revenue": round(float(r["Revenue"]), 2)}
                for _, r in monthly.iterrows()
            ],
            "chart_data": [
                {
                    "Date": r["Date"].strftime("%Y-%m-%d"),
                    "Revenue": round(float(r["Revenue"]), 4),
                    "Series": r["Series"],
                }
                for _, r in combined.iterrows()
            ],
            "daily_forecast": [
                {
                    "Date": r["Date"].strftime("%Y-%m-%d"),
                    "Forecast_Revenue": round(float(r["Forecast_Revenue"]), 2),
                }
                for _, r in forecast.iterrows()
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

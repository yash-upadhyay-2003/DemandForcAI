# DemandForcAI - Daily Sales Revenue Forecasting

An enterprise-grade, end-to-end Machine Learning pipeline for multi-step recursive daily sales forecasting.

## 📌 Project Overview
This project builds, evaluates, and deploys an automated daily revenue forecasting system for wholesale order transactions. It models historical sales velocity, calendar effects, and distinct product assortment demand lookbacks to recursively forecast 30, 60, and 90-day future revenues.

## 🏆 Champion Model Architecture
- **Algorithm**: Regularized Ridge Regression Pipeline (`StandardScaler` + `Ridge(alpha=1000)`)
- **Validation**: 5-fold walk-forward cross-validation using `TimeSeriesSplit`
- **Test Performance**:
  - **MAE**: $2,050.81
  - **RMSE**: $3,656.24
  - **R² Score**: 0.007
- **Features Used**:
  1. `Lag_1` (Previous day sales)
  2. `Lag_7` (Sales 7 days ago — weekly cycle)
  3. `Lag_30` (Sales 30 days ago — monthly cycle)
  4. `Rolling_Mean_7` (7-day moving average)
  5. `Rolling_Mean_30` (30-day moving average)
  6. `Product_Lag_1` (Previous day unique product assortment count)

## 📂 Repository Contents
- `notebookNB.ipynb`: Master Jupyter notebook following the full 11-stage ML lifecycle (EDA, temporal imputation, lag feature engineering, model benchmarking, hyperparameter tuning, recursive forecasting, and exports).
- `New_Query_2026_10_22_10_01_37.csv`: Historical wholesale sales transaction dataset.
- `data/daily_sales_history.csv`: Continuous daily timeline dataset with historical sales and product demand metrics.
- `data/final_sales_forecasting_model.pkl`: Serialized Tuned Ridge Regression pipeline (`scikit-learn==1.9.1`).
- `data/final_sales_feature_names.pkl`: Ordered feature column list.
- `main.py`: Production FastAPI backend service exposing `/health`, `/api/forecast`, `/api/history`, and `/api/metadata`.
- `streamlit_app.py`: Interactive Streamlit dashboard with 30/60/90 days horizon selector, KPI cards, and charts.
- `Dockerfile`: Production container specification for cloud deployment (Render).
- `requirements.txt`: Pinned dependencies for reproducible deployment.
- `test_smoke.py`: Automated test suite verifying health check, model asset integrity, and forecast generation.

## 🚀 90-Day Recursive Forecast Output
- **Horizon**: 30, 60, or 90 Calendar Days
- **Total Projected Sales (90 Days)**: $127,373.08
- **Average Projected Daily Sales**: $1,415.26

## 🧪 Local Testing & Verification
```bash
# Run automated smoke test suite
pytest test_smoke.py -v

# Run FastAPI backend locally
uvicorn main:app --host 0.0.0.0 --port 8000

# Run Streamlit dashboard locally
streamlit run streamlit_app.py
```

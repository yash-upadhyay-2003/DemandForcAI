# DemandForcAI - Daily Sales Revenue Forecasting

An enterprise-grade, end-to-end Machine Learning pipeline for multi-step recursive daily sales forecasting.

## 📌 Project Overview
This project builds, evaluates, and deploys an automated daily revenue forecasting system for wholesale order transactions. It models historical sales velocity, calendar effects, and distinct product assortment demand lookbacks to recursively forecast 90-day future revenues.

## 🏆 Champion Model Architecture
- **Algorithm**: Regularized Ridge Regression Pipeline (`StandardScaler` + `Ridge(alpha=1000)`)
- **Validation**: 5-fold walk-forward cross-validation using `TimeSeriesSplit`
- **Test Performance**:
  - **MAE**: $2,050.81
  - **RMSE**: $3,656.24
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
- `daily_sales_history.csv`: Continuous daily timeline dataset with historical sales and product demand metrics.
- `final_sales_forecasting_model.pkl`: Serialized Tuned Ridge Regression pipeline.
- `final_sales_feature_names.pkl`: Ordered feature column list.

## 🚀 90-Day Recursive Forecast Output
- **Horizon**: 90 Calendar Days
- **Total Projected Sales**: $127,373.08
- **Average Projected Daily Sales**: $1,415.26

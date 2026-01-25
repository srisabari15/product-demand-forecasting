
import mlflow

# Store experiments locally
mlflow.set_tracking_uri("file:./mlruns")

# Group all runs under one experiment
mlflow.set_experiment("Product_Demand_Forecasting")

print("train.py started")

STEP 1: Imports

import pandas as pd
import joblib
import os
import numpy as np

import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print("All imports successful")

# STEP 2: Load Dataset

DATA_PATH = r"D:\\ML ops\Dataset\\Historical Product Demand.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset loaded")


# STEP 3: Clean Order_Demand column

df["Order_Demand"] = (
    df["Order_Demand"]
    .astype(str)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .astype(int)
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day


# STEP 5: Feature Selection

X = df[
    ["Product_Code", "Warehouse", "Product_Category", "year", "month", "day"]
]
y = df["Order_Demand"]


# STEP 6: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-test split completed")


# STEP 7: Preprocessing Pipeline (NaN SAFE)

categorical_features = [
    "Product_Code",
    "Warehouse",
    "Product_Category"
]

numerical_features = ["year", "month", "day"]

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ]
)


# STEP 8: Model Pipeline

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])

print("Starting model training...")
model.fit(X_train, y_train)
print("Model training finished")


# STEP 9: MLflow Training & Tracking

with mlflow.start_run():

    # -----------------------------
    # Log Parameters
    # -----------------------------
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("encoding", "OneHotEncoder")
    mlflow.log_param("imputation", "median + most_frequent")
    mlflow.log_param("dataset", "Historical Product Demand.csv")

    # -----------------------------
    # Train Model
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    predictions = model.predict(X_test)

    # -----------------------------
    # Evaluate Metrics (VERSION SAFE)
    # -----------------------------
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    medae = median_absolute_error(y_test, predictions)

    n = X_test.shape[0]
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # -----------------------------
    # Log Metrics
    # -----------------------------
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Adjusted_R2", adj_r2)
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("Median_AE", medae)

    # -----------------------------
    # Log Model Artifact
    # -----------------------------
    mlflow.sklearn.log_model(model, artifact_path="model")

    # -----------------------------
    # Save Local Model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/demand_model.pkl")

    # -----------------------------
    # Print Metrics
    # -----------------------------
    print(f"MAE        : {mae:.2f}")
    print(f"RMSE       : {rmse:.2f}")
    print(f"R2         : {r2:.4f}")
    print(f"Adj R2     : {adj_r2:.4f}")
    print(f"MAPE       : {mape:.2f}%")
    print(f"Median AE  : {medae:.2f}")

print("Training completed successfully")

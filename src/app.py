import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----------------------------------
# Paths
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "demand_model.pkl")

model = None  # lazy-loaded

# ----------------------------------
# FastAPI app
# ----------------------------------
app = FastAPI(
    title="Product Demand Forecasting API",
    description="Predict product demand using ML model",
    version="1.0"
)

# ----------------------------------
# Load model safely (NO CI BREAK)
# ----------------------------------
def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model file not found")
        model = joblib.load(MODEL_PATH)

@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except RuntimeError:
        # Allows CI to pass even if model is missing
        pass

# ----------------------------------
# Input Schema
# ----------------------------------
class DemandRequest(BaseModel):
    Product_Code: str
    Warehouse: str
    Product_Category: str
    year: int
    month: int
    day: int

# ----------------------------------
# Health Check
# ----------------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# ----------------------------------
# Prediction Endpoint
# ----------------------------------
@app.post("/predict")
def predict_demand(request: DemandRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train or provide the model."
        )

    input_df = pd.DataFrame([{
        "Product_Code": request.Product_Code,
        "Warehouse": request.Warehouse,
        "Product_Category": request.Product_Category,
        "year": request.year,
        "month": request.month,
        "day": request.day
    }])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_order_demand": int(prediction)
    }

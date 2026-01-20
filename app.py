from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# ----------------------------------
# App
# ----------------------------------
app = FastAPI(
    title="Product Demand Forecasting API",
    version="1.0"
)

# ----------------------------------
# Load model safely
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "demand_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# ----------------------------------
# Input schema
# ----------------------------------
class DemandRequest(BaseModel):
    Product_Code: str
    Warehouse: str
    Product_Category: str
    year: int
    month: int
    day: int

# ----------------------------------
# Routes
# ----------------------------------
@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: DemandRequest):

    df = pd.DataFrame([{
        "Product_Code": data.Product_Code,
        "Warehouse": data.Warehouse,
        "Product_Category": data.Product_Category,
        "year": data.year,
        "month": data.month,
        "day": data.day
    }])

    prediction = model.predict(df)[0]

    return {
        "predicted_order_demand": int(prediction)
    }

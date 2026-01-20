import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "demand_model.pkl")

model = joblib.load(MODEL_PATH)

# ----------------------------------
# FastAPI app
# ----------------------------------
app = FastAPI(
    title="Product Demand Forecasting API",
    description="Predict product demand using ML model",
    version="1.0"
)

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

    # Convert input to DataFrame
    input_df = pd.DataFrame([{
        "Product_Code": request.Product_Code,
        "Warehouse": request.Warehouse,
        "Product_Category": request.Product_Category,
        "year": request.year,
        "month": request.month,
        "day": request.day
    }])

    # Predict
    prediction = model.predict(input_df)[0]

    return {
        "predicted_order_demand": int(prediction)
    }


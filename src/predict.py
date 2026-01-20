import joblib
import pandas as pd

# Load model
model = joblib.load("models/demand_model.pkl")

# Example input
data = {
    "Product_Code": ["Product_001"],
    "Warehouse": ["Whse_A"],
    "Product_Category": ["Category_01"],
    "year": [2025],
    "month": [1],
    "day": [15]
}

df = pd.DataFrame(data)

# Predict
prediction = model.predict(df)

print(f"Predicted Demand: {int(prediction[0])}")

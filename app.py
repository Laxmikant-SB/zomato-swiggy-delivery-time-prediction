from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import joblib
from scripts.data_clean_utils import perform_data_cleaning

# ---------- FastAPI App ----------
app = FastAPI(title="Zomato-Swiggy Delivery Time Prediction")

# ---------- Input Data Schema ----------
class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

# ---------- Load Model & Preprocessor ----------
model_path = "models/delivery_time_pred_model.pkl"
preprocessor_path = "models/preprocessor.joblib"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

model_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', model)
])

# ---------- Home Endpoint ----------
@app.get("/")
def home():
    return {"message": "Welcome to Zomato-Swiggy Delivery Time Prediction App"}

# ---------- Predict Endpoint ----------
@app.post("/predict")
def predict_delivery_time(data: Data):
    df = pd.DataFrame([data.model_dump()])  # Pydantic v2: use model_dump()
    
    # Convert numeric columns safely
    for col in ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    cleaned_data = perform_data_cleaning(df)
    prediction = model_pipe.predict(cleaned_data)[0]
    
    return {"predicted_delivery_time": float(prediction)}

# ---------- Run Server ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

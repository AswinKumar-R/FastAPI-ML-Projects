from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Initialize FastAPI
app = FastAPI()

# Load the Prophet model
with open('TCS_Stock_Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the request body
class PredictionRequest(BaseModel):
    start_date: str  # e.g., "2025-01-01"
    periods: int  # Number of future periods to predict

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Prophet Forecast API!"}

@app.post("/forecast")
async def forecast(request: PredictionRequest):
    try:
        # Generate future dates
        future_dates = pd.date_range(start=request.start_date, periods=request.periods, freq="D").to_frame(name="ds")
        
        # Predict using the loaded model
        forecast = model.predict(future_dates)

        # Extract and return relevant columns
        forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records")
        return {"forecast": forecast_result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from model import load_model, make_predictions

# Create a FastAPI app
app = FastAPI()

# Define the request body model using Pydantic
class FeatureData(BaseModel):
    # Define the feature columns and their data types
   metascore:float
   genres_:bool
   genres_Accounting: bool
   genres_Action: bool
   genres_Adventure: bool
   genres_Animation: bool
   genres_Audio_Production: bool
   genres_Casual: bool
   genres_Design: bool
   genres_Early_Access: bool
   genres_Education: bool
   genres_Free_to_Play: bool
   genres_Indie: bool
   genres_Massively_Multiplayer: bool
   genres_Photo_Editing: bool
   genres_RPG: bool
   genres_Racing: bool
   genres_Simulation:bool
   genres_Software_Training: bool
   genres_Sports: bool
   genres_Strategy: bool
   genres_Utilities: bool
   genres_Video: bool
   genres_Web_Publishing:bool

# Load the trained model
model = load_model('linear_regression_model.pkl')

# Create a POST route for making predictions
@app.post("/predict/")
async def predict_price(data: FeatureData):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Use the trained model to make predictions
        predictions = make_predictions(model, input_data)

        # Return the predictions
        return {"predictions": predictions[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

  
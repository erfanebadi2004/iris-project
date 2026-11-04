import joblib
import pandas as pd
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

try:
    with open("model.joblib", "rb") as classifier:
      model = joblib.load(classifier)
except Exception as e:
  print("=" * 50)
  print("MODEL LOAD FAILED:", repr(e))
  print("=" * 50)
  model = None

app = FastAPI()

class FeatureInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
class BatchPredictionRequest(BaseModel):
    sample: List[FeatureInput]

@app.post("/predict")
def predict_model(request: BatchPredictionRequest):
    
  try:
    df_samples = pd.DataFrame([s.model_dump() for s in request.sample])
    
    prediction = model.predict(df_samples)
    result = prediction.tolist()
    
    return {"prediction": result}
  except Exception as e:
      print(str(e))
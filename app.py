import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_VERSION = "v0.1"
model = joblib.load("pipeline.joblib")
app = FastAPI()

class PatientFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(features: PatientFeatures):
    row = pd.DataFrame([features.model_dump()])
    prediction = model.predict(row)[0]
    return {"prediction": float(prediction)}
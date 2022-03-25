import joblib as jbl

from fastapi import FastAPI
from fastapi import status
from pydantic import BaseModel
import sys

sys.path.append('.')
from src.models.train_model import *

app = FastAPI()

model = jbl.load('./models/quantile_model.joblib')

# schema
class QuantileModel(BaseModel):
    V17: list

@app.get('/')
def root():
    return {'data': 'Credit Card Anomaly Detection API'}

@app.post('/predict', status_code=status.HTTP_200_OK)
def predict(body: QuantileModel):
    body_dict = body.dict()
    data = body_dict['V17']
    predictions = model.predict(data)

    return {'predictions': predictions}

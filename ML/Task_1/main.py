from fastapi import FastAPI, status
from pydantic import BaseModel
from typing import List, Optional
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
import pickle

# Load saved parameters from the pickle file
async def load():
    with open('data/model.pkl', 'rb') as f:
        loaded_params = pickle.load(f)
        return loaded_params['model']

def to_df(item):
    return pd.DataFrame(jsonable_encoder(item))

class Item(BaseModel):
    name: Optional[str] = None
    year: int
    selling_price: Optional[int] = None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class ItemResponse(BaseModel):
    prediction: int
class Items(BaseModel):
    objects: List[Item]

app = FastAPI()
model = None
model = None
# @app.on_event("startup")
# async def startup_event():
#     global model
#     model = await load()

@app.get("/")
def read_root():
    return {"message": 123 }

@app.post("/predict_item", response_model=ItemResponse)
async def predict_item(item: Item) -> float:
    response = {"prediction": 123 }
    return response


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return {"country":"RU","state":"MOW","stateName":"Moscow","continent":"EU"}
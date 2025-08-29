import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI

app = FastAPI()
preprocessor = joblib.load("preprocessor.pkl")
model = tf.keras.models.load_model('model.h5')

@app.get("/health")
async def health():
    return {"status": "OK"}

@app.post("/predict")
async def predict(data:dict):
    instances = data["instances"]
    df = pd.DataFrame(instances)
    X = preprocessor.transform(df)
    prob = model.predict(X).flatten()
    pred = (prob > 0.5).astype(int)
    
    result = []
    for i, inst in enumerate(instances):
        result.append({
            "input" : inst,
            "prediction" : int(pred[i]),
            "probability" : float(prob[i]),
        })
    return {"result":result}
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # 👈 Importante
from pydantic import BaseModel
import joblib
import json
import os
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd


app = FastAPI(
    title="ML Models API",
    description="Serve ML models (Iris, Titanic, Penguins) with metadata and predictions",
    version="1.0.0"
)

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 👈 en producción pon aquí el dominio de tu frontend (ej: "http://localhost:5173")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

AVAILABLE_MODELS = {
    "iris": {
        "model_path": os.path.join(BASE_MODELS_DIR, "iris", "iris_pipeline.joblib"),
        "metadata_path": os.path.join(BASE_MODELS_DIR, "iris", "iris_metadata.json"),
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    },
    "titanic": {
        "model_path": os.path.join(BASE_MODELS_DIR, "titanic", "titanic_pipeline.joblib"),
        "metadata_path": os.path.join(BASE_MODELS_DIR, "titanic", "titanic_metadata.json"),
        "features": ["age", "pclass", "who"]
    },
    "penguins": {
        "model_path": os.path.join(BASE_MODELS_DIR, "penguins", "penguins_pipeline.joblib"),
        "metadata_path": os.path.join(BASE_MODELS_DIR, "penguins", "penguins_metadata.json"),
        "features": []
    }
}

LOADED_MODELS = {}


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PenguinsRequest(BaseModel):
    species: str
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


class TitanicRequest(BaseModel):
    age: float
    pclass: float
    who: str  # "man", "woman", "child"


# ====== Endpoints ======
@app.get("/")
def root():
    return {"message": "Bienvenidos a ML Models API"}


@app.get("/models")
def list_models():
    return {"available_models": list(AVAILABLE_MODELS.keys())}


@app.get("/metadata/{model_name}")
def get_metadata(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    metadata_path = AVAILABLE_MODELS[model_name]["metadata_path"]
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Metadata file not found")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


@app.post("/predict/iris")
def predict_iris(request: IrisRequest):
    iris = load_iris()
    model_name = "iris"
    model_path = AVAILABLE_MODELS[model_name]["model_path"]

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    if model_name not in LOADED_MODELS:
        LOADED_MODELS[model_name] = joblib.load(model_path)

    model = LOADED_MODELS[model_name]
    features = np.array([[request.sepal_length,
                          request.sepal_width,
                          request.petal_length,
                          request.petal_width]])

    try:
        prediction = model.predict(features).tolist()
        probabilities = model.predict_proba(features).tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return {
        "prediction": iris.target_names[prediction][0],
        "probabilities": probabilities
    }


@app.post("/predict/titanic")
def predict_titanic(request: TitanicRequest):
    model_name = "titanic"
    model_path = AVAILABLE_MODELS[model_name]["model_path"]

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    if model_name not in LOADED_MODELS:
        LOADED_MODELS[model_name] = joblib.load(model_path)

    model = LOADED_MODELS[model_name]

    features = pd.DataFrame([{
        "pclass": request.pclass,
        "who": request.who,
        "age": request.age
    }])

    try:
        prediction = model.predict(features).tolist()
        probabilities = model.predict_proba(features).tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return {
        "prediction": int(prediction[0]),  # 0 = murio, 1 = sobrevivio
        "probabilities": probabilities
    }


@app.post("/predict/penguins")
def predict_penguins(request: PenguinsRequest):
    penguins = sns.load_dataset('penguins')
    model_name = "penguins"
    model_path = AVAILABLE_MODELS[model_name]["model_path"]

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    if model_name not in LOADED_MODELS:
        LOADED_MODELS[model_name] = joblib.load(model_path)

    model = LOADED_MODELS[model_name]
    features = pd.DataFrame([{
        "species": request.species,
        "island": request.island,
        "bill_length_mm": request.bill_length_mm,
        "bill_depth_mm": request.bill_depth_mm,
        "flipper_length_mm": request.flipper_length_mm,
        "body_mass_g": request.body_mass_g
    }])

    try:
        prediction = model.predict(features).tolist()
        probabilities = model.predict_proba(features).tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return {
        "prediction": prediction[0],
        "probabilities": probabilities
    }

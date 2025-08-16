from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import os
import numpy as np
from sklearn.datasets import load_iris

app = FastAPI(
    title="ML Models API",
    description="Serve ML models (Iris, Titanic, Penguins) with metadata and predictions",
    version="1.0.0"
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
        "features": []  # definan sus estructiras que neecsiten
    },
    "penguins": {
        "model_path": os.path.join(BASE_MODELS_DIR, "penguins", "penguins_pipeline.joblib"),
        "metadata_path": os.path.join(BASE_MODELS_DIR, "penguins", "penguins_metadata.json"),
        "features": []   # definan sus estructiras que neecsiten
    }
}

LOADED_MODELS = {}

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


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

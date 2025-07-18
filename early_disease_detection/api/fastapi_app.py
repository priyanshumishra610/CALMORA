import bentoml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Disease Predictor API", version="1.0.0")

# Load BentoML Runner
MODEL_NAME = "disease_predictor"
runner = bentoml.sklearn.get("disease_predictor:latest").to_runner()

@app.on_event("startup")
async def startup_event():
    logging.info("Starting BentoML runner...")
    await runner.setup()
    logging.info("BentoML runner started.")

@app.on_event("shutdown")
async def shutdown_event():
    await runner.shutdown()
    logging.info("BentoML runner shut down.")

class PredictRequest(BaseModel):
    data: List[Any]  # List of feature lists

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        logging.info(f"Received prediction request: {request.data}")
        result = await runner.predict.async_run(request.data)
        return {"predictions": result}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/explain")
async def explain(request: PredictRequest):
    # Placeholder for SHAP or other explainability logic
    return {"message": "Explainability endpoint coming soon."}

# Docker/K8s deployment:
# To containerize: `bentoml build` then `docker build -t disease-predictor-api .`
# For K8s: Use BentoML's deployment YAML generator or Helm chart

# Example test request (as comment):
# import requests
# response = requests.post(
#     "http://localhost:8000/predict",
#     json={"data": [[0.1, 0.2, 0.3, 0.4, ...]]}
# )
# print(response.json())

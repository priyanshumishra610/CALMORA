import bentoml
import mlflow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "disease_predictor"
MODEL_STAGE = "Production"  # or use a version tag
BENTO_SERVICE_NAME = "disease_predictor_service"

# Reference for Dockerfile (for containerization)
# To containerize: `bentoml build` then `docker build -t disease-predictor:latest .`
# For K8s: Use BentoML's deployment YAML generator or Helm chart

# Step 1: Load model from MLflow Registry
mlflow.set_tracking_uri("http://localhost:5000")
logging.info(f"Loading model '{MODEL_NAME}' from MLflow Registry (stage: {MODEL_STAGE})...")
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.sklearn.load_model(model_uri)
logging.info("Model loaded from MLflow.")

# Step 2: Save model to BentoML
bento_model = bentoml.sklearn.save_model(
    name=MODEL_NAME,
    model=model,
    signatures={"predict": {"batchable": True}},
    labels={"stage": MODEL_STAGE}
)
logging.info(f"Model saved to BentoML: {bento_model}")

# Step 3: Create BentoML Service (see api/fastapi_app.py for FastAPI integration)
# To build the Bento, run: `bentoml build` in the project root

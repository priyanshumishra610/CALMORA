import os
import logging
from zenml import pipeline, step
from dotenv import load_dotenv
import mlflow
import bentoml

# Load environment variables
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "disease_predictor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
BENTO_SERVICE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/bentos'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

@step
def load_model_from_mlflow_step(model_name: str = MODEL_NAME, model_stage: str = MODEL_STAGE) -> object:
    """
    Load model from MLflow Model Registry.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{model_stage}"
    logging.info(f"Loading model from MLflow Registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logging.info("Model loaded from MLflow.")
    return model

@step
def package_with_bentoml_step(model, model_name: str = MODEL_NAME, model_stage: str = MODEL_STAGE) -> str:
    """
    Package model with BentoML and save service file.
    """
    os.makedirs(BENTO_SERVICE_PATH, exist_ok=True)
    bento_model = bentoml.sklearn.save_model(
        name=model_name,
        model=model,
        signatures={"predict": {"batchable": True}},
        labels={"stage": model_stage}
    )
    logging.info(f"Model saved to BentoML: {bento_model}")
    # Write a bento_service.py for FastAPI integration
    service_code = f'''
import bentoml
from bentoml.io import NumpyNdarray
from fastapi import FastAPI
import numpy as np

bento_model = bentoml.sklearn.get("{model_name}:latest")
runner = bento_model.to_runner()

svc = bentoml.Service("{model_name}_service", runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_arr: np.ndarray) -> np.ndarray:
    return runner.predict.run(input_arr)
'''
    service_path = os.path.join(BENTO_SERVICE_PATH, "bento_service.py")
    with open(service_path, "w") as f:
        f.write(service_code)
    logging.info(f"BentoML service file written to {service_path}")
    # DVC: Run `dvc add {service_path}` to version this file
    return str(bento_model)

@pipeline
def deployment_pipeline(model_name: str = MODEL_NAME, model_stage: str = MODEL_STAGE):
    model = load_model_from_mlflow_step(model_name=model_name, model_stage=model_stage)
    package_with_bentoml_step(model, model_name=model_name, model_stage=model_stage)

# Docker/K8s deployment:
# To containerize: `bentoml build` then `docker build -t disease-predictor:latest .`
# For K8s: Use BentoML's deployment YAML generator or Helm chart

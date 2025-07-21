import os
import logging
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Any, Optional
from dotenv import load_dotenv
import mlflow
import joblib
import numpy as np
import shap
import jwt
from datetime import datetime, timedelta
from api.routes import symptoms
from api.routes import auth

# --- Load environment variables ---
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "disease_predictor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
FASTAPI_SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Auth setup (simple JWT RBAC) ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
USERS = {
    "doctor": {"username": "doctor", "password": "password", "role": "doctor"},
    "patient": {"username": "patient", "password": "password", "role": "patient"}
}

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, FASTAPI_SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, FASTAPI_SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in USERS:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return USERS[username]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

def require_role(role: str):
    def role_checker(user=Depends(get_current_user)):
        if user["role"] != role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker

# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    data: List[List[Any]]

class PredictResponse(BaseModel):
    predictions: List[Any]
    confidences: Optional[List[float]] = None

class ExplainRequest(BaseModel):
    data: List[List[Any]]

class ExplainResponse(BaseModel):
    shap_values: List[List[float]]
    base_values: List[float]
    feature_names: List[str]

# --- FastAPI App ---
app = FastAPI(title="Early Disease Detection API", version="1.0.0")

# Register API routes
app.include_router(symptoms.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")

# --- Model Loading ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
logging.info(f"Loading model from MLflow Registry: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
logging.info("Model loaded from MLflow.")

# Store model in app state for use in routes
app.state.model = model

# --- SHAP Explainer (TreeExplainer for RF/XGBoost) ---
explainer = None
try:
    explainer = shap.TreeExplainer(model)
    logging.info("SHAP explainer initialized.")
except Exception as e:
    logging.warning(f"SHAP explainer could not be initialized: {e}")

# --- Auth Token Endpoint ---
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

# --- Health Endpoint ---
@app.get("/health")
def health():
    try:
        # Check model and DB status (DB check can be added here)
        _ = model.predict(np.zeros((1, model.n_features_in_)))
        return {"status": "ok", "model": "loaded"}
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {"status": "error", "detail": str(e)}

# --- Predict Endpoint ---
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, user=Depends(get_current_user)):
    try:
        X = np.array(request.data)
        preds = model.predict(X)
        confidences = model.predict_proba(X)[:, 1].tolist() if hasattr(model, 'predict_proba') else None
        logging.info(f"Prediction made for user {user['username']}")
        return PredictResponse(predictions=preds.tolist(), confidences=confidences)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Explain Endpoint ---
@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest, user=Depends(require_role("doctor"))):
    if explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not available")
    try:
        X = np.array(request.data)
        shap_values = explainer.shap_values(X)
        base_values = explainer.expected_value.tolist() if hasattr(explainer, 'expected_value') else []
        feature_names = getattr(explainer, 'feature_names', [])
        logging.info(f"SHAP explanation generated for user {user['username']}")
        return ExplainResponse(
            shap_values=shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            base_values=base_values,
            feature_names=feature_names
        )
    except Exception as e:
        logging.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TODO: Add CORS, logging, and exception middleware if needed

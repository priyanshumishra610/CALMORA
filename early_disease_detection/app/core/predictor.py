"""
Predictor: Loads and runs real ML model for risk prediction.
"""
import logging
from typing import List, Any
import numpy as np
import mlflow
import os

class Predictor:
    def __init__(self, model_uri: str = None):
        self.model = None
        if model_uri is None:
            model_uri = os.getenv("MODEL_URI", "models:/disease_predictor/Production")
        try:
            self.model = mlflow.sklearn.load_model(model_uri)
            logging.info(f"Loaded model from MLflow: {model_uri}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def predict_proba(self, features: List[Any]) -> List[float]:
        """
        Predicts risk probabilities for input features.
        Args:
            features (List[Any]): List of input features (single sample).
        Returns:
            List[float]: Probabilities for each class.
        """
        try:
            X = np.array([features])
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                logging.info(f"Predicted probabilities: {probs}")
                return probs.tolist()
            else:
                # Fallback: use predict and return as [1-p, p]
                pred = self.model.predict(X)[0]
                return [1 - pred, pred]
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return [0.0, 0.0] 
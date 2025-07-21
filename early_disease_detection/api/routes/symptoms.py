"""
FastAPI route for Symptom NLP Checker and Risk Prediction.
"""
from fastapi import APIRouter, HTTPException, Request
from app.models.symptoms import SymptomInput, SymptomResponse
from app.core.nlp import SymptomNLP
from app.core.mappings import SymptomMapping
from app.core.panic_guard import PanicGuard
from app.core.explainability import ExplainabilityEngine
from app.core.lifestyle import LifestyleRecommender
from app.core.predictor import Predictor
from app.utils.exception_utils import handle_exception
import logging
from app.services.data_monitor import DataMonitor

router = APIRouter()

# TODO: Load model, mapping, and other dependencies via DI or app state
nlp_engine = SymptomNLP()
mapping_engine = SymptomMapping()
panic_guard = PanicGuard()
explain_engine = ExplainabilityEngine(model=None)  # TODO: Pass actual model
lifestyle_engine = LifestyleRecommender()

predictor = Predictor()  # Loads model from MLflow
data_monitor = DataMonitor()

@router.post("/symptoms", response_model=SymptomResponse)
def check_symptoms(input_data: SymptomInput, request: Request):
    """
    Parses free-text symptoms, predicts risk, explains results, and provides tips.
    """
    try:
        # NLP: Extract symptoms
        symptoms = nlp_engine.extract_symptoms(input_data.text)
        # Mapping: Map symptoms to disease risk
        risk_scores = mapping_engine.map_symptoms(symptoms)
        # Real risk prediction using model
        real_risk = {}
        for disease, score in risk_scores.items():
            # For demo, use mapping score as features; in production, use real features
            proba = predictor.predict_proba([score])
            real_risk[disease] = proba[1]  # Probability of positive class
        risk = [
            {"disease": k, "risk_score": v} for k, v in real_risk.items()
        ]
        # Data Drift Monitoring
        drift_report = data_monitor.check_drift([{k: v for k, v in risk_scores.items()}])
        if drift_report["drift"]:
            logging.warning("Drift detected in /symptoms input!")
        # Use model from app state if available
        model = getattr(request.app.state, "model", None)
        explain_engine = ExplainabilityEngine(model) if model else explain_engine
        # Explainability: Generate SHAP/LIME explanation
        explanation = explain_engine.explain(input_data.text, real_risk)
        # Panic Guard: Generate calm message
        message = panic_guard.rephrase(real_risk)
        # Lifestyle: Recommend tips
        tips = lifestyle_engine.recommend(symptoms, real_risk)
        return SymptomResponse(
            risk=risk,
            message=message,
            shap=explanation,
            lifestyle=[{"tip": t} for t in tips]
        )
    except Exception as e:
        handle_exception(e, context="/symptoms route") 
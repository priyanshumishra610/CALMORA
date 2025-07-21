"""
Tests for ExplainabilityEngine (SHAP/LIME explainability).
"""
import pytest
from app.core.explainability import ExplainabilityEngine

def test_explain():
    engine = ExplainabilityEngine(model=None)
    result = engine.explain("input", "prediction")
    assert "shap_values" in result 
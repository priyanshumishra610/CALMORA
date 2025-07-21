"""
Tests for SymptomMapping (symptom-to-disease mapping).
"""
import pytest
from app.core.mappings import SymptomMapping

def test_map_symptoms():
    mapping = SymptomMapping()
    symptoms = ["fever", "cough"]
    result = mapping.map_symptoms(symptoms)
    assert "flu" in result or "covid" in result 
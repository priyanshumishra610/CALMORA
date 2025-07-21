"""
Test NLP and mapping on realistic symptom sentences.
"""
import pytest
from app.core.nlp import SymptomNLP
from app.core.mappings import SymptomMapping

import os

TEST_INPUTS_DIR = os.path.join(os.path.dirname(__file__), "test_inputs")

@pytest.mark.parametrize("filename,expected", [
    ("flu.txt", ["fever", "cough", "body aches"]),
    ("migraine.txt", ["headache", "nausea"]),
    ("covid.txt", ["cough", "shortness of breath", "taste"]),
    ("healthy.txt", []),
])
def test_nlp_and_mapping(filename, expected):
    nlp = SymptomNLP()
    mapping = SymptomMapping()
    with open(os.path.join(TEST_INPUTS_DIR, filename)) as f:
        text = f.read()
    symptoms = nlp.extract_symptoms(text)
    # At least one expected symptom should be found (except healthy)
    if expected:
        assert any(e in symptoms for e in expected)
        mapped = mapping.map_symptoms(symptoms)
        assert isinstance(mapped, dict)
        assert mapped  # Should map to at least one disease
    else:
        assert symptoms == [] or symptoms == ['no symptoms']
        mapped = mapping.map_symptoms(symptoms)
        assert mapped == {} or not mapped 
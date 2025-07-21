"""
Tests for SymptomNLP (NLP symptom extraction).
"""
import pytest
from app.core.nlp import SymptomNLP

def test_extract_symptoms():
    nlp = SymptomNLP()
    text = "I have a fever and headache."
    result = nlp.extract_symptoms(text)
    assert "fever" in result
    assert "headache" in result 
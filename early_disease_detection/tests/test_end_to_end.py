"""
End-to-end tests for Calmora Symptom Checker API.
"""
import pytest
import requests

API_URL = "http://localhost:8000/api/v1/symptoms"

@pytest.mark.parametrize("input_text", [
    "I have a fever and cough.",
    "Severe headache and nausea.",
    "Just feeling tired.",
    "",  # Edge case: empty input
    "randomtextwithnosymptoms"  # Edge case: no symptoms
])
def test_symptom_checker_end_to_end(input_text):
    response = requests.post(API_URL, json={"text": input_text}, timeout=10)
    assert response.status_code == 200 or response.status_code == 422  # 422 for validation error
    if response.status_code == 200:
        data = response.json()
        assert "risk" in data
        assert "message" in data
        assert "shap" in data
        assert "lifestyle" in data
        # Check calm message
        assert any(word in data["message"].lower() for word in ["calm", "risk", "consult", "advice"])
        # SHAP values should be present or None
        assert "shap_values" in data["shap"] or data["shap"]["shap_values"] is None
        # Tips should be a list
        assert isinstance(data["lifestyle"], list) 
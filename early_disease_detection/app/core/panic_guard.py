"""
Panic Guard: Generates gentle, panic-aware phrasing for risky outputs.
"""

from typing import Any, Dict
import logging

class PanicGuard:
    """
    Generates calm, non-alarming language for risk predictions.
    """
    def __init__(self):
        # TODO: Load templates or config if needed
        pass

    def rephrase(self, risk_output: Dict[str, float]) -> str:
        """
        Returns a gentle, user-friendly message for risky predictions.
        """
        try:
            if not risk_output:
                return "No significant health risks detected. If you feel unwell, consult a healthcare professional."
            max_risk = max(risk_output.values())
            if max_risk < 0.3:
                return "Your symptoms suggest a low risk. Stay hydrated and monitor your health."
            elif max_risk < 0.7:
                return "Some symptoms may indicate a moderate risk. Please consider consulting a healthcare provider if you feel worse."
            else:
                return "Some symptoms may indicate a higher risk. Please remain calm and seek medical advice if you feel unwell."
        except Exception as e:
            logging.error(f"PanicGuard error: {e}")
            return "Unable to assess risk. Please try again later."

# TODO: Add unit tests and error handling 
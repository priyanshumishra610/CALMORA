"""
Lifestyle & Diet Recommender: Rule-based health tips.
"""

from typing import List, Dict
import logging

class LifestyleRecommender:
    """
    Provides personalized lifestyle, diet, and mindfulness tips.
    """
    def __init__(self):
        # TODO: Load rules/config from file or env
        pass

    def recommend(self, symptoms: List[str], risk: Dict[str, float]) -> List[str]:
        """
        Returns a list of health tips based on symptoms and risk.
        """
        try:
            tips = []
            if "fever" in symptoms:
                tips.append("Stay hydrated and rest as much as possible.")
            if "cough" in symptoms:
                tips.append("Use a humidifier and avoid irritants like smoke.")
            if "headache" in symptoms:
                tips.append("Try to rest in a quiet, dark room and stay hydrated.")
            if not tips:
                tips.append("Maintain a balanced diet, exercise regularly, and practice mindfulness.")
            return tips
        except Exception as e:
            logging.error(f"LifestyleRecommender error: {e}")
            return ["Unable to generate tips at this time."]

# TODO: Add unit tests and error handling 
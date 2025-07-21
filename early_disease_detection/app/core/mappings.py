"""
Symptom-to-disease mapping utilities.
"""

import pandas as pd
from typing import Dict, List
import logging
import os

# TODO: Load mapping file path from config/env
MAPPING_CSV_PATH = "ml/symptom_mapping.csv"

class SymptomMapping:
    """
    Loads and queries symptom-to-disease mappings.
    """
    def __init__(self, mapping_csv: str = MAPPING_CSV_PATH):
        if not os.path.exists(mapping_csv):
            logging.error(f"Mapping file not found: {mapping_csv}")
            raise FileNotFoundError(f"Mapping file not found: {mapping_csv}")
        self.df = pd.read_csv(mapping_csv)
        logging.info(f"Loaded symptom mapping from {mapping_csv}")

    def map_symptoms(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Maps symptoms to likely disease classes using the loaded CSV mapping.
        """
        try:
            if not hasattr(self, 'df') or self.df is None:
                logging.error("Mapping DataFrame not loaded.")
                return {}
            # Assume CSV has columns: 'symptom', 'disease', 'weight' (optional)
            scores = {}
            for symptom in symptoms:
                matches = self.df[self.df['symptom'].str.lower() == symptom.lower()]
                for _, row in matches.iterrows():
                    disease = row['disease']
                    weight = row['weight'] if 'weight' in row and not pd.isnull(row['weight']) else 1.0
                    scores[disease] = scores.get(disease, 0) + float(weight)
            # Normalize scores to [0,1]
            if scores:
                max_score = max(scores.values())
                for k in scores:
                    scores[k] = scores[k] / max_score
            logging.info(f"Mapped symptoms to disease scores (CSV): {scores}")
            return scores
        except Exception as e:
            logging.error(f"Mapping error: {e}")
            return {}

# TODO: Add unit tests and error handling 
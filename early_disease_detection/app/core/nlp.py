"""
Symptom NLP module using Hugging Face Transformers (BERT/RoBERTa).
"""

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict
import logging

# TODO: Load model name from config/env
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

class SymptomNLP:
    """
    NLP engine for parsing free-text symptoms using Bio_ClinicalBERT.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        logging.info(f"Loaded Bio_ClinicalBERT NER model: {model_name}")

    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extracts symptoms/medical entities from free-text input using Bio_ClinicalBERT NER.
        """
        try:
            entities = self.ner_pipeline(text)
            # Extract unique entities labeled as symptoms/medical problems
            symptoms = set()
            for ent in entities:
                # Bio_ClinicalBERT may use labels like 'PROBLEM', 'SYMPTOM', etc.
                if ent.get("entity_group", "").lower() in ["problem", "symptom", "disease", "condition"]:
                    symptoms.add(ent["word"].lower())
            logging.info(f"Extracted symptoms/entities: {symptoms}")
            return list(symptoms)
        except Exception as e:
            logging.error(f"NLP extraction error: {e}")
            return []

    def get_embedding(self, text: str):
        """
        Returns embedding for the input text.
        """
        return self.nlp(text)

# TODO: Add unit tests and error handling 
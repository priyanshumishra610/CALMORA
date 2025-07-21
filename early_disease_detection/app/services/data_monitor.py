"""
DataMonitor: Uses Evidently AI to check for input data drift and logs reports.
"""
import logging
from typing import List, Dict, Any
from evidently.report import Report
from evidently.metrics import DataDriftPreset
import pandas as pd
import os
import datetime

class DataMonitor:
    def __init__(self, reference_data_path: str = None, drift_log_path: str = "logs/drift_events.log", drift_threshold: float = 0.5):
        self.reference_data = None
        self.drift_log_path = drift_log_path
        self.drift_threshold = drift_threshold
        if reference_data_path is None:
            reference_data_path = os.getenv("REFERENCE_DATA_PATH", "data/processed/processed_data.csv")
        try:
            self.reference_data = pd.read_csv(reference_data_path)
            logging.info(f"Loaded reference data for drift monitoring: {reference_data_path}")
        except Exception as e:
            logging.error(f"Failed to load reference data: {e}")
            self.reference_data = None

    def log_drift_event(self, drift_score: float, drift_detected: bool, report: dict):
        os.makedirs(os.path.dirname(self.drift_log_path), exist_ok=True)
        with open(self.drift_log_path, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} | Drift Score: {drift_score:.3f} | Drift Detected: {drift_detected}\n")
        # TODO: Integrate with Prometheus or alerting system

    def check_drift(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Checks for data drift using Evidently. Logs and returns drift report.
        Args:
            input_data (List[Dict]): List of input feature dicts (single or batch).
        Returns:
            Dict: Drift report and alert flag.
        """
        try:
            if self.reference_data is None:
                logging.warning("No reference data loaded for drift check.")
                return {"drift": False, "report": None}
            current_df = pd.DataFrame(input_data)
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=self.reference_data, current_data=current_df)
            result = report.as_dict()
            drift_score = result['metrics'][0]['result']['dataset_drift_score']
            drift_detected = drift_score > self.drift_threshold
            self.log_drift_event(drift_score, drift_detected, result)
            if drift_detected:
                logging.warning(f"Data drift detected! Score: {drift_score:.3f} (Threshold: {self.drift_threshold})")
                # TODO: Integrate with Prometheus or alerting system
            else:
                logging.info(f"No data drift detected. Score: {drift_score:.3f}")
            return {"drift": drift_detected, "drift_score": drift_score, "report": result}
        except Exception as e:
            logging.error(f"Drift check error: {e}")
            return {"drift": False, "report": None} 
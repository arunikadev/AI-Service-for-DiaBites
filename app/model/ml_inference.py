import logging
import joblib
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Maps diabetes type string (from API Form) to integer index expected by LightGBM OHE
# Model was trained with dtype integers: 0=type1, 1=type2, 2=gestational/prediabetes
DIABETES_TYPE_MAP = {
    "type1": 0,
    "type2": 1,
    "gestational": 2,
    "prediabetes": 2,
}

class MLInference:
    def __init__(self, weight_path: str):
        self.pipeline = joblib.load(weight_path)

        # Output classes: 0=Caution, 1=Not Recommended, 2=Recommended
        self.classes = {
            0: "Caution",
            1: "Not Recommended",
            2: "Recommended"
        }

    def predict_recommendation(self, user_data: Dict[str, Any], nutrition_data: Dict[str, float]) -> str:
        dt_raw = user_data["diabetes_type"]
        dt_int = DIABETES_TYPE_MAP.get(dt_raw, 1)

        processed_user = {
            "age_group": user_data["age_group"],
            "bmi_category": user_data["bmi_category"],
            "diabetes_type": dt_int,
        }

        combined_data = {**processed_user, **nutrition_data}
        df = pd.DataFrame([combined_data])

        logger.info(f"[ML] Input ke model: {combined_data}")
        logger.info(f"[ML] diabetes_type: '{dt_raw}' → {dt_int}")

        # pipeline.predict() returns string label directly ('Caution'/'Not Recommended'/'Recommended')
        result = str(self.pipeline.predict(df)[0])

        logger.info(f"[ML] Prediksi: '{result}'")
        return result

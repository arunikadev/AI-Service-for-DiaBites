import joblib
import pandas as pd
from typing import Dict, Any

class MLInference:
    def __init__(self, weight_path: str):
        # Load pipeline yang berisi preprocessor dan model LGBM
        self.pipeline = joblib.load(weight_path)
        
        # Kelas rekomendasi berdasarkan training script (0: Caution, 1: Not Rec, 2: Recommended)
        # Urutan ini diambil dari script notebook training.
        self.classes = {
            0: "Caution", 
            1: "Not Recommended", 
            2: "Recommended"
        }
        
    def predict_recommendation(self, user_data: Dict[str, Any], nutrition_data: Dict[str, float]) -> str:
        """
        Menggabungkan data user dan gizi untuk diprediksi LightGBM.
        """
        combined_data = {**user_data, **nutrition_data}
        
        # Jadikan DataFrame dengan 1 baris
        df = pd.DataFrame([combined_data])
        
        # Prediksi
        prediction_index = self.pipeline.predict(df)[0]
        
        # Mengembalikan kelas rekomendasi sebagai string
        return self.classes.get(prediction_index, "Unknown")

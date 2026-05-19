from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os
import io
import numpy as np
from PIL import Image

from app.model.yolo_line import YOLOInference
from app.model.crnn_line import CRNNInference
from app.model.ml_inference import MLInference
from app.utils.nutrition_parser import parse_nutrition

app = FastAPI(title="DiaBites AI-Service")

# GLOBAL MODEL
yolo_model = None
crnn_model = None
ml_model = None

# STARTUP
@app.on_event("startup")
async def startup_event():
    global yolo_model, crnn_model, ml_model
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
    weights_dir = os.path.join(
        base_dir,
        "weights"
    )

    try:
        yolo_model = YOLOInference(
            os.path.join(weights_dir, "best.pt")
        )
        crnn_model = CRNNInference(
            weight_path=os.path.join(
                weights_dir,
                "crnn_model.keras"
            ),
            vocab_path=os.path.join(
                weights_dir,
                "vocab.json"
            )
        )
        ml_model = MLInference(
            os.path.join(weights_dir, "model1.pkl")
        )

        print("Semua model berhasil di-load!")
    except Exception as e:
        print(f"Error loading models: {e}")

# ROOT
@app.get("/")
def read_root():
    return {
        "message": "Welcome to DiaBites AI-Service API"
    }

# PREDICT
@app.post("/predict")
async def predict_nutrition(
    image: UploadFile = File(...),
    age_group: str = Form(...),
    bmi_category: str = Form(...),
    diabetes_type: str = Form(...)
):
    try:

        # READ IMAGE
        image_bytes = await image.read()
        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
        image_np = np.array(pil_image)

        # YOLO LINE DETECTION
        detections = yolo_model.predict_lines(
            image_np
        )
        if not detections:
            return {
                "status": "failed",
                "message": "Tidak ada text line terdeteksi."
            }

        # OCR PER LINE
        full_text_lines = []
        for det in detections:
            line_img = det["crop"]
            raw_text = crnn_model.predict_text(
                line_img
            )
            if raw_text.strip():
                full_text_lines.append(
                    raw_text.strip()
                )
        combined_raw_text = " ".join(
            full_text_lines
        )

        # NUTRITION PARSER
        nutrition_data = parse_nutrition(
            combined_raw_text
        )

        # ML RECOMMENDATION
        user_data = {
            "age_group": age_group,
            "bmi_category": bmi_category,
            "diabetes_type": diabetes_type
        }
        recommendation = ml_model.predict_recommendation(
            user_data,
            nutrition_data
        )

        # RESPONSE
        return {
            "status": "success",
            "lines_detected": len(detections),
            "raw_ocr_text": combined_raw_text,
            "detected_nutrition": nutrition_data,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os
import io
import numpy as np
from PIL import Image

from app.model.yolo_line import YOLOInference
from app.model.crnn_line import CRNNInference
from app.model.ml_inference import MLInference
from app.utils.nutrition_parser import parse_nutrition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DiaBites AI-Service")

# GLOBAL MODEL
yolo_model = None
crnn_model = None
ml_model = None

# STARTUP
@app.on_event("startup")
async def startup_event():
    global yolo_model, crnn_model, ml_model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "weights")

    logger.info(f"Memuat model dari: {weights_dir}")

    try:
        yolo_model = YOLOInference(os.path.join(weights_dir, "29m1.pt"))
        logger.info("YOLO model (best.pt) berhasil dimuat")

        crnn_model = CRNNInference(
            weight_path=os.path.join(weights_dir, "crnn_model-GS.keras"),
            vocab_path=os.path.join(weights_dir, "vocab.json")
        )
        logger.info("CRNN model berhasil dimuat")

        ml_model = MLInference(os.path.join(weights_dir, "model1.pkl"))
        logger.info("LightGBM model berhasil dimuat")

        logger.info("Semua model berhasil di-load dan siap menerima request")
    except Exception as e:
        logger.error(f"GAGAL memuat model: {e}", exc_info=True)

# ROOT
@app.get("/")
def read_root():
    return {"message": "Welcome to DiaBites AI-Service API"}

# HEALTH CHECK — verifikasi semua model loaded
@app.get("/health")
def health_check():
    status = {
        "yolo": yolo_model is not None,
        "crnn": crnn_model is not None,
        "ml":   ml_model is not None,
    }
    all_ready = all(status.values())
    return {
        "status": "ready" if all_ready else "not_ready",
        "models": status,
    }

# PREDICT
@app.post("/predict")
async def predict_nutrition(
    image: UploadFile = File(...),
    age_group: str = Form(...),
    bmi_category: str = Form(...),
    diabetes_type: str = Form(...)
):
    logger.info(
        f"[PREDICT] Request diterima | age_group={age_group} | "
        f"bmi_category={bmi_category} | diabetes_type={diabetes_type} | "
        f"file={image.filename} ({image.content_type})"
    )

    if yolo_model is None or crnn_model is None or ml_model is None:
        logger.error("[PREDICT] Salah satu model belum ter-load!")
        raise HTTPException(status_code=503, detail="Model belum siap, coba beberapa saat lagi")

    try:
        # READ IMAGE
        image_bytes = await image.read()
        # pil_image = Image.open(io.BytesIO(image_bytes)).convert("L").convert("RGB")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)
        logger.info(f"[PREDICT] Gambar dibaca: {pil_image.size} px")

        # YOLO LINE DETECTION
        detections = yolo_model.predict_lines(image_np)
        logger.info(f"[PREDICT] YOLO deteksi {len(detections)} line(s)")

        if not detections:
            logger.warning("[PREDICT] Tidak ada text line terdeteksi")
            return {"status": "failed", "message": "Tidak ada text line terdeteksi."}

        # OCR PER LINE
        full_text_lines = []
        for i, det in enumerate(detections):
            line_img = det["crop"]
            raw_text = crnn_model.predict_text(line_img)
            logger.info(f"[PREDICT] Line {i+1}: '{raw_text.strip()}'")
            if raw_text.strip():
                full_text_lines.append(raw_text.strip())

        combined_raw_text = " ".join(full_text_lines)
        logger.info(f"[PREDICT] Teks OCR gabungan: '{combined_raw_text}'")

        # NUTRITION PARSER
        nutrition_data = parse_nutrition(combined_raw_text)
        logger.info(f"[PREDICT] Hasil parse nutrition: {nutrition_data}")

        # ML RECOMMENDATION
        user_data = {
            "age_group": age_group,
            "bmi_category": bmi_category,
            "diabetes_type": diabetes_type,
        }
        recommendation = ml_model.predict_recommendation(user_data, nutrition_data)
        logger.info(f"[PREDICT] Rekomendasi: {recommendation}")

        # RESPONSE
        return {
            "status": "success",
            "lines_detected": len(detections),
            "raw_ocr_text": combined_raw_text,
            "detected_nutrition": nutrition_data,
            "recommendation": recommendation,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PREDICT] Error tidak terduga: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

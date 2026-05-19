import sys
import os
from PIL import Image
import numpy as np
import io
import json

# Tambahkan direktori 'app' ke sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from model.yolo_line import YOLOInference
from model.crnn_line import CRNNInference
from utils.nutrition_parser import parse_nutrition

# LOAD MODEL
root_dir = os.path.dirname(base_dir)
weights_dir = os.path.join(root_dir, "weights")

print("Loading models...")

try:
    yolo_model = YOLOInference(
        os.path.join(weights_dir, "halo.pt")
    )

    crnn_model = CRNNInference(
        weight_path=os.path.join(weights_dir, "crnn_model.keras"),
        vocab_path=os.path.join(weights_dir, "vocab.json")
    )

    print("[OK] Model berhasil di-load!")

except Exception as e:
    print("[ERROR] Error loading models:", e)
    sys.exit(1)

# LOAD IMAGE

image_filename = "test_image.jpeg"
image_path = os.path.join(os.path.dirname(__file__), image_filename)

if not os.path.exists(image_path):
    print(f"[ERROR] Gambar tidak ditemukan: {image_path}")
    sys.exit(1)

print(f"Memproses gambar: {image_path}")

image = Image.open(image_path).convert("RGB")

# Convert PIL → numpy
image_np = np.array(image)

# YOLO DETECT PER LINE

# diasumsikan return:
# [
#   {
#       "box": [x1, y1, x2, y2],
#       "crop": PIL.Image
#   }
# ]

detections = yolo_model.predict_lines(image_np)

if not detections:
    print("[ERROR] Tidak ada line terdeteksi.")
    sys.exit(1)

# SORT TOP → BOTTOM
detections = sorted(
    detections,
    key=lambda x: x["box"][1]
)

# OCR PER LINE

full_text_lines = []

for det in detections:
    line_img = det["crop"]
    raw_text = crnn_model.predict_text(line_img)
    if raw_text.strip():
        full_text_lines.append(raw_text.strip())

# COMBINE TEXT
combined_raw_text = " ".join(full_text_lines)

print("\n======================================")
print("Teks OCR:")
print(combined_raw_text)
print("======================================")

# PARSE NUTRITION
parsed_data = parse_nutrition(combined_raw_text)

print("[OK] Data Gizi:")
print(json.dumps(parsed_data, indent=4))
print("======================================")